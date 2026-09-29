"""
Main Benchmark Runner and Execution Orchestrator.
"""

import os
import sys
import time
import json
import uuid
import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import yaml

from .schemas import (
    ManifestSample,
    ModelInferenceResult,
    EvaluationSampleRecord,
    ScoreDirection,
    ThresholdMethod,
    SampleStatus,
    PredictionOutcome,
    GroundTruthLabel,
    TimingRecord,
)
from .config import BenchmarkConfig, ModelConfig
from .manifest import BenchmarkManifest
from .validation import BenchmarkValidator, format_validation_errors
from .versioning import build_run_metadata
from .adapters import get_adapter, BaseModelAdapter
from .scoring.scores import RawScoreRecord
from .scoring.threshold import ThresholdEngine
from .scoring.calibration import ThresholdCalibrator, CalibrationResult
from .scoring.alignment import ScoreTableAligner
from .metrics.classification import compute_classification_metrics
from .metrics.biometric import compute_biometric_metrics
from .metrics.pad import compute_pad_metrics
from .metrics.latency import compute_latency_metrics
from .reporting.plots import generate_all_plots
from .reporting.report import BenchmarkReporter


class BenchmarkRunner:
    """Orchestrates reproducible end-to-end benchmarking runs."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config

    def validate(self) -> Tuple[bool, str]:
        """Runs full validation checks on configuration, manifests, and split leakage."""
        errors = BenchmarkValidator.validate_config(self.config)

        test_manifest = None
        calib_manifest = None

        if self.config.data.manifest_path and os.path.exists(self.config.data.manifest_path):
            full_manifest = BenchmarkManifest.from_csv(
                self.config.data.manifest_path,
                base_image_dir=self.config.data.base_image_dir
            )
            manifest_errs = BenchmarkValidator.validate_manifest(
                full_manifest,
                check_images=self.config.data.check_image_existence,
                verify_hashes=self.config.data.verify_hashes,
                allow_missing_images=self.config.data.allow_missing_images,
            )
            errors.extend(manifest_errs)

            # Check splits
            test_manifest = full_manifest.get_split(self.config.data.test_split_name)
            calib_manifest = full_manifest.get_split(self.config.data.calibration_split_name)

        if self.config.data.calibration_manifest_path and os.path.exists(self.config.data.calibration_manifest_path):
            calib_manifest = BenchmarkManifest.from_csv(
                self.config.data.calibration_manifest_path,
                base_image_dir=self.config.data.base_image_dir
            )
            calib_errs = BenchmarkValidator.validate_manifest(
                calib_manifest,
                check_images=self.config.data.check_image_existence,
                verify_hashes=self.config.data.verify_hashes,
                allow_missing_images=self.config.data.allow_missing_images,
            )
            errors.extend(calib_errs)

        if calib_manifest and test_manifest and len(calib_manifest) > 0 and len(test_manifest) > 0:
            leakage_errs = BenchmarkValidator.validate_split_leakage(
                calib_manifest,
                test_manifest,
                verify_hashes=self.config.data.verify_hashes,
            )
            errors.extend(leakage_errs)

        is_valid = len(errors) == 0
        report = format_validation_errors(errors)
        return is_valid, report

    def run(self) -> Dict[str, str]:
        """Executes the complete benchmark workflow for all configured models."""
        # 1. Validation
        is_valid, val_report = self.validate()
        if not is_valid:
            raise ValueError(f"Benchmark validation failed:\n{val_report}")

        # 2. Load Manifests
        test_manifest: Optional[BenchmarkManifest] = None
        calib_manifest: Optional[BenchmarkManifest] = None

        if self.config.data.manifest_path and os.path.exists(self.config.data.manifest_path):
            full_manifest = BenchmarkManifest.from_csv(
                self.config.data.manifest_path,
                base_image_dir=self.config.data.base_image_dir
            )
            t_split = full_manifest.get_split(self.config.data.test_split_name)
            c_split = full_manifest.get_split(self.config.data.calibration_split_name)

            test_manifest = t_split if len(t_split) > 0 else full_manifest
            if len(c_split) > 0:
                calib_manifest = c_split

        if self.config.data.calibration_manifest_path and os.path.exists(self.config.data.calibration_manifest_path):
            calib_manifest = BenchmarkManifest.from_csv(
                self.config.data.calibration_manifest_path,
                base_image_dir=self.config.data.base_image_dir
            )

        # 3. Prepare unique versioned run directory
        run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_id = f"run_{run_timestamp}_{uuid.uuid4().hex[:6]}"
        dataset_ver = self.config.benchmark.dataset_version or "default_dataset"
        run_dir = os.path.join(self.config.outputs.root, dataset_ver, run_id)
        os.makedirs(run_dir, exist_ok=True)

        results_by_signal: Dict[str, List[ModelInferenceResult]] = {}
        run_outputs: Dict[str, str] = {"run_dir": run_dir}

        # 4. Evaluate each model
        for model_cfg in self.config.models:
            print(f"[*] Benchmarking Model: {model_cfg.name} (Adapter: {model_cfg.adapter})...")
            adapter = get_adapter(
                adapter_name=model_cfg.adapter,
                name=model_cfg.name,
                version=model_cfg.version,
                device=model_cfg.device,
                score_direction=self.config.evaluation.score_direction,
                score_csv_path=model_cfg.score_csv_path,
                extra_params=model_cfg.extra_params,
            )

            # Load model
            adapter.load(model_cfg.checkpoint_path or model_cfg.score_csv_path)

            # Warmup
            if self.config.latency.warmup_runs > 0:
                adapter.warmup(self.config.latency.warmup_runs)

            # Calibration step if calibration manifest available
            calibration_result: Optional[CalibrationResult] = None
            calib_scores: List[float] = []
            calib_gts: List[str] = []

            if calib_manifest and len(calib_manifest) > 0:
                print(f"[*] Running Calibration on {len(calib_manifest)} samples...")
                for sample in calib_manifest:
                    res = adapter.predict(sample)
                    if res.status == SampleStatus.PASS and res.raw_score is not None:
                        calib_scores.append(res.raw_score)
                        calib_gts.append(sample.ground_truth)

                if calib_scores:
                    calibration_result = ThresholdCalibrator.calibrate(
                        scores=calib_scores,
                        ground_truths=calib_gts,
                        method=self.config.evaluation.threshold_method,
                        score_direction=self.config.evaluation.score_direction,
                        target_far=self.config.evaluation.target_far,
                        target_frr=self.config.evaluation.target_frr,
                        manual_threshold=self.config.evaluation.manual_threshold,
                    )
                    selected_threshold = calibration_result.selected_threshold
                else:
                    selected_threshold = self.config.evaluation.manual_threshold or 0.5
            else:
                # No calibration set provided; use manual threshold or default
                if self.config.evaluation.manual_threshold is not None:
                    selected_threshold = float(self.config.evaluation.manual_threshold)
                else:
                    selected_threshold = 0.5

            # Test set evaluation
            sample_list = test_manifest.samples if test_manifest else []
            if not sample_list and model_cfg.score_csv_path:
                # Build mock samples from score CSV keys if manifest omitted
                adapter_score_map = getattr(adapter, "score_map", {})
                sample_list = [
                    ManifestSample(sample_id=sid, ground_truth="live", image_path="")
                    for sid in adapter_score_map.keys()
                ]

            inference_results: List[ModelInferenceResult] = []
            timing_records: List[TimingRecord] = []
            evaluation_records: List[EvaluationSampleRecord] = []

            for s_idx, sample in enumerate(sample_list):
                inf_res = adapter.predict(sample)
                inference_results.append(inf_res)
                timing_records.append(inf_res.timing)

                # Apply threshold
                pred_outcome = ThresholdEngine.decide(
                    score=inf_res.raw_score,
                    threshold=selected_threshold,
                    score_direction=self.config.evaluation.score_direction,
                    status=inf_res.status,
                    treat_error_as_reject=self.config.evaluation.treat_error_as_reject,
                )

                is_pos_gt = sample.is_positive_ground_truth()
                is_accept_pred = (pred_outcome == PredictionOutcome.ACCEPT)
                is_correct = (is_pos_gt == is_accept_pred) if pred_outcome != PredictionOutcome.ERROR else None
                is_fa = (not is_pos_gt) and is_accept_pred
                is_fr = is_pos_gt and (not is_accept_pred)

                eval_rec = EvaluationSampleRecord(
                    sample_id=sample.sample_id,
                    ground_truth=sample.ground_truth,
                    normalized_ground_truth="live" if is_pos_gt else "spoof",
                    model_name=model_cfg.name,
                    model_version=model_cfg.version,
                    raw_score=inf_res.raw_score,
                    score_direction=self.config.evaluation.score_direction.value,
                    threshold=selected_threshold,
                    prediction=pred_outcome.value,
                    status=inf_res.status.value,
                    is_correct=is_correct,
                    is_false_accept=is_fa,
                    is_false_reject=is_fr,
                    total_latency_ms=inf_res.timing.total_latency_ms,
                    preprocess_latency_ms=inf_res.timing.preprocess_latency_ms,
                    inference_latency_ms=inf_res.timing.inference_latency_ms,
                    postprocess_latency_ms=inf_res.timing.postprocess_latency_ms,
                    network_latency_ms=inf_res.timing.network_latency_ms,
                    error_type=inf_res.error_type,
                    error_message=inf_res.error_message,
                    subject_id=sample.subject_id,
                    scenario=sample.scenario,
                    device=sample.device,
                    spoof_type=sample.spoof_type,
                    material=sample.material,
                    split=sample.split,
                )
                evaluation_records.append(eval_rec)

            results_by_signal[model_cfg.name] = inference_results

            # 5. Calculate Metrics
            gts = [r.ground_truth for r in evaluation_records]
            preds = [r.prediction for r in evaluation_records]
            scores = [r.raw_score for r in evaluation_records]
            stypes = [r.spoof_type for r in evaluation_records]
            latencies = [r.total_latency_ms for r in evaluation_records]

            cls_metrics = compute_classification_metrics(
                ground_truths=gts,
                predictions=preds,
                scores=scores,
                score_direction=self.config.evaluation.score_direction,
            )
            bio_metrics = compute_biometric_metrics(
                ground_truths=gts,
                predictions=preds,
                scores=scores,
                score_direction=self.config.evaluation.score_direction,
            )
            pad_metrics = compute_pad_metrics(
                ground_truths=gts,
                predictions=preds,
                spoof_types=stypes,
            )
            lat_metrics = compute_latency_metrics(
                timing_records=timing_records,
                thresholds_ms=self.config.latency.thresholds_ms,
            )

            # 6. Threshold Sweep Analysis
            valid_sweep_scores = [s for s in scores if s is not None]
            valid_sweep_gts = [gts[i] for i, s in enumerate(scores) if s is not None]
            sweep_rows = ThresholdEngine.sweep(
                scores=valid_sweep_scores,
                ground_truths=valid_sweep_gts,
                score_direction=self.config.evaluation.score_direction,
                steps=self.config.evaluation.threshold_sweep_steps,
            )

            # 7. Generate Diagnostic Plots
            plot_files = {}
            if self.config.outputs.export_plots and valid_sweep_scores:
                plot_dir = os.path.join(run_dir, "plots")
                plot_files = generate_all_plots(
                    scores=valid_sweep_scores,
                    ground_truths=valid_sweep_gts,
                    latencies_ms=latencies,
                    output_dir=plot_dir,
                    score_direction=self.config.evaluation.score_direction,
                    threshold=selected_threshold,
                    eer=bio_metrics.eer,
                    target_far=self.config.evaluation.target_far,
                    p50_ms=lat_metrics.warm_median_p50_ms,
                    p95_ms=lat_metrics.warm_p95_ms,
                    p99_ms=lat_metrics.warm_p99_ms,
                )

            # 8. Hard cases extraction
            hard_cases = BenchmarkReporter.extract_hard_cases(
                records=evaluation_records,
                score_direction=self.config.evaluation.score_direction,
                top_k=self.config.outputs.top_k_hard_cases,
            )

            # 9. Assemble and save artifacts
            metadata = build_run_metadata(
                config=self.config,
                manifest=test_manifest,
                calibration_manifest=calib_manifest,
                model_name=model_cfg.name,
                model_version=model_cfg.version,
            )

            # Save metadata.json
            with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Save config.yaml
            with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
                yaml.dump(self.config.raw_dict, f, default_flow_style=False)

            # Save predictions.csv
            pred_df = pd.DataFrame([r.__dict__ for r in evaluation_records])
            pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

            # Save raw_scores.csv
            raw_scores_df = pred_df[["sample_id", "ground_truth", "raw_score", "total_latency_ms", "status"]]
            raw_scores_df.to_csv(os.path.join(run_dir, "raw_scores.csv"), index=False)

            # Save threshold_analysis.csv
            if sweep_rows:
                pd.DataFrame(sweep_rows).to_csv(os.path.join(run_dir, "threshold_analysis.csv"), index=False)

            # Save metrics.json
            combined_metrics = {
                "classification": cls_metrics.to_dict(),
                "biometric": bio_metrics.to_dict(),
                "pad": pad_metrics.to_dict(),
                "latency": lat_metrics.to_dict(),
                "selected_threshold": selected_threshold,
            }
            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(combined_metrics, f, indent=2)

            # Save latency.csv
            pred_df[["sample_id", "total_latency_ms", "preprocess_latency_ms", "inference_latency_ms", "postprocess_latency_ms"]].to_csv(
                os.path.join(run_dir, "latency.csv"), index=False
            )

            # Generate Reports
            report_md_path = os.path.join(run_dir, "report.md")
            BenchmarkReporter.generate_markdown_report(
                metadata=metadata,
                cls_metrics=cls_metrics,
                bio_metrics=bio_metrics,
                pad_metrics=pad_metrics,
                lat_metrics=lat_metrics,
                calibration_result=calibration_result,
                hard_cases=hard_cases,
                plots=plot_files,
                output_path=report_md_path,
            )

            if self.config.outputs.generate_html:
                report_html_path = os.path.join(run_dir, "report.html")
                BenchmarkReporter.generate_html_report(
                    metadata=metadata,
                    cls_metrics=cls_metrics,
                    bio_metrics=bio_metrics,
                    pad_metrics=pad_metrics,
                    lat_metrics=lat_metrics,
                    calibration_result=calibration_result,
                    hard_cases=hard_cases,
                    plots=plot_files,
                    output_path=report_html_path,
                )

        # 10. Multi-signal alignment & fusion table export
        if self.config.outputs.export_fusion_table and len(results_by_signal) > 1:
            joined_table = ScoreTableAligner.align_results(
                manifest=test_manifest.samples if test_manifest else None,
                results_by_signal=results_by_signal,
            )
            fusion_path = os.path.join(run_dir, "fusion_scores.csv")
            ScoreTableAligner.export_fusion_table(joined_table, fusion_path)
            run_outputs["fusion_scores"] = fusion_path

        run_outputs["report_md"] = os.path.join(run_dir, "report.md")
        return run_outputs
