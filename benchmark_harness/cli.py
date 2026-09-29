"""
Command-line interface for YellowSense Contactless Fingerprint Benchmark Harness.
"""

import sys
import os
import argparse
from typing import List, Optional

from .config import load_config
from .runner import BenchmarkRunner
from .manifest import BenchmarkManifest
from .validation import BenchmarkValidator, format_validation_errors
from .scoring.threshold import ThresholdEngine
from .scoring.calibration import ThresholdCalibrator
from .scoring.alignment import ScoreTableAligner
from .reporting.report import BenchmarkReporter
from .adapters import get_adapter
from .schemas import ScoreDirection, ThresholdMethod, SampleStatus, ManifestSample


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_harness",
        description="YellowSense UIDAI Contactless Fingerprint Benchmark & Testing Harness",
    )
    subparsers = parser.add_subparsers(dest="command", help="Benchmark command to execute")

    # Command: validate
    val_p = subparsers.add_parser("validate", help="Validate benchmark configuration, manifest, and split leakage")
    val_p.add_argument("--config", "-c", type=str, required=True, help="Path to benchmark YAML/JSON config")

    # Command: run
    run_p = subparsers.add_parser("run", help="Execute complete benchmark run")
    run_p.add_argument("--config", "-c", type=str, required=True, help="Path to benchmark YAML/JSON config")

    # Command: threshold
    th_p = subparsers.add_parser("threshold", help="Run threshold sweep & calibration analysis")
    th_p.add_argument("--config", "-c", type=str, required=True, help="Path to benchmark YAML/JSON config")

    # Command: compare
    comp_p = subparsers.add_parser("compare", help="Compare metrics across historical benchmark runs")
    comp_p.add_argument("--runs", "-r", nargs="+", required=True, help="List of benchmark run directories")
    comp_p.add_argument("--output", "-o", type=str, default=None, help="Path to save comparison markdown summary")

    # Command: join-scores
    join_p = subparsers.add_parser("join-scores", help="Strictly align multiple score CSVs by sample_id for fusion")
    join_p.add_argument("--manifest", "-m", type=str, required=False, help="Path to reference manifest CSV")
    join_p.add_argument("--signals", "-s", nargs="+", required=True, help="Key=Path pairs, e.g. dualscale=ds.csv model3d=3d.csv flash=flash.csv")
    join_p.add_argument("--output", "-o", type=str, required=True, help="Output path for joined fusion CSV")

    return parser


def handle_validate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    runner = BenchmarkRunner(config)
    is_valid, report = runner.validate()
    print("=" * 60)
    print("YellowSense Benchmark Validation Report")
    print("=" * 60)
    print(report)
    print("=" * 60)
    return 0 if is_valid else 1


def handle_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    runner = BenchmarkRunner(config)
    try:
        outputs = runner.run()
        print("\n" + "=" * 60)
        print(" Benchmark Run Completed Successfully!")
        print("=" * 60)
        print(f"[*] Run Artifacts Directory: {outputs.get('run_dir')}")
        print(f"[*] Summary Markdown Report: {outputs.get('report_md')}")
        if "fusion_scores" in outputs:
            print(f"[*] Aligned Fusion Table   : {outputs.get('fusion_scores')}")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n[!] Benchmark execution failed: {str(e)}", file=sys.stderr)
        return 1


def handle_threshold(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if not config.data.manifest_path or not os.path.exists(config.data.manifest_path):
        print(f"[!] Manifest not found: {config.data.manifest_path}")
        return 1

    manifest = BenchmarkManifest.from_csv(config.data.manifest_path, base_image_dir=config.data.base_image_dir)
    calib = manifest.get_split(config.data.calibration_split_name)
    target_manifest = calib if len(calib) > 0 else manifest

    model_cfg = config.models[0]
    adapter = get_adapter(
        adapter_name=model_cfg.adapter,
        name=model_cfg.name,
        version=model_cfg.version,
        score_direction=config.evaluation.score_direction,
        score_csv_path=model_cfg.score_csv_path,
        extra_params=model_cfg.extra_params,
    )
    adapter.load(model_cfg.checkpoint_path or model_cfg.score_csv_path)

    scores = []
    gts = []
    for s in target_manifest:
        res = adapter.predict(s)
        if res.status == SampleStatus.PASS and res.raw_score is not None:
            scores.append(res.raw_score)
            gts.append(s.ground_truth)

    if not scores:
        print("[!] No valid scores collected for threshold analysis.")
        return 1

    print("=" * 60)
    print("Threshold Calibration Operating Points")
    print("=" * 60)

    # 1. EER
    eer_calib = ThresholdCalibrator.calibrate(scores, gts, method=ThresholdMethod.EER, score_direction=config.evaluation.score_direction)
    print(f"[*] EER Operating Point:")
    print(f"    Threshold: {eer_calib.selected_threshold:.4f} | FAR: {eer_calib.calib_far*100:.3f}% | FRR: {eer_calib.calib_frr*100:.3f}% | TAR: {eer_calib.calib_tar*100:.2f}%")

    # 2. Target FAR <= 1.0%
    far1_calib = ThresholdCalibrator.calibrate(scores, gts, method=ThresholdMethod.TARGET_FAR, target_far=0.01, score_direction=config.evaluation.score_direction)
    print(f"[*] Target FAR <= 1.0% Operating Point:")
    print(f"    Threshold: {far1_calib.selected_threshold:.4f} | FAR: {far1_calib.calib_far*100:.3f}% | FRR: {far1_calib.calib_frr*100:.3f}% | TAR: {far1_calib.calib_tar*100:.2f}%")

    # 3. Target FAR <= 0.1%
    far01_calib = ThresholdCalibrator.calibrate(scores, gts, method=ThresholdMethod.TARGET_FAR, target_far=0.001, score_direction=config.evaluation.score_direction)
    print(f"[*] Target FAR <= 0.1% Operating Point:")
    print(f"    Threshold: {far01_calib.selected_threshold:.4f} | FAR: {far01_calib.calib_far*100:.3f}% | FRR: {far01_calib.calib_frr*100:.3f}% | TAR: {far01_calib.calib_tar*100:.2f}%")

    # 4. Min Balanced Error
    ber_calib = ThresholdCalibrator.calibrate(scores, gts, method=ThresholdMethod.MIN_BALANCED_ERROR, score_direction=config.evaluation.score_direction)
    print(f"[*] Minimum Balanced Error Operating Point:")
    print(f"    Threshold: {ber_calib.selected_threshold:.4f} | ACER: {ber_calib.calib_acer*100:.3f}% | FAR: {ber_calib.calib_far*100:.3f}% | FRR: {ber_calib.calib_frr*100:.3f}%")
    print("=" * 60)
    return 0


def handle_compare(args: argparse.Namespace) -> int:
    df = BenchmarkReporter.compare_runs(args.runs, output_markdown_path=args.output)
    print("=" * 60)
    print("Benchmark Historical Runs Comparison")
    print("=" * 60)
    print(df.to_string(index=False) if not df.empty else "No valid run directories found.")
    print("=" * 60)
    if args.output:
        print(f"[*] Saved comparison to: {args.output}")
    return 0


def handle_join_scores(args: argparse.Namespace) -> int:
    manifest_samples = None
    if args.manifest and os.path.exists(args.manifest):
        bm = BenchmarkManifest.from_csv(args.manifest)
        manifest_samples = bm.samples

    results_by_signal = {}
    for item in args.signals:
        if "=" not in item:
            print(f"[!] Invalid signal spec '{item}'. Expected format: name=path.csv")
            return 1
        name, csv_path = item.split("=", 1)
        adapter = get_adapter(adapter_name="score_csv", name=name, score_csv_path=csv_path)
        adapter.load()
        score_map = getattr(adapter, "score_map", {})
        res_list = []
        for sid, (score, lat, row) in score_map.items():
            sample_obj = ManifestSample(sample_id=sid, ground_truth="live", image_path="")
            res_list.append(adapter.predict(sample_obj))
        results_by_signal[name] = res_list

    joined = ScoreTableAligner.align_results(manifest=manifest_samples, results_by_signal=results_by_signal)
    ScoreTableAligner.export_fusion_table(joined, args.output)

    print("=" * 60)
    print(f"Scores Successfully Joined ({joined.total_samples} samples across {len(joined.signal_names)} signals)")
    print("=" * 60)
    print(f"[*] Fully Aligned: {joined.is_fully_aligned}")
    for sig, missing in joined.missing_samples_by_signal.items():
        if missing:
            print(f"[!] Signal '{sig}' missing {len(missing)} samples: {missing[:5]}...")
    print(f"[*] Exported to: {args.output}")
    print("=" * 60)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "validate":
        return handle_validate(args)
    elif args.command == "run":
        return handle_run(args)
    elif args.command == "threshold":
        return handle_threshold(args)
    elif args.command == "compare":
        return handle_compare(args)
    elif args.command == "join-scores":
        return handle_join_scores(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
