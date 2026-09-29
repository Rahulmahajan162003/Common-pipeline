"""
Markdown and HTML benchmark report generator with hard-case diagnostic analysis.
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional
import pandas as pd

from ..schemas import EvaluationSampleRecord, ScoreDirection, ThresholdMethod
from ..metrics.classification import BinaryClassificationMetrics
from ..metrics.biometric import BiometricMetrics
from ..metrics.pad import PADMetrics
from ..metrics.latency import LatencyMetrics
from ..scoring.calibration import CalibrationResult


class BenchmarkReporter:
    """Generates Markdown and HTML benchmark reports."""

    @staticmethod
    def extract_hard_cases(
        records: List[EvaluationSampleRecord],
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        top_k: int = 15,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extracts high-confidence false accepts, false rejects, slowest samples, and errors."""
        false_accepts = []
        false_rejects = []
        slowest_samples = []
        failed_samples = []

        for r in records:
            # Latency tracking
            slowest_samples.append({
                "sample_id": r.sample_id,
                "ground_truth": r.ground_truth,
                "total_latency_ms": r.total_latency_ms,
                "raw_score": r.raw_score,
                "prediction": r.prediction,
                "scenario": r.scenario or "N/A",
                "device": r.device or "N/A",
            })

            if r.status != "PASS" or r.prediction == "error":
                failed_samples.append({
                    "sample_id": r.sample_id,
                    "ground_truth": r.ground_truth,
                    "status": r.status,
                    "error_type": r.error_type or "UnknownError",
                    "error_message": r.error_message or "N/A",
                    "total_latency_ms": r.total_latency_ms,
                })
                continue

            if r.is_false_accept and r.raw_score is not None:
                false_accepts.append({
                    "sample_id": r.sample_id,
                    "ground_truth": r.ground_truth,
                    "raw_score": r.raw_score,
                    "threshold": r.threshold,
                    "prediction": r.prediction,
                    "spoof_type": r.spoof_type or "N/A",
                    "material": r.material or "N/A",
                    "device": r.device or "N/A",
                    "scenario": r.scenario or "N/A",
                    "total_latency_ms": r.total_latency_ms,
                })

            if r.is_false_reject and r.raw_score is not None:
                false_rejects.append({
                    "sample_id": r.sample_id,
                    "ground_truth": r.ground_truth,
                    "raw_score": r.raw_score,
                    "threshold": r.threshold,
                    "prediction": r.prediction,
                    "device": r.device or "N/A",
                    "scenario": r.scenario or "N/A",
                    "total_latency_ms": r.total_latency_ms,
                })

        # Sort false accepts by most confident error (highest score for higher_is_accept)
        if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
            false_accepts.sort(key=lambda x: x["raw_score"], reverse=True)
            false_rejects.sort(key=lambda x: x["raw_score"])
        else:
            false_accepts.sort(key=lambda x: x["raw_score"])
            false_rejects.sort(key=lambda x: x["raw_score"], reverse=True)

        slowest_samples.sort(key=lambda x: x["total_latency_ms"], reverse=True)

        return {
            "false_accepts": false_accepts[:top_k],
            "false_rejects": false_rejects[:top_k],
            "slowest_samples": slowest_samples[:top_k],
            "failed_samples": failed_samples[:top_k],
        }

    @classmethod
    def generate_markdown_report(
        cls,
        metadata: Dict[str, Any],
        cls_metrics: BinaryClassificationMetrics,
        bio_metrics: BiometricMetrics,
        pad_metrics: PADMetrics,
        lat_metrics: LatencyMetrics,
        calibration_result: Optional[CalibrationResult],
        hard_cases: Dict[str, List[Dict[str, Any]]],
        plots: Dict[str, str],
        output_path: str,
    ) -> str:
        """Constructs a comprehensive Markdown report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        model_info = metadata.get("model", {})
        eval_info = metadata.get("evaluation_settings", {})
        data_info = metadata.get("test_dataset_summary", {})
        git_info = metadata.get("git", {})
        env_info = metadata.get("environment", {})

        md = []
        md.append(f"# Benchmark Report: {metadata.get('benchmark_name', 'Fingerprint Evaluation')}")
        md.append(f"**Run Timestamp:** `{metadata.get('timestamp_utc')}` | **Dataset Version:** `{metadata.get('dataset_version')}` | **Task:** `{metadata.get('task_type')}`\n")

        # 1. Executive Summary
        md.append("## 1. Executive Summary\n")
        md.append("| Metric | Value | Reference Target |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Model Evaluated** | `{model_info.get('name', 'N/A')}` (v{model_info.get('version', '1.0.0')}) | - |")
        md.append(f"| **Accuracy** | `{cls_metrics.accuracy * 100:.2f}%` | > 95.0% |")
        md.append(f"| **Balanced Accuracy** | `{cls_metrics.balanced_accuracy * 100:.2f}%` | > 95.0% |")
        md.append(f"| **Equal Error Rate (EER)** | `{(bio_metrics.eer * 100):.2f}%` (th={bio_metrics.eer_threshold:.4f}) | < 2.0% |" if bio_metrics.eer is not None else "| **Equal Error Rate (EER)** | N/A | - |")
        md.append(f"| **FAR / APCER** | `{bio_metrics.far * 100:.3f}%` | <= 0.1% |")
        md.append(f"| **FRR / BPCER** | `{bio_metrics.frr * 100:.3f}%` | <= 1.0% |")
        md.append(f"| **TAR @ FAR <= 0.1%** | `{bio_metrics.tar_at_far_01pct * 100:.2f}%` | > 98.0% |" if bio_metrics.tar_at_far_01pct is not None else "| **TAR @ FAR <= 0.1%** | N/A | - |")
        md.append(f"| **P50 (Median) Latency** | `{lat_metrics.warm_median_p50_ms:.1f} ms` | < 500 ms |")
        md.append(f"| **P95 Latency** | `{lat_metrics.warm_p95_ms:.1f} ms` | < 1500 ms |")
        md.append(f"| **P99 Latency** | `{lat_metrics.warm_p99_ms:.1f} ms` | < 3000 ms |")
        md.append(f"| **Max Latency** | `{lat_metrics.max_ms:.1f} ms` | < 7000 ms |\n")

        # 2. Dataset & Protocol
        md.append("## 2. Dataset & Calibration Protocol\n")
        md.append(f"- **Total Test Samples:** {data_info.get('total_samples', 0)} ({data_info.get('live_samples', 0)} bona fide, {data_info.get('spoof_samples', 0)} attacks)")
        md.append(f"- **Unique Subjects:** {data_info.get('unique_subjects', 'N/A')}")
        md.append(f"- **Manifest SHA-256:** `{data_info.get('manifest_hash', 'N/A')[:16]}...`")
        md.append(f"- **Score Direction:** `{eval_info.get('score_direction')}`")
        md.append(f"- **Threshold Method:** `{eval_info.get('threshold_method')}`")
        if calibration_result:
            md.append(f"- **Operating Threshold Selected:** `{calibration_result.selected_threshold:.4f}`")
            md.append(f"- **Calibration Set Performance:** FAR = {calibration_result.calib_far*100:.3f}%, FRR = {calibration_result.calib_frr*100:.3f}%, TAR = {calibration_result.calib_tar*100:.2f}%")
        md.append("")

        # 3. Biometric Verification & PAD Metrics
        md.append("## 3. Biometric & PAD Performance Metrics\n")
        md.append("### Classification & Biometrics")
        md.append("| Metric | Value | Metric | Value |")
        md.append("| :--- | :--- | :--- | :--- |")
        md.append(f"| **Accuracy** | {cls_metrics.accuracy*100:.2f}% | **Balanced Accuracy** | {cls_metrics.balanced_accuracy*100:.2f}% |")
        md.append(f"| **Precision** | {cls_metrics.precision*100:.2f}% | **Recall (Sensitivity)** | {cls_metrics.recall*100:.2f}% |")
        md.append(f"| **Specificity** | {cls_metrics.specificity*100:.2f}% | **F1 Score** | {cls_metrics.f1_score*100:.2f}% |")
        md.append(f"| **ROC-AUC** | {cls_metrics.roc_auc:.4f}" if cls_metrics.roc_auc else "| **ROC-AUC** | N/A | **PR-AUC** | " + (f"{cls_metrics.pr_auc:.4f}" if cls_metrics.pr_auc else "N/A") + " |")
        md.append(f"| **FAR (False Accept Rate)** | {bio_metrics.far*100:.4f}% | **FRR (False Reject Rate)** | {bio_metrics.frr*100:.4f}% |")
        md.append(f"| **TAR (1 - FRR)** | {bio_metrics.tar*100:.4f}% | **EER** | " + (f"{bio_metrics.eer*100:.2f}%" if bio_metrics.eer is not None else "N/A") + " |")
        md.append(f"| **TAR @ FAR=1.0%** | " + (f"{bio_metrics.tar_at_far_1pct*100:.2f}%" if bio_metrics.tar_at_far_1pct is not None else "N/A") + f" | **TAR @ FAR=0.1%** | " + (f"{bio_metrics.tar_at_far_01pct*100:.2f}%" if bio_metrics.tar_at_far_01pct is not None else "N/A") + " |")
        md.append(f"| **APCER (ISO PAD)** | {pad_metrics.apcer*100:.3f}% | **BPCER (ISO PAD)** | {pad_metrics.bpcer*100:.3f}% |")
        md.append(f"| **ACER (ISO PAD)** | {pad_metrics.acer*100:.3f}% | **Total Errors / Exceptions** | {cls_metrics.total_errors} |\n")

        # Confusion Matrix
        md.append("### Confusion Matrix")
        md.append("| | Predicted Bona Fide / Accept | Predicted Attack / Reject |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Actual Bona Fide** | **TP:** {cls_metrics.tp} | **FN:** {cls_metrics.fn} |")
        md.append(f"| **Actual Attack** | **FP:** {cls_metrics.fp} | **TN:** {cls_metrics.tn} |\n")

        # 4. Latency Profiling
        md.append("## 4. Latency Profiling & Tail Breakdown\n")
        md.append("| Stage | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Max (ms) |")
        md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
        md.append(f"| **End-to-End** | {lat_metrics.warm_mean_ms:.1f} | {lat_metrics.warm_median_p50_ms:.1f} | {lat_metrics.warm_p95_ms:.1f} | {lat_metrics.warm_p99_ms:.1f} | {lat_metrics.max_ms:.1f} |")
        for st_name, st_val in lat_metrics.stages.items():
            md.append(f"| *{st_name.capitalize()}* | {st_val.mean_ms:.1f} | {st_val.p50_ms:.1f} | {st_val.p95_ms:.1f} | {st_val.p99_ms:.1f} | {st_val.max_ms:.1f} |")
        md.append("")

        md.append("### Tail Latency Analysis")
        md.append("| Threshold | Slow Samples Count | Percentage |")
        md.append("| :--- | :--- | :--- |")
        for k, v in lat_metrics.slow_cases_count.items():
            pct = lat_metrics.slow_cases_percentage.get(k, 0.0)
            md.append(f"| **Latency {k}** | {v} | {pct:.2f}% |")
        md.append("")

        # 5. Diagnostic Plots
        md.append("## 5. Diagnostic Performance Plots\n")
        md.append("Diagnostic charts generated for this run:\n")
        for name, rel_p in plots.items():
            fname = os.path.basename(rel_p)
            md.append(f"- **{name.replace('_', ' ').title()}:** `plots/{fname}`")
        md.append("")

        # 6. Hard-Case Analysis
        md.append("## 6. Hard-Case & Tail Failure Analysis\n")

        md.append("### High-Confidence False Accepts (Spoofs Mistaken as Live)")
        if hard_cases["false_accepts"]:
            md.append("| Sample ID | Ground Truth | Score | Threshold | Spoof Type | Device | Latency (ms) |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for item in hard_cases["false_accepts"]:
                md.append(f"| `{item['sample_id']}` | {item['ground_truth']} | **{item['raw_score']:.4f}** | {item['threshold']:.4f} | {item['spoof_type']} | {item['device']} | {item['total_latency_ms']:.1f} |")
        else:
            md.append("*No False Accepts encountered.*")
        md.append("")

        md.append("### High-Confidence False Rejects (Genuine Fingers Rejected)")
        if hard_cases["false_rejects"]:
            md.append("| Sample ID | Ground Truth | Score | Threshold | Scenario | Device | Latency (ms) |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
            for item in hard_cases["false_rejects"]:
                md.append(f"| `{item['sample_id']}` | {item['ground_truth']} | **{item['raw_score']:.4f}** | {item['threshold']:.4f} | {item['scenario']} | {item['device']} | {item['total_latency_ms']:.1f} |")
        else:
            md.append("*No False Rejects encountered.*")
        md.append("")

        md.append("### Slowest Inferences (> Tail Latencies)")
        if hard_cases["slowest_samples"]:
            md.append("| Sample ID | Latency (ms) | Ground Truth | Score | Prediction | Device |")
            md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
            for item in hard_cases["slowest_samples"][:10]:
                sc_str = f"{item['raw_score']:.4f}" if item['raw_score'] is not None else "N/A"
                md.append(f"| `{item['sample_id']}` | **{item['total_latency_ms']:.1f}** | {item['ground_truth']} | {sc_str} | {item['prediction']} | {item['device']} |")
        md.append("")

        if hard_cases["failed_samples"]:
            md.append("### Infrastructure & Model Execution Errors")
            md.append("| Sample ID | Status | Error Type | Error Message |")
            md.append("| :--- | :--- | :--- | :--- |")
            for item in hard_cases["failed_samples"]:
                md.append(f"| `{item['sample_id']}` | `{item['status']}` | `{item['error_type']}` | {item['error_message']} |")
            md.append("")

        # 7. Reproducibility Metadata
        md.append("## 7. Reproducibility & Environment Specs\n")
        md.append(f"- **Git Commit:** `{git_info.get('git_commit')}` (Branch: `{git_info.get('git_branch')}`, Dirty: `{git_info.get('git_dirty')}`)")
        md.append(f"- **OS / Platform:** `{env_info.get('os')} {env_info.get('os_release')}` ({env_info.get('machine')})")
        md.append(f"- **Python Version:** `{env_info.get('python_version')}`")
        md.append(f"- **CUDA Device:** `{env_info.get('cuda_device_name', 'CPU')}` (Available: `{env_info.get('cuda_available')}`)")
        md.append(f"- **Config Hash:** `{metadata.get('config_hash')}`")
        md.append("")

        content = "\n".join(md)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return content

    @classmethod
    def generate_html_report(
        cls,
        metadata: Dict[str, Any],
        cls_metrics: BinaryClassificationMetrics,
        bio_metrics: BiometricMetrics,
        pad_metrics: PADMetrics,
        lat_metrics: LatencyMetrics,
        calibration_result: Optional[CalibrationResult],
        hard_cases: Dict[str, List[Dict[str, Any]]],
        plots: Dict[str, str],
        output_path: str,
    ) -> str:
        """Generates a standalone HTML benchmark report."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        model_name = metadata.get("model", {}).get("name", "Model Evaluation")

        # Encode plots to base64 if present
        b64_plots: Dict[str, str] = {}
        for k, p in plots.items():
            if os.path.exists(p):
                with open(p, "rb") as img_f:
                    b64_plots[k] = base64.b64encode(img_f.read()).decode("utf-8")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>YellowSense Benchmark - {model_name}</title>
    <style>
        :root {{
            --bg: #0f172a;
            --card-bg: #1e293b;
            --card-border: #334155;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent-blue: #38bdf8;
            --accent-green: #4ade80;
            --accent-red: #f87171;
            --accent-yellow: #fbbf24;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .header {{
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--card-border);
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 26px;
            color: var(--accent-blue);
        }}
        .meta-bar {{
            color: var(--text-muted);
            font-size: 14px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 16px;
        }}
        .card-label {{
            font-size: 12px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 4px;
            font-weight: 600;
        }}
        .card-value {{
            font-size: 24px;
            font-weight: bold;
            color: var(--text-main);
        }}
        .card-sub {{
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0 24px 0;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            overflow: hidden;
            font-size: 14px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--card-border);
        }}
        th {{
            background: #111827;
            color: var(--text-muted);
            font-weight: 600;
        }}
        .plots-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .plot-card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }}
        .plot-card img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        .tag-pass {{ color: var(--accent-green); font-weight: bold; }}
        .tag-fail {{ color: var(--accent-red); font-weight: bold; }}
        .tag-warn {{ color: var(--accent-yellow); font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>YellowSense Fingerprint Benchmark Report</h1>
        <div class="meta-bar">
            <strong>Model:</strong> {model_name} (v{metadata.get('model', {}).get('version', '1.0.0')}) | 
            <strong>Dataset:</strong> {metadata.get('dataset_version')} | 
            <strong>Date:</strong> {metadata.get('timestamp_utc', '')[:19]} UTC
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-label">Accuracy</div>
            <div class="card-value tag-pass">{cls_metrics.accuracy*100:.2f}%</div>
            <div class="card-sub">Balanced: {cls_metrics.balanced_accuracy*100:.2f}%</div>
        </div>
        <div class="card">
            <div class="card-label">Equal Error Rate (EER)</div>
            <div class="card-value">{f"{bio_metrics.eer*100:.2f}%" if bio_metrics.eer is not None else "N/A"}</div>
            <div class="card-sub">Threshold: {f"{bio_metrics.eer_threshold:.4f}" if bio_metrics.eer_threshold is not None else "N/A"}</div>
        </div>
        <div class="card">
            <div class="card-label">FAR / APCER</div>
            <div class="card-value tag-warn">{bio_metrics.far*100:.3f}%</div>
            <div class="card-sub">False Accepts: {cls_metrics.fp}</div>
        </div>
        <div class="card">
            <div class="card-label">FRR / BPCER</div>
            <div class="card-value">{bio_metrics.frr*100:.3f}%</div>
            <div class="card-sub">False Rejects: {cls_metrics.fn}</div>
        </div>
        <div class="card">
            <div class="card-label">P50 / P95 Latency</div>
            <div class="card-value">{lat_metrics.warm_median_p50_ms:.0f} / {lat_metrics.warm_p95_ms:.0f} ms</div>
            <div class="card-sub">Max: {lat_metrics.max_ms:.0f} ms | P99: {lat_metrics.warm_p99_ms:.0f} ms</div>
        </div>
    </div>

    <h2>Diagnostic Performance Plots</h2>
    <div class="plots-grid">
        {"".join([f'<div class="plot-card"><h3>{k.replace("_", " ").title()}</h3><img src="data:image/png;base64,{v}" alt="{k}" /></div>' for k, v in b64_plots.items()])}
    </div>

    <h2>Latency Breakdown & Tail Analysis</h2>
    <table>
        <thead>
            <tr><th>Metric / Threshold</th><th>Latency / Count</th><th>Percentage / Notes</th></tr>
        </thead>
        <tbody>
            <tr><td><strong>Warm Mean Latency</strong></td><td>{lat_metrics.warm_mean_ms:.1f} ms</td><td>Steady state</td></tr>
            <tr><td><strong>P90 / P95 / P99 Latency</strong></td><td>{lat_metrics.warm_p90_ms:.1f} / {lat_metrics.warm_p95_ms:.1f} / {lat_metrics.warm_p99_ms:.1f} ms</td><td>Tail percentiles</td></tr>
            {"".join([f"<tr><td>Slow Samples ({k})</td><td>{v} samples</td><td>{lat_metrics.slow_cases_percentage.get(k, 0.0):.2f}%</td></tr>" for k, v in lat_metrics.slow_cases_count.items()])}
        </tbody>
    </table>

    <h2>High-Confidence False Accepts</h2>
    <table>
        <thead>
            <tr><th>Sample ID</th><th>Ground Truth</th><th>Score</th><th>Threshold</th><th>Spoof Type</th><th>Device</th><th>Latency</th></tr>
        </thead>
        <tbody>
            {"".join([f"<tr><td><code>{item['sample_id']}</code></td><td>{item['ground_truth']}</td><td class='tag-fail'><strong>{item['raw_score']:.4f}</strong></td><td>{item['threshold']:.4f}</td><td>{item['spoof_type']}</td><td>{item['device']}</td><td>{item['total_latency_ms']:.1f} ms</td></tr>" for item in hard_cases["false_accepts"]]) if hard_cases["false_accepts"] else "<tr><td colspan='7'>No False Accepts</td></tr>"}
        </tbody>
    </table>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return html

    @staticmethod
    def compare_runs(run_directories: List[str], output_markdown_path: Optional[str] = None) -> pd.DataFrame:
        """Loads and compares summary metrics across multiple historical benchmark run directories."""
        rows: List[Dict[str, Any]] = []

        for r_dir in run_directories:
            metrics_path = os.path.join(r_dir, "metrics.json")
            meta_path = os.path.join(r_dir, "metadata.json")

            if not os.path.exists(metrics_path):
                continue

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)

            meta_data = {}
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_data = json.load(f)

            cls_m = metrics_data.get("classification", {})
            bio_m = metrics_data.get("biometric", {})
            pad_m = metrics_data.get("pad", {})
            lat_m = metrics_data.get("latency", {})
            model_info = meta_data.get("model", {})

            rows.append({
                "run_dir": os.path.basename(r_dir),
                "model_name": model_info.get("name", "N/A"),
                "dataset_version": meta_data.get("dataset_version", "N/A"),
                "accuracy": f"{cls_m.get('accuracy', 0.0)*100:.2f}%",
                "balanced_acc": f"{cls_m.get('balanced_accuracy', 0.0)*100:.2f}%",
                "eer": f"{bio_m.get('eer', 0.0)*100:.2f}%" if bio_m.get('eer') is not None else "N/A",
                "far": f"{bio_m.get('far', 0.0)*100:.3f}%",
                "frr": f"{bio_m.get('frr', 0.0)*100:.3f}%",
                "tar_at_far_01pct": f"{bio_m.get('tar_at_far_01pct', 0.0)*100:.2f}%" if bio_m.get('tar_at_far_01pct') is not None else "N/A",
                "apcer": f"{pad_m.get('apcer', 0.0)*100:.3f}%",
                "bpcer": f"{pad_m.get('bpcer', 0.0)*100:.3f}%",
                "p50_latency_ms": f"{lat_m.get('warm_median_p50_ms', 0.0):.1f}",
                "p95_latency_ms": f"{lat_m.get('warm_p95_ms', 0.0):.1f}",
                "p99_latency_ms": f"{lat_m.get('warm_p99_ms', 0.0):.1f}",
                "max_latency_ms": f"{lat_m.get('max_ms', 0.0):.1f}",
            })

        df = pd.DataFrame(rows)

        if output_markdown_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_markdown_path)), exist_ok=True)
            with open(output_markdown_path, "w", encoding="utf-8") as f:
                f.write("# Benchmark Comparison Summary\n\n")
                f.write(df.to_markdown(index=False) if hasattr(df, "to_markdown") else str(df))

        return df
