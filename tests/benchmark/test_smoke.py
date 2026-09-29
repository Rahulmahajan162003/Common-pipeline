"""
End-to-end smoke test for the complete benchmark harness execution.
"""

import os
import pytest
from benchmark_harness.config import load_config
from benchmark_harness.runner import BenchmarkRunner


def test_end_to_end_single_finger_benchmark(tmp_path):
    config = load_config("configs/benchmark_single_finger.yaml")
    config.outputs.root = str(tmp_path / "results")

    runner = BenchmarkRunner(config)
    outputs = runner.run()

    run_dir = outputs["run_dir"]
    assert os.path.exists(run_dir)
    assert os.path.exists(os.path.join(run_dir, "metadata.json"))
    assert os.path.exists(os.path.join(run_dir, "config.yaml"))
    assert os.path.exists(os.path.join(run_dir, "predictions.csv"))
    assert os.path.exists(os.path.join(run_dir, "raw_scores.csv"))
    assert os.path.exists(os.path.join(run_dir, "threshold_analysis.csv"))
    assert os.path.exists(os.path.join(run_dir, "metrics.json"))
    assert os.path.exists(os.path.join(run_dir, "latency.csv"))
    assert os.path.exists(os.path.join(run_dir, "report.md"))
    assert os.path.exists(os.path.join(run_dir, "report.html"))
    assert os.path.exists(os.path.join(run_dir, "plots", "roc.png"))
    assert os.path.exists(os.path.join(run_dir, "plots", "det.png"))


def test_end_to_end_multi_signal_and_fusion(tmp_path):
    config = load_config("configs/benchmark_multi_signal.yaml")
    config.outputs.root = str(tmp_path / "results_multi")

    runner = BenchmarkRunner(config)
    outputs = runner.run()

    run_dir = outputs["run_dir"]
    assert os.path.exists(os.path.join(run_dir, "fusion_scores.csv"))
    assert "fusion_scores" in outputs


def test_score_only_evaluation(tmp_path):
    config = load_config("configs/benchmark_score_only.yaml")
    config.outputs.root = str(tmp_path / "results_score_only")

    runner = BenchmarkRunner(config)
    outputs = runner.run()

    run_dir = outputs["run_dir"]
    assert os.path.exists(os.path.join(run_dir, "predictions.csv"))
    assert os.path.exists(os.path.join(run_dir, "metrics.json"))
