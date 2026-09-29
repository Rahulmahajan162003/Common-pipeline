"""
Unit tests for deterministic repeatability and run reproducibility.
"""

import os
import json
import pytest
from benchmark_harness.config import load_config
from benchmark_harness.runner import BenchmarkRunner


def test_benchmark_reproducibility(tmp_path):
    config_path = "configs/benchmark_single_finger.yaml"
    cfg1 = load_config(config_path)
    cfg1.outputs.root = str(tmp_path / "run1")
    cfg1.outputs.export_plots = False

    cfg2 = load_config(config_path)
    cfg2.outputs.root = str(tmp_path / "run2")
    cfg2.outputs.export_plots = False

    runner1 = BenchmarkRunner(cfg1)
    out1 = runner1.run()

    runner2 = BenchmarkRunner(cfg2)
    out2 = runner2.run()

    # Read metrics.json from both runs
    with open(os.path.join(out1["run_dir"], "metrics.json"), "r", encoding="utf-8") as f:
        metrics1 = json.load(f)

    with open(os.path.join(out2["run_dir"], "metrics.json"), "r", encoding="utf-8") as f:
        metrics2 = json.load(f)

    # Classification and biometric metrics must be bitwise identical
    assert metrics1["classification"] == metrics2["classification"]
    assert metrics1["biometric"] == metrics2["biometric"]
    assert metrics1["selected_threshold"] == metrics2["selected_threshold"]
