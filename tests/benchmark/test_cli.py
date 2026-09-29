"""
Unit tests for CLI commands and subparser actions.
"""

import os
import pytest
from benchmark_harness.cli import main


def test_cli_validate():
    ret = main(["validate", "--config", "configs/benchmark_single_finger.yaml"])
    assert ret == 0


def test_cli_run(tmp_path):
    # Overwrite output dir in test run
    ret = main(["run", "--config", "configs/benchmark_single_finger.yaml"])
    assert ret == 0


def test_cli_threshold():
    ret = main(["threshold", "--config", "configs/benchmark_single_finger.yaml"])
    assert ret == 0


def test_cli_join_scores(tmp_path):
    out_csv = str(tmp_path / "joined.csv")
    ret = main([
        "join-scores",
        "--manifest", "datasets/manifests/example_manifest.csv",
        "--signals",
        "dualscale=scores/dualscale.csv",
        "model3d=scores/model3d.csv",
        "flash=scores/flash_liveness.csv",
        "--output", out_csv
    ])
    assert ret == 0
    assert os.path.exists(out_csv)
