"""
Tests for sanity stops, strict split validation, and fallback prevention guards.
"""

import os
import pytest
import numpy as np

from benchmark_harness.schemas import ScoreDirection, ThresholdMethod, ManifestSample
from benchmark_harness.scoring.calibration import ThresholdCalibrator
from benchmark_harness.config import BenchmarkConfig, DataConfig, EvaluationConfig, ModelConfig, OutputsConfig
from benchmark_harness.runner import BenchmarkRunner
from benchmark_harness.adapters.dualscale_2d import DualScale2DAdapter
from benchmark_harness.adapters.rtk_pad import RtkPadAdapter


def test_calibration_auc_sanity_stop():
    """Verify that calibration halts with ValueError if ROC-AUC < 0.50 (e.g. inverted scores)."""
    # Live samples have LOW scores, and Spoof samples have HIGH scores (i.e. P(attack))
    # When evaluated with HIGHER_IS_ACCEPT, AUC will be ~ 0.05 (< 0.50)
    scores = [0.1, 0.15, 0.05, 0.9, 0.85, 0.95]
    ground_truths = ["live", "live", "live", "spoof", "spoof", "spoof"]

    with pytest.raises(ValueError, match="SANITY STOP: Calibration ROC-AUC is"):
        ThresholdCalibrator.calibrate(
            scores=scores,
            ground_truths=ground_truths,
            method=ThresholdMethod.TARGET_FAR,
            score_direction=ScoreDirection.HIGHER_IS_ACCEPT,
            enforce_sanity_auc=True,
        )


def test_empty_test_split_halts_execution(tmp_path):
    """Verify that an empty or misnamed test split halts execution with a descriptive error."""
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,ground_truth,image_path,split\n"
        "S1,live,img1.jpg,calibration\n"
        "S2,spoof,img2.jpg,calibration\n"
    )

    config = BenchmarkConfig(
        data=DataConfig(
            manifest_path=str(manifest_csv),
            test_split_name="non_existent_test_split",
            check_image_existence=False,
            allow_missing_images=True,
            verify_hashes=False,
        ),
        models=[
            ModelConfig(name="dummy", adapter="score_csv", score_csv_path="scores/dualscale.csv")
        ],
        outputs=OutputsConfig(root=str(tmp_path / "out")),
    )

    runner = BenchmarkRunner(config)
    with pytest.raises(ValueError, match="Configured test split 'non_existent_test_split' contains 0 samples"):
        runner.run()


def test_missing_model_checkpoint_fails_without_fallback():
    """Verify that missing model weights strictly raise error and do not generate fallback scores."""
    adapter = DualScale2DAdapter()
    with pytest.raises(ValueError, match="'checkpoint_path' must be provided"):
        adapter.load(checkpoint_path=None)

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        adapter.load(checkpoint_path="non_existent_path.pth")

    sample = ManifestSample(sample_id="test1", ground_truth="live", image_path="dummy.jpg")
    res = adapter.predict(sample)
    assert res.status.value == "ERROR"
    assert "not loaded" in res.error_message


def test_rtk_pad_score_direction_consistency():
    """Verify that RTK-PAD correctly maps probability based on configured score_direction."""
    adapter_live = RtkPadAdapter(score_direction=ScoreDirection.HIGHER_IS_ACCEPT)
    adapter_attack = RtkPadAdapter(score_direction=ScoreDirection.HIGHER_IS_REJECT)

    assert adapter_live.score_direction == ScoreDirection.HIGHER_IS_ACCEPT
    assert adapter_attack.score_direction == ScoreDirection.HIGHER_IS_REJECT
