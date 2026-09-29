"""
Unit tests for threshold decision logic, calibration methods, and sweep analysis.
"""

import pytest
import numpy as np
from benchmark_harness.schemas import ScoreDirection, ThresholdMethod, SampleStatus, PredictionOutcome
from benchmark_harness.scoring.threshold import ThresholdEngine
from benchmark_harness.scoring.calibration import ThresholdCalibrator


def test_threshold_decide_higher_is_accept():
    # score >= 0.7 -> accept
    assert ThresholdEngine.decide(0.8, threshold=0.7, score_direction=ScoreDirection.HIGHER_IS_ACCEPT) == PredictionOutcome.ACCEPT
    assert ThresholdEngine.decide(0.6, threshold=0.7, score_direction=ScoreDirection.HIGHER_IS_ACCEPT) == PredictionOutcome.REJECT
    assert ThresholdEngine.decide(0.7, threshold=0.7, score_direction=ScoreDirection.HIGHER_IS_ACCEPT) == PredictionOutcome.ACCEPT


def test_threshold_decide_lower_is_accept():
    # distance metric: score <= 0.4 -> accept
    assert ThresholdEngine.decide(0.3, threshold=0.4, score_direction=ScoreDirection.LOWER_IS_ACCEPT) == PredictionOutcome.ACCEPT
    assert ThresholdEngine.decide(0.5, threshold=0.4, score_direction=ScoreDirection.LOWER_IS_ACCEPT) == PredictionOutcome.REJECT
    assert ThresholdEngine.decide(0.4, threshold=0.4, score_direction=ScoreDirection.LOWER_IS_ACCEPT) == PredictionOutcome.ACCEPT


def test_threshold_sweep_monotonicity():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gts = ["spoof", "spoof", "spoof", "spoof", "live", "live", "live", "live", "live"]
    sweep = ThresholdEngine.sweep(scores, gts, score_direction=ScoreDirection.HIGHER_IS_ACCEPT, steps=50)

    assert len(sweep) > 0
    # As threshold increases, FAR should be non-increasing and FRR non-decreasing
    fars = [r["far"] for r in sweep]
    frrs = [r["frr"] for r in sweep]
    for i in range(len(fars) - 1):
        assert fars[i] >= fars[i+1] - 1e-9
        assert frrs[i] <= frrs[i+1] + 1e-9


def test_calibration_eer():
    scores = [0.95, 0.90, 0.85, 0.80, 0.15, 0.10, 0.05, 0.00]
    gts = ["live", "live", "live", "live", "spoof", "spoof", "spoof", "spoof"]
    calib = ThresholdCalibrator.calibrate(scores, gts, method=ThresholdMethod.EER, score_direction=ScoreDirection.HIGHER_IS_ACCEPT)

    assert calib.selected_threshold > 0.15
    assert calib.selected_threshold < 0.80
    assert calib.calib_far == 0.0
    assert calib.calib_frr == 0.0


def test_calibration_target_far():
    # 10 impostor scores uniformly spread, 10 genuine scores
    imp_scores = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    gen_scores = [0.42, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    scores = gen_scores + imp_scores
    gts = ["live"] * len(gen_scores) + ["spoof"] * len(imp_scores)

    # Target FAR <= 10% (0.10)
    calib = ThresholdCalibrator.calibrate(scores, gts, method=ThresholdMethod.TARGET_FAR, target_far=0.10, score_direction=ScoreDirection.HIGHER_IS_ACCEPT)
    assert calib.calib_far <= 0.101
    assert calib.calib_tar >= 0.90
