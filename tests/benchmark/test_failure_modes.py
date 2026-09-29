"""
Unit tests for explicit failure handling, NaN scores, crashes, and single-class checks.
"""

import pytest
from benchmark_harness.schemas import (
    ManifestSample,
    ScoreDirection,
    SampleStatus,
    PredictionOutcome,
)
from benchmark_harness.scoring.threshold import ThresholdEngine
from benchmark_harness.manifest import BenchmarkManifest
from benchmark_harness.validation import BenchmarkValidator
from benchmark_harness.scoring.scores import validate_score


def test_nan_and_inf_scores():
    assert validate_score(float("nan")) is False
    assert validate_score(float("inf")) is False
    assert validate_score(float("-inf")) is False
    assert validate_score(None) is False
    assert validate_score(0.85) is True


def test_infrastructure_error_separation():
    # If a model throws an error, status is ERROR
    # When treat_error_as_reject=False, outcome must be ERROR (not REJECT)
    outcome = ThresholdEngine.decide(
        score=None,
        threshold=0.5,
        status=SampleStatus.ERROR,
        treat_error_as_reject=False
    )
    assert outcome == PredictionOutcome.ERROR

    # When treat_error_as_reject=True, outcome is REJECT
    outcome_rej = ThresholdEngine.decide(
        score=None,
        threshold=0.5,
        status=SampleStatus.ERROR,
        treat_error_as_reject=True
    )
    assert outcome_rej == PredictionOutcome.REJECT


def test_single_class_manifest_validation():
    # Only genuine samples
    live_only = [
        ManifestSample(sample_id="S1", ground_truth="live", image_path=""),
        ManifestSample(sample_id="S2", ground_truth="live", image_path=""),
    ]
    bm = BenchmarkManifest(live_only)
    errs = BenchmarkValidator.validate_manifest(bm, check_images=False)
    assert any(e.category == "SINGLE_CLASS_DATASET" for e in errs)
