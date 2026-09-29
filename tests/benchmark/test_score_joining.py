"""
Unit tests for multi-signal score table alignment and fusion preparation.
"""

import pytest
import pandas as pd
from benchmark_harness.schemas import ModelInferenceResult, ManifestSample, ScoreDirection, SampleStatus, TimingRecord
from benchmark_harness.scoring.alignment import ScoreTableAligner


def test_score_table_alignment_success():
    manifest = [
        ManifestSample(sample_id="S001", ground_truth="live", image_path=""),
        ManifestSample(sample_id="S002", ground_truth="spoof", image_path=""),
    ]

    # Model 1 results
    res_ds = [
        ModelInferenceResult(sample_id="S001", model_name="dualscale", model_version="1.0", raw_score=0.92, score_direction=ScoreDirection.HIGHER_IS_ACCEPT, timing=TimingRecord(40.0)),
        ModelInferenceResult(sample_id="S002", model_name="dualscale", model_version="1.0", raw_score=0.10, score_direction=ScoreDirection.HIGHER_IS_ACCEPT, timing=TimingRecord(42.0)),
    ]

    # Model 2 results (in different order)
    res_3d = [
        ModelInferenceResult(sample_id="S002", model_name="model3d", model_version="1.0", raw_score=0.18, score_direction=ScoreDirection.HIGHER_IS_ACCEPT, timing=TimingRecord(60.0)),
        ModelInferenceResult(sample_id="S001", model_name="model3d", model_version="1.0", raw_score=0.88, score_direction=ScoreDirection.HIGHER_IS_ACCEPT, timing=TimingRecord(58.0)),
    ]

    joined = ScoreTableAligner.align_results(
        manifest=manifest,
        results_by_signal={"dualscale": res_ds, "model3d": res_3d}
    )

    assert joined.is_fully_aligned is True
    assert len(joined.dataframe) == 2
    df = joined.dataframe

    s001_row = df[df["sample_id"] == "S001"].iloc[0]
    assert s001_row["dualscale_score"] == 0.92
    assert s001_row["model3d_score"] == 0.88
    assert s001_row["ground_truth"] == "live"

    s002_row = df[df["sample_id"] == "S002"].iloc[0]
    assert s002_row["dualscale_score"] == 0.10
    assert s002_row["model3d_score"] == 0.18
    assert s002_row["ground_truth"] == "spoof"


def test_score_table_missing_sample_detection():
    manifest = [
        ManifestSample(sample_id="S001", ground_truth="live", image_path=""),
        ManifestSample(sample_id="S002", ground_truth="spoof", image_path=""),
        ManifestSample(sample_id="S003", ground_truth="live", image_path=""),
    ]

    res_ds = [
        ModelInferenceResult(sample_id="S001", model_name="dualscale", model_version="1.0", raw_score=0.92, score_direction=ScoreDirection.HIGHER_IS_ACCEPT),
        ModelInferenceResult(sample_id="S002", model_name="dualscale", model_version="1.0", raw_score=0.10, score_direction=ScoreDirection.HIGHER_IS_ACCEPT),
        # S003 missing in dualscale!
    ]

    res_flash = [
        ModelInferenceResult(sample_id="S001", model_name="flash", model_version="1.0", raw_score=0.95, score_direction=ScoreDirection.HIGHER_IS_ACCEPT),
        ModelInferenceResult(sample_id="S002", model_name="flash", model_version="1.0", raw_score=0.08, score_direction=ScoreDirection.HIGHER_IS_ACCEPT),
        ModelInferenceResult(sample_id="S003", model_name="flash", model_version="1.0", raw_score=0.91, score_direction=ScoreDirection.HIGHER_IS_ACCEPT),
    ]

    joined = ScoreTableAligner.align_results(
        manifest=manifest,
        results_by_signal={"dualscale": res_ds, "flash": res_flash}
    )

    assert joined.is_fully_aligned is False
    assert "S003" in joined.missing_samples_by_signal["dualscale"]
    assert len(joined.missing_samples_by_signal["flash"]) == 0
