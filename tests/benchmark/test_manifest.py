"""
Unit tests for manifest parsing, integrity validation, and split leakage detection.
"""

import os
import pytest
from benchmark_harness.schemas import ManifestSample, ValidationError
from benchmark_harness.manifest import BenchmarkManifest
from benchmark_harness.validation import BenchmarkValidator


def test_duplicate_sample_id_detection():
    samples = [
        ManifestSample(sample_id="S001", ground_truth="live", image_path="datasets/synthetic_data/S001.jpg"),
        ManifestSample(sample_id="S001", ground_truth="spoof", image_path="datasets/synthetic_data/S002.jpg"),
    ]
    manifest = BenchmarkManifest(samples)
    errors = BenchmarkValidator.validate_manifest(manifest, check_images=False)
    assert any(e.category == "DUPLICATE_SAMPLE_ID" for e in errors)


def test_invalid_ground_truth_label():
    samples = [
        ManifestSample(sample_id="S001", ground_truth="invalid_label_123", image_path="datasets/synthetic_data/S001.jpg"),
    ]
    manifest = BenchmarkManifest(samples)
    errors = BenchmarkValidator.validate_manifest(manifest, check_images=False)
    assert any(e.category == "INVALID_GROUND_TRUTH" for e in errors)


def test_missing_image_file_detection(tmp_path):
    non_existent = str(tmp_path / "does_not_exist.jpg")
    samples = [
        ManifestSample(sample_id="S001", ground_truth="live", image_path=non_existent),
        ManifestSample(sample_id="S002", ground_truth="spoof", image_path=non_existent),
    ]
    manifest = BenchmarkManifest(samples)
    errors = BenchmarkValidator.validate_manifest(manifest, check_images=True, allow_missing_images=False)
    assert any(e.category == "IMAGE_FILE_NOT_FOUND" for e in errors)


def test_subject_leakage_detection():
    calib_samples = [
        ManifestSample(sample_id="C001", subject_id="SUBJ_A", ground_truth="live", image_path=""),
        ManifestSample(sample_id="C002", subject_id="SUBJ_B", ground_truth="spoof", image_path=""),
    ]
    test_samples = [
        ManifestSample(sample_id="T001", subject_id="SUBJ_A", ground_truth="live", image_path=""),  # Leakage!
        ManifestSample(sample_id="T002", subject_id="SUBJ_C", ground_truth="spoof", image_path=""),
    ]
    calib_bm = BenchmarkManifest(calib_samples)
    test_bm = BenchmarkManifest(test_samples)

    errors = BenchmarkValidator.validate_split_leakage(calib_bm, test_bm, verify_hashes=False)
    assert any(e.category == "SUBJECT_LEAKAGE" for e in errors)
    leak_err = next(e for e in errors if e.category == "SUBJECT_LEAKAGE")
    assert leak_err.details["subject_id"] == "SUBJ_A"


def test_split_sample_id_overlap():
    calib_samples = [
        ManifestSample(sample_id="S001", subject_id="SUBJ_A", ground_truth="live", image_path=""),
    ]
    test_samples = [
        ManifestSample(sample_id="S001", subject_id="SUBJ_B", ground_truth="live", image_path=""),
    ]
    errors = BenchmarkValidator.validate_split_leakage(BenchmarkManifest(calib_samples), BenchmarkManifest(test_samples), verify_hashes=False)
    assert any(e.category == "SPLIT_SAMPLE_OVERLAP" for e in errors)


def test_manifest_csv_roundtrip(tmp_path):
    csv_file = str(tmp_path / "test_manifest.csv")
    samples = [
        ManifestSample(sample_id="S001", subject_id="U1", ground_truth="live", image_path="img1.jpg", split="calibration"),
        ManifestSample(sample_id="S002", subject_id="U2", ground_truth="spoof", image_path="img2.jpg", split="test"),
    ]
    bm = BenchmarkManifest(samples)
    bm.to_csv(csv_file)

    loaded_bm = BenchmarkManifest.from_csv(csv_file)
    assert len(loaded_bm) == 2
    assert loaded_bm[0].sample_id == "S001"
    assert loaded_bm[0].split == "calibration"
    assert loaded_bm[1].ground_truth == "spoof"
