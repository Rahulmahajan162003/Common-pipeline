"""
Unit tests for binary classification, biometric verification, and PAD metrics.
"""

import pytest
from benchmark_harness.metrics.classification import compute_classification_metrics
from benchmark_harness.metrics.biometric import compute_biometric_metrics
from benchmark_harness.metrics.pad import compute_pad_metrics
from benchmark_harness.schemas import ScoreDirection


def test_perfect_classifier_metrics():
    gts = ["live", "live", "spoof", "spoof"]
    preds = ["accept", "accept", "reject", "reject"]
    scores = [0.9, 0.8, 0.1, 0.2]

    cls_m = compute_classification_metrics(gts, preds, scores)
    assert cls_m.accuracy == 1.0
    assert cls_m.f1_score == 1.0
    assert cls_m.tp == 2
    assert cls_m.tn == 2
    assert cls_m.fp == 0
    assert cls_m.fn == 0
    assert cls_m.roc_auc == 1.0

    bio_m = compute_biometric_metrics(gts, preds, scores)
    assert bio_m.far == 0.0
    assert bio_m.frr == 0.0
    assert bio_m.tar == 1.0
    assert bio_m.eer == 0.0

    pad_m = compute_pad_metrics(gts, preds)
    assert pad_m.apcer == 0.0
    assert pad_m.bpcer == 0.0
    assert pad_m.acer == 0.0


def test_completely_wrong_classifier_metrics():
    gts = ["live", "live", "spoof", "spoof"]
    preds = ["reject", "reject", "accept", "accept"]
    scores = [0.1, 0.2, 0.9, 0.8]

    cls_m = compute_classification_metrics(gts, preds, scores)
    assert cls_m.accuracy == 0.0
    assert cls_m.f1_score == 0.0
    assert cls_m.tp == 0
    assert cls_m.tn == 0
    assert cls_m.fp == 2
    assert cls_m.fn == 2

    bio_m = compute_biometric_metrics(gts, preds, scores)
    assert bio_m.far == 1.0
    assert bio_m.frr == 1.0
    assert bio_m.tar == 0.0

    pad_m = compute_pad_metrics(gts, preds)
    assert pad_m.apcer == 1.0
    assert pad_m.bpcer == 1.0
    assert pad_m.acer == 1.0


def test_pad_metrics_spoof_type_breakdown():
    gts = ["spoof", "spoof", "spoof", "spoof", "live"]
    preds = ["accept", "reject", "accept", "reject", "accept"]  # 2 false accepts out of 4 attacks
    stypes = ["printed_photo", "printed_photo", "silicone_pad", "silicone_pad", "none"]

    pad_m = compute_pad_metrics(gts, preds, spoof_types=stypes)
    assert pad_m.apcer == 0.5  # 2 / 4
    assert pad_m.bpcer == 0.0  # 0 / 1
    assert pad_m.acer == 0.25
    assert pad_m.apcer_by_spoof_type["printed_photo"] == 0.5
    assert pad_m.apcer_by_spoof_type["silicone_pad"] == 0.5


def test_empty_predictions_handling():
    cls_m = compute_classification_metrics([], [])
    assert cls_m.total_evaluated == 0
    assert cls_m.accuracy == 0.0

    bio_m = compute_biometric_metrics([], [])
    assert bio_m.num_genuine == 0
    assert bio_m.num_impostor == 0
