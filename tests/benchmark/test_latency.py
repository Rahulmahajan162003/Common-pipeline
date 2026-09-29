"""
Unit tests for latency metrics, percentile distribution, and slow case profiling.
"""

import pytest
from benchmark_harness.schemas import TimingRecord
from benchmark_harness.metrics.latency import compute_latency_metrics


def test_latency_percentiles():
    # 100 samples from 10ms to 1000ms
    timings = [TimingRecord(total_latency_ms=float(i * 10)) for i in range(1, 101)]
    lat_m = compute_latency_metrics(timings, thresholds_ms=[500.0, 800.0, 1000.0])

    assert lat_m.total_samples == 100
    assert lat_m.min_ms == 10.0
    assert lat_m.max_ms == 1000.0
    assert abs(lat_m.warm_median_p50_ms - 505.0) <= 10.0
    assert abs(lat_m.warm_p90_ms - 901.0) <= 15.0
    assert abs(lat_m.warm_p95_ms - 950.5) <= 15.0
    assert abs(lat_m.warm_p99_ms - 990.1) <= 15.0

    # Slow cases
    assert lat_m.slow_cases_count[">500ms"] == 50
    assert lat_m.slow_cases_percentage[">500ms"] == 50.0
    assert lat_m.slow_cases_count[">800ms"] == 20
    assert lat_m.slow_cases_percentage[">800ms"] == 20.0


def test_latency_stage_breakdown():
    timings = [
        TimingRecord(
            total_latency_ms=100.0,
            preprocess_latency_ms=20.0,
            inference_latency_ms=70.0,
            postprocess_latency_ms=10.0,
        ),
        TimingRecord(
            total_latency_ms=120.0,
            preprocess_latency_ms=25.0,
            inference_latency_ms=85.0,
            postprocess_latency_ms=10.0,
        ),
    ]

    lat_m = compute_latency_metrics(timings)
    assert "preprocess" in lat_m.stages
    assert "inference" in lat_m.stages
    assert "postprocess" in lat_m.stages

    assert lat_m.stages["preprocess"].mean_ms == 22.5
    assert lat_m.stages["inference"].mean_ms == 77.5
    assert lat_m.stages["postprocess"].mean_ms == 10.0
