"""
Latency metrics calculation, percentile distribution, and slow-case profiling.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np

from ..schemas import TimingRecord


@dataclass
class LatencyBreakdownStage:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass
class LatencyMetrics:
    total_samples: int
    cold_start_latency_ms: Optional[float]
    warm_mean_ms: float
    warm_median_p50_ms: float
    warm_p90_ms: float
    warm_p95_ms: float
    warm_p99_ms: float
    min_ms: float
    max_ms: float
    std_dev_ms: float
    slow_cases_count: Dict[str, int]        # e.g., ">1000ms": 5, ">7000ms": 1
    slow_cases_percentage: Dict[str, float] # e.g., ">7000ms": 0.5%
    stages: Dict[str, LatencyBreakdownStage]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        return res


def compute_latency_metrics(
    timing_records: List[TimingRecord],
    thresholds_ms: Optional[List[float]] = None,
    cold_start_record: Optional[TimingRecord] = None,
) -> LatencyMetrics:
    """Computes comprehensive latency percentiles, breakdown stages, and slow-case counts."""
    if thresholds_ms is None:
        thresholds_ms = [1000.0, 2000.0, 5000.0, 7000.0]

    if not timing_records:
        return LatencyMetrics(
            total_samples=0,
            cold_start_latency_ms=cold_start_record.total_latency_ms if cold_start_record else None,
            warm_mean_ms=0.0,
            warm_median_p50_ms=0.0,
            warm_p90_ms=0.0,
            warm_p95_ms=0.0,
            warm_p99_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            std_dev_ms=0.0,
            slow_cases_count={f">{int(t)}ms": 0 for t in thresholds_ms},
            slow_cases_percentage={f">{int(t)}ms": 0.0 for t in thresholds_ms},
            stages={},
        )

    totals = np.array([r.total_latency_ms for r in timing_records], dtype=np.float64)
    n = len(totals)

    mean_v = float(np.mean(totals))
    p50_v = float(np.percentile(totals, 50))
    p90_v = float(np.percentile(totals, 90))
    p95_v = float(np.percentile(totals, 95))
    p99_v = float(np.percentile(totals, 99))
    min_v = float(np.min(totals))
    max_v = float(np.max(totals))
    std_v = float(np.std(totals))

    slow_counts: Dict[str, int] = {}
    slow_pcts: Dict[str, float] = {}
    for th in thresholds_ms:
        lbl = f">{int(th)}ms"
        cnt = int(np.sum(totals > th))
        slow_counts[lbl] = cnt
        slow_pcts[lbl] = float(cnt / n * 100.0) if n > 0 else 0.0

    # Stage breakdowns
    stages: Dict[str, LatencyBreakdownStage] = {}
    for stage_name, attr in [
        ("preprocess", "preprocess_latency_ms"),
        ("inference", "inference_latency_ms"),
        ("postprocess", "postprocess_latency_ms"),
        ("network", "network_latency_ms"),
    ]:
        vals = [getattr(r, attr) for r in timing_records if getattr(r, attr) is not None]
        if vals:
            arr = np.array(vals, dtype=np.float64)
            stages[stage_name] = LatencyBreakdownStage(
                mean_ms=float(np.mean(arr)),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                max_ms=float(np.max(arr)),
            )

    return LatencyMetrics(
        total_samples=n,
        cold_start_latency_ms=cold_start_record.total_latency_ms if cold_start_record else None,
        warm_mean_ms=mean_v,
        warm_median_p50_ms=p50_v,
        warm_p90_ms=p90_v,
        warm_p95_ms=p95_v,
        warm_p99_ms=p99_v,
        min_ms=min_v,
        max_ms=max_v,
        std_dev_ms=std_v,
        slow_cases_count=slow_counts,
        slow_cases_percentage=slow_pcts,
        stages=stages,
    )
