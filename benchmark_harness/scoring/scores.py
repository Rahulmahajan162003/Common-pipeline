"""
Raw score containers, validation, and serialization.
"""

import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..schemas import ScoreDirection, SampleStatus, PredictionOutcome


@dataclass
class RawScoreRecord:
    sample_id: str
    model_name: str
    model_version: str
    ground_truth: str
    raw_score: Optional[float]
    score_direction: ScoreDirection
    total_latency_ms: float
    status: SampleStatus = SampleStatus.PASS
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


def validate_score(score: Optional[float]) -> bool:
    """Returns True if score is a valid finite float, False if None, NaN, or Inf."""
    if score is None:
        return False
    if not isinstance(score, (int, float)):
        return False
    return not (math.isnan(score) or math.isinf(score))
