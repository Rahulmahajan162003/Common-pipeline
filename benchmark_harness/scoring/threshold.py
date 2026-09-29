"""
Threshold evaluation engine and sweep analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..schemas import ScoreDirection, SampleStatus, PredictionOutcome, GroundTruthLabel


class ThresholdEngine:
    """Evaluates raw model scores against decision thresholds."""

    @staticmethod
    def decide(
        score: Optional[float],
        threshold: float,
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        status: SampleStatus = SampleStatus.PASS,
        treat_error_as_reject: bool = False,
    ) -> PredictionOutcome:
        """Applies threshold rule to a single score."""
        if status != SampleStatus.PASS or score is None or np.isnan(score) or np.isinf(score):
            if treat_error_as_reject:
                return PredictionOutcome.REJECT
            return PredictionOutcome.ERROR

        if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
            return PredictionOutcome.ACCEPT if score >= threshold else PredictionOutcome.REJECT
        elif score_direction == ScoreDirection.LOWER_IS_ACCEPT:
            return PredictionOutcome.ACCEPT if score <= threshold else PredictionOutcome.REJECT
        else:
            raise ValueError(f"Unsupported score direction: {score_direction}")

    @staticmethod
    def sweep(
        scores: List[float],
        ground_truths: List[str],
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        steps: int = 100,
    ) -> List[Dict[str, float]]:
        """Generates a comprehensive threshold sweep table."""
        if not scores or not ground_truths or len(scores) != len(ground_truths):
            return []

        # Convert ground truths to booleans (True = Live/Genuine, False = Spoof/Impostor)
        labels = np.array([GroundTruthLabel.is_positive(gt) for gt in ground_truths], dtype=bool)
        score_arr = np.array(scores, dtype=np.float64)

        num_pos = int(np.sum(labels))
        num_neg = int(len(labels) - num_pos)

        if num_pos == 0 or num_neg == 0:
            return []

        min_s = float(np.min(score_arr))
        max_s = float(np.max(score_arr))

        if min_s == max_s:
            thresholds = [min_s]
        else:
            margin = (max_s - min_s) * 0.01
            thresholds = np.linspace(min_s - margin, max_s + margin, steps).tolist()

        sweep_rows: List[Dict[str, float]] = []

        for th in thresholds:
            if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                preds = score_arr >= th
            else:
                preds = score_arr <= th

            tp = int(np.sum(preds & labels))
            fp = int(np.sum(preds & (~labels)))
            tn = int(np.sum((~preds) & (~labels)))
            fn = int(np.sum((~preds) & labels))

            far = fp / num_neg if num_neg > 0 else 0.0
            frr = fn / num_pos if num_pos > 0 else 0.0
            tar = 1.0 - frr
            acc = (tp + tn) / len(labels)
            acer = (far + frr) / 2.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            sweep_rows.append({
                "threshold": float(th),
                "far": float(far),
                "frr": float(frr),
                "tar": float(tar),
                "apcer": float(far),   # In PAD semantics, APCER is impostor/spoof accepted (FAR)
                "bpcer": float(frr),   # In PAD semantics, BPCER is bona fide rejected (FRR)
                "acer": float(acer),
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            })

        return sweep_rows


@dataclass
class ThresholdSweepResult:
    rows: List[Dict[str, float]]
    min_score: float
    max_score: float
    score_direction: ScoreDirection
