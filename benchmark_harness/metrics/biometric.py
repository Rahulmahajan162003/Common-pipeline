"""
Biometric verification metrics: FAR, FRR, TAR, EER, and TAR @ fixed FAR operating points.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import interp1d

from ..schemas import GroundTruthLabel, ScoreDirection


@dataclass
class BiometricMetrics:
    far: float  # False Accept Rate
    frr: float  # False Reject Rate
    tar: float  # Genuine Accept Rate (1 - FRR)
    eer: Optional[float]
    eer_threshold: Optional[float]
    tar_at_far_1pct: Optional[float]    # TAR @ FAR = 1%
    tar_at_far_01pct: Optional[float]   # TAR @ FAR = 0.1%
    tar_at_far_001pct: Optional[float]  # TAR @ FAR = 0.01%
    num_genuine: int
    num_impostor: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_biometric_metrics(
    ground_truths: List[str],
    predictions: List[str],
    scores: Optional[List[Optional[float]]] = None,
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
) -> BiometricMetrics:
    """Computes biometric verification metrics and operating points."""
    valid_gt: List[bool] = []
    valid_pred: List[bool] = []
    valid_scores: List[float] = []

    for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        pred_clean = str(pred).lower().strip()
        if pred_clean in ("error", "skipped", "none"):
            continue

        is_pos = GroundTruthLabel.is_positive(gt)
        is_accept = pred_clean in ("accept", "live", "bonafide", "genuine", "1", "true")

        valid_gt.append(is_pos)
        valid_pred.append(is_accept)

        if scores is not None and idx < len(scores) and scores[idx] is not None:
            valid_scores.append(float(scores[idx]))

    if not valid_gt:
        return BiometricMetrics(
            far=0.0,
            frr=0.0,
            tar=0.0,
            eer=None,
            eer_threshold=None,
            tar_at_far_1pct=None,
            tar_at_far_01pct=None,
            tar_at_far_001pct=None,
            num_genuine=0,
            num_impostor=0,
        )

    y_true = np.array(valid_gt, dtype=bool)
    y_pred = np.array(valid_pred, dtype=bool)

    num_genuine = int(np.sum(y_true))
    num_impostor = int(np.sum(~y_true))

    # Point-in-time rates from discrete predictions
    fp = int(np.sum(y_pred & (~y_true)))
    fn = int(np.sum((~y_pred) & y_true))

    far = fp / num_impostor if num_impostor > 0 else 0.0
    frr = fn / num_genuine if num_genuine > 0 else 0.0
    tar = 1.0 - frr

    # Continuous curve metrics if continuous scores are present
    eer = None
    eer_threshold = None
    tar_1 = None
    tar_01 = None
    tar_001 = None

    if len(valid_scores) == len(y_true) and num_genuine > 0 and num_impostor > 0:
        score_arr = np.array(valid_scores, dtype=np.float64)
        gen_scores = score_arr[y_true]
        imp_scores = score_arr[~y_true]

        # Calculate ROC points over dense thresholds
        thresholds = np.linspace(np.min(score_arr) - 0.01, np.max(score_arr) + 0.01, 1000)
        far_list = []
        frr_list = []
        tar_list = []

        for th in thresholds:
            if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                cur_far = np.mean(imp_scores >= th)
                cur_frr = np.mean(gen_scores < th)
            else:
                cur_far = np.mean(imp_scores <= th)
                cur_frr = np.mean(gen_scores > th)

            far_list.append(cur_far)
            frr_list.append(cur_frr)
            tar_list.append(1.0 - cur_frr)

        far_arr = np.array(far_list)
        frr_arr = np.array(frr_list)
        tar_arr = np.array(tar_list)

        # EER: where far == frr
        diffs = np.abs(far_arr - frr_arr)
        min_idx = int(np.argmin(diffs))
        eer = float((far_arr[min_idx] + frr_arr[min_idx]) / 2.0)
        eer_threshold = float(thresholds[min_idx])

        # TAR @ fixed FAR operating points
        def get_tar_at_target_far(target_f: float) -> Optional[float]:
            valid_indices = np.where(far_arr <= target_f)[0]
            if len(valid_indices) > 0:
                return float(np.max(tar_arr[valid_indices]))
            return None

        tar_1 = get_tar_at_target_far(0.01)
        tar_01 = get_tar_at_target_far(0.001)
        tar_001 = get_tar_at_target_far(0.0001)

    return BiometricMetrics(
        far=float(far),
        frr=float(frr),
        tar=float(tar),
        eer=eer,
        eer_threshold=eer_threshold,
        tar_at_far_1pct=tar_1,
        tar_at_far_01pct=tar_01,
        tar_at_far_001pct=tar_001,
        num_genuine=num_genuine,
        num_impostor=num_impostor,
    )
