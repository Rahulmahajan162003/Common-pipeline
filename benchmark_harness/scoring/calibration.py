"""
Threshold calibration engine for selecting operating points from calibration data.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from ..schemas import ScoreDirection, ThresholdMethod, GroundTruthLabel


@dataclass
class CalibrationResult:
    selected_threshold: float
    method: ThresholdMethod
    calib_far: float
    calib_frr: float
    calib_tar: float
    calib_accuracy: float
    calib_acer: float
    calib_num_samples: int
    score_direction: ScoreDirection
    details: Dict[str, Any]


class ThresholdCalibrator:
    """Calibrates decision thresholds on held-out calibration data."""

    @staticmethod
    def calibrate(
        scores: List[float],
        ground_truths: List[str],
        method: ThresholdMethod = ThresholdMethod.TARGET_FAR,
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        target_far: float = 0.001,
        target_frr: Optional[float] = None,
        manual_threshold: Optional[float] = None,
        sweep_steps: int = 500,
    ) -> CalibrationResult:
        """Determines the optimal operating threshold using calibration data."""
        if method == ThresholdMethod.MANUAL:
            if manual_threshold is None:
                raise ValueError("Manual threshold selection requires 'manual_threshold' to be set.")
            th = float(manual_threshold)
            return CalibrationResult(
                selected_threshold=th,
                method=ThresholdMethod.MANUAL,
                calib_far=0.0,
                calib_frr=0.0,
                calib_tar=1.0,
                calib_accuracy=1.0,
                calib_acer=0.0,
                calib_num_samples=len(scores),
                score_direction=score_direction,
                details={"manual_threshold": th},
            )

        if not scores or not ground_truths:
            raise ValueError("Calibration requires non-empty scores and ground truth labels.")

        labels = np.array([GroundTruthLabel.is_positive(gt) for gt in ground_truths], dtype=bool)
        score_arr = np.array(scores, dtype=np.float64)

        pos_scores = score_arr[labels]
        neg_scores = score_arr[~labels]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            raise ValueError(
                f"Calibration set must contain both genuine and spoof samples. "
                f"Found {len(pos_scores)} genuine and {len(neg_scores)} spoof samples."
            )

        # Fine-grained threshold grid
        min_s = float(np.min(score_arr))
        max_s = float(np.max(score_arr))
        margin = (max_s - min_s) * 0.01 if max_s > min_s else 0.01
        threshold_candidates = np.linspace(min_s - margin, max_s + margin, sweep_steps)

        best_th: float = min_s
        best_far: float = 1.0
        best_frr: float = 1.0
        details: Dict[str, Any] = {}

        if method == ThresholdMethod.EER:
            # Find threshold minimizing abs(FAR - FRR)
            min_diff = float("inf")
            for th in threshold_candidates:
                if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    far = np.mean(neg_scores >= th)
                    frr = np.mean(pos_scores < th)
                else:
                    far = np.mean(neg_scores <= th)
                    frr = np.mean(pos_scores > th)

                diff = abs(far - frr)
                if diff < min_diff:
                    min_diff = diff
                    best_th = float(th)
                    best_far = float(far)
                    best_frr = float(frr)

            details["eer_gap"] = abs(best_far - best_frr)

        elif method == ThresholdMethod.TARGET_FAR:
            # Find threshold where FAR <= target_far with lowest FRR (highest TAR)
            valid_candidates = []
            for th in threshold_candidates:
                if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    far = np.mean(neg_scores >= th)
                    frr = np.mean(pos_scores < th)
                else:
                    far = np.mean(neg_scores <= th)
                    frr = np.mean(pos_scores > th)

                if far <= target_far:
                    valid_candidates.append((th, far, frr))

            if valid_candidates:
                # Pick the one with lowest FRR
                valid_candidates.sort(key=lambda x: (x[2], x[1]))
                best_th, best_far, best_frr = valid_candidates[0]
                details["target_far_met"] = True
            else:
                # If no candidate achieved target_far, pick lowest FAR
                best_tuple = min(
                    [
                        (
                            th,
                            np.mean(neg_scores >= th) if score_direction == ScoreDirection.HIGHER_IS_ACCEPT else np.mean(neg_scores <= th),
                            np.mean(pos_scores < th) if score_direction == ScoreDirection.HIGHER_IS_ACCEPT else np.mean(pos_scores > th),
                        )
                        for th in threshold_candidates
                    ],
                    key=lambda x: x[1]
                )
                best_th, best_far, best_frr = best_tuple
                details["target_far_met"] = False
                details["warning"] = f"Calibration data could not satisfy FAR <= {target_far}. Selected closest point."

            details["requested_target_far"] = target_far

        elif method == ThresholdMethod.TARGET_FRR:
            req_frr = target_frr or 0.01
            valid_candidates = []
            for th in threshold_candidates:
                if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    far = np.mean(neg_scores >= th)
                    frr = np.mean(pos_scores < th)
                else:
                    far = np.mean(neg_scores <= th)
                    frr = np.mean(pos_scores > th)

                if frr <= req_frr:
                    valid_candidates.append((th, far, frr))

            if valid_candidates:
                valid_candidates.sort(key=lambda x: (x[1], x[2]))
                best_th, best_far, best_frr = valid_candidates[0]
            else:
                best_th = float(threshold_candidates[0])
                best_far = float(np.mean(neg_scores >= best_th) if score_direction == ScoreDirection.HIGHER_IS_ACCEPT else np.mean(neg_scores <= best_th))
                best_frr = float(np.mean(pos_scores < best_th) if score_direction == ScoreDirection.HIGHER_IS_ACCEPT else np.mean(pos_scores > best_th))

            details["requested_target_frr"] = req_frr

        elif method == ThresholdMethod.MIN_BALANCED_ERROR:
            min_ber = float("inf")
            for th in threshold_candidates:
                if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    far = np.mean(neg_scores >= th)
                    frr = np.mean(pos_scores < th)
                else:
                    far = np.mean(neg_scores <= th)
                    frr = np.mean(pos_scores > th)

                ber = (far + frr) / 2.0
                if ber < min_ber:
                    min_ber = ber
                    best_th = float(th)
                    best_far = float(far)
                    best_frr = float(frr)

            details["min_balanced_error"] = (best_far + best_frr) / 2.0

        calib_tar = 1.0 - best_frr
        calib_acer = (best_far + best_frr) / 2.0
        # Compute accuracy on calibration set at best_th
        if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
            preds = score_arr >= best_th
        else:
            preds = score_arr <= best_th
        acc = float(np.mean(preds == labels))

        return CalibrationResult(
            selected_threshold=float(best_th),
            method=method,
            calib_far=float(best_far),
            calib_frr=float(best_frr),
            calib_tar=float(calib_tar),
            calib_accuracy=acc,
            calib_acer=float(calib_acer),
            calib_num_samples=len(scores),
            score_direction=score_direction,
            details=details,
        )
