"""
General binary classification metrics calculation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
from sklearn import metrics as sk_metrics

from ..schemas import GroundTruthLabel, PredictionOutcome, ScoreDirection


@dataclass
class BinaryClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    tp: int
    fp: int
    tn: int
    fn: int
    total_evaluated: int
    total_errors: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_classification_metrics(
    ground_truths: List[str],
    predictions: List[str],
    scores: Optional[List[Optional[float]]] = None,
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
) -> BinaryClassificationMetrics:
    """Computes standard binary classification metrics."""
    if len(ground_truths) != len(predictions):
        raise ValueError(f"Mismatched lengths: ground_truths ({len(ground_truths)}) vs predictions ({len(predictions)})")

    valid_gt: List[bool] = []
    valid_pred: List[bool] = []
    valid_scores: List[float] = []
    total_errors = 0

    for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        pred_clean = str(pred).lower().strip()
        if pred_clean in ("error", "skipped", "none"):
            total_errors += 1
            continue

        is_pos_gt = GroundTruthLabel.is_positive(gt)
        is_pos_pred = pred_clean in ("accept", "live", "bonafide", "genuine", "1", "true")

        valid_gt.append(is_pos_gt)
        valid_pred.append(is_pos_pred)

        if scores is not None and idx < len(scores) and scores[idx] is not None:
            s_val = float(scores[idx])
            # If lower_is_accept, invert score for standard AUC calculation where higher = positive
            if score_direction == ScoreDirection.LOWER_IS_ACCEPT:
                valid_scores.append(-s_val)
            else:
                valid_scores.append(s_val)

    if not valid_gt:
        return BinaryClassificationMetrics(
            accuracy=0.0,
            balanced_accuracy=0.0,
            precision=0.0,
            recall=0.0,
            specificity=0.0,
            f1_score=0.0,
            roc_auc=None,
            pr_auc=None,
            tp=0,
            fp=0,
            tn=0,
            fn=0,
            total_evaluated=0,
            total_errors=total_errors,
        )

    y_true = np.array(valid_gt, dtype=int)
    y_pred = np.array(valid_pred, dtype=int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = (rec + spec) / 2.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    roc_auc = None
    pr_auc = None
    if len(valid_scores) == len(y_true) and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(sk_metrics.roc_auc_score(y_true, valid_scores))
            pr_auc = float(sk_metrics.average_precision_score(y_true, valid_scores))
        except Exception:
            pass

    return BinaryClassificationMetrics(
        accuracy=float(acc),
        balanced_accuracy=float(bal_acc),
        precision=float(prec),
        recall=float(rec),
        specificity=float(spec),
        f1_score=float(f1),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        total_evaluated=len(y_true),
        total_errors=total_errors,
    )
