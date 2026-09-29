"""
Presentation Attack Detection (PAD) metrics per ISO/IEC 30107-3: APCER, BPCER, ACER.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

from ..schemas import GroundTruthLabel


@dataclass
class PADMetrics:
    apcer: float  # Attack Presentation Classification Error Rate (proportion of attacks classified as bona fide)
    bpcer: float  # Bona Fide Presentation Classification Error Rate (proportion of bona fide rejected)
    acer: float   # Average Classification Error Rate: (APCER + BPCER) / 2
    apcer_by_spoof_type: Dict[str, float]  # APCER broken down by spoof material/scenario
    num_bonafide: int
    num_attacks: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_pad_metrics(
    ground_truths: List[str],
    predictions: List[str],
    spoof_types: Optional[List[Optional[str]]] = None,
) -> PADMetrics:
    """Computes ISO/IEC 30107-3 Presentation Attack Detection metrics."""
    bonafide_total = 0
    bonafide_incorrect = 0

    attack_total = 0
    attack_incorrect = 0

    attack_by_type_total: Dict[str, int] = defaultdict(int)
    attack_by_type_incorrect: Dict[str, int] = defaultdict(int)

    for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        pred_clean = str(pred).lower().strip()
        if pred_clean in ("error", "skipped", "none"):
            continue

        is_bonafide = GroundTruthLabel.is_positive(gt)
        is_classified_as_bonafide = pred_clean in ("accept", "live", "bonafide", "genuine", "1", "true")

        stype = (spoof_types[idx] if (spoof_types and idx < len(spoof_types) and spoof_types[idx]) else "general_spoof")

        if is_bonafide:
            bonafide_total += 1
            if not is_classified_as_bonafide:
                bonafide_incorrect += 1  # False reject of bona fide
        else:
            attack_total += 1
            attack_by_type_total[stype] += 1
            if is_classified_as_bonafide:
                attack_incorrect += 1  # False accept of attack
                attack_by_type_incorrect[stype] += 1

    bpcer = (bonafide_incorrect / bonafide_total) if bonafide_total > 0 else 0.0
    apcer = (attack_incorrect / attack_total) if attack_total > 0 else 0.0
    acer = (apcer + bpcer) / 2.0

    apcer_by_type: Dict[str, float] = {}
    for stype, count in attack_by_type_total.items():
        inc = attack_by_type_incorrect[stype]
        apcer_by_type[stype] = float(inc / count) if count > 0 else 0.0

    return PADMetrics(
        apcer=float(apcer),
        bpcer=float(bpcer),
        acer=float(acer),
        apcer_by_spoof_type=apcer_by_type,
        num_bonafide=bonafide_total,
        num_attacks=attack_total,
    )
