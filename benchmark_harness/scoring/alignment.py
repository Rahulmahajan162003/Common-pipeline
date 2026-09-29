"""
Multi-signal score alignment and fusion table generation.
Strictly joins scores from DualScale-2D, 3D, and Flash Liveness on sample_id.
"""

import os
import csv
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import pandas as pd

from ..schemas import ModelInferenceResult, ManifestSample, GroundTruthLabel


@dataclass
class JoinedScoreTable:
    """Represents a strictly aligned multi-signal score dataset."""
    dataframe: pd.DataFrame
    signal_names: List[str]
    total_samples: int
    missing_samples_by_signal: Dict[str, List[str]]
    extra_samples_by_signal: Dict[str, List[str]]
    is_fully_aligned: bool


class ScoreTableAligner:
    """Aligns multiple model evaluation outputs strictly by sample_id."""

    @staticmethod
    def align_results(
        manifest: Optional[List[ManifestSample]] = None,
        results_by_signal: Optional[Dict[str, List[ModelInferenceResult]]] = None,
    ) -> JoinedScoreTable:
        """Joins results from multiple signals by sample_id."""
        if not results_by_signal:
            return JoinedScoreTable(
                dataframe=pd.DataFrame(),
                signal_names=[],
                total_samples=0,
                missing_samples_by_signal={},
                extra_samples_by_signal={},
                is_fully_aligned=True,
            )

        signal_names = list(results_by_signal.keys())

        # Determine reference sample_id set and ground truths
        ref_sample_ids: List[str] = []
        gt_map: Dict[str, str] = {}
        sample_meta_map: Dict[str, Dict[str, Any]] = {}

        if manifest:
            for s in manifest:
                ref_sample_ids.append(s.sample_id)
                gt_map[s.sample_id] = GroundTruthLabel.normalize_label(s.ground_truth)
                sample_meta_map[s.sample_id] = {
                    "subject_id": s.subject_id,
                    "device": s.device,
                    "scenario": s.scenario,
                    "spoof_type": s.spoof_type,
                    "split": s.split,
                }
        else:
            # Build union of sample IDs
            all_ids_set: Set[str] = set()
            for sig, res_list in results_by_signal.items():
                for r in res_list:
                    all_ids_set.add(r.sample_id)
            ref_sample_ids = sorted(list(all_ids_set))

        # Check for duplicates within each signal
        signal_maps: Dict[str, Dict[str, ModelInferenceResult]] = {}
        for sig, res_list in results_by_signal.items():
            s_map = {}
            for r in res_list:
                if r.sample_id in s_map:
                    raise ValueError(f"Duplicate sample_id '{r.sample_id}' found in results for signal '{sig}'.")
                s_map[r.sample_id] = r
            signal_maps[sig] = s_map

        missing_by_signal: Dict[str, List[str]] = {}
        extra_by_signal: Dict[str, List[str]] = {}
        is_aligned = True

        for sig in signal_names:
            sig_ids = set(signal_maps[sig].keys())
            ref_set = set(ref_sample_ids)
            missing = sorted(list(ref_set - sig_ids))
            extra = sorted(list(sig_ids - ref_set))
            missing_by_signal[sig] = missing
            extra_by_signal[sig] = extra
            if missing or extra:
                is_aligned = False

        # Assemble rows
        rows: List[Dict[str, Any]] = []
        for sid in ref_sample_ids:
            row: Dict[str, Any] = {
                "sample_id": sid,
                "ground_truth": gt_map.get(sid, "unknown"),
            }
            if sid in sample_meta_map:
                row.update(sample_meta_map[sid])

            for sig in signal_names:
                res = signal_maps[sig].get(sid)
                if res:
                    row[f"{sig}_score"] = res.raw_score
                    row[f"{sig}_latency_ms"] = res.timing.total_latency_ms
                    row[f"{sig}_status"] = res.status.value
                else:
                    row[f"{sig}_score"] = None
                    row[f"{sig}_latency_ms"] = None
                    row[f"{sig}_status"] = "MISSING"

            rows.append(row)

        df = pd.DataFrame(rows)

        return JoinedScoreTable(
            dataframe=df,
            signal_names=signal_names,
            total_samples=len(df),
            missing_samples_by_signal=missing_by_signal,
            extra_samples_by_signal=extra_by_signal,
            is_fully_aligned=is_aligned,
        )

    @staticmethod
    def export_fusion_table(joined_table: JoinedScoreTable, output_csv_path: str) -> None:
        """Exports the aligned score table to CSV for downstream fusion training."""
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)
        joined_table.dataframe.to_csv(output_csv_path, index=False)
