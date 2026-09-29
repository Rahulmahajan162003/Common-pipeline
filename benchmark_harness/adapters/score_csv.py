"""
Adapter for evaluating precomputed score CSV files without running neural networks.
"""

import os
import csv
import time
import math
from typing import Dict, Any, Optional, Tuple

from .base import BaseModelAdapter
from ..schemas import (
    ManifestSample,
    ModelInferenceResult,
    ScoreDirection,
    SampleStatus,
    PredictionOutcome,
    TimingRecord,
)


class ScoreCSVAdapter(BaseModelAdapter):
    """Adapter that reads precomputed scores and latencies from a CSV file."""

    def __init__(
        self,
        name: str = "score_csv",
        version: str = "1.0.0",
        score_csv_path: Optional[str] = None,
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            version=version,
            device="cpu",
            score_direction=score_direction,
            extra_params=extra_params,
        )
        self.score_csv_path = score_csv_path or (extra_params.get("score_csv_path") if extra_params else None)
        self.score_map: Dict[str, Tuple[float, Optional[float], Dict[str, Any]]] = {}  # sample_id -> (score, latency, extra)

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        csv_path = checkpoint_path or self.score_csv_path
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"Score CSV file not found: '{csv_path}'")

        self.score_map.clear()
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Score CSV '{csv_path}' is empty or invalid.")

            # Identify score column
            score_col = None
            for candidate in ["score", "raw_score", "prob", "probability", "confidence", "similarity"]:
                for fn in reader.fieldnames:
                    if fn.strip().lower() == candidate:
                        score_col = fn
                        break
                if score_col:
                    break

            if not score_col:
                # Fallback to second column if at least 2 columns
                if len(reader.fieldnames) >= 2:
                    score_col = reader.fieldnames[1]
                else:
                    raise ValueError(f"Could not identify score column in score CSV '{csv_path}'. Columns: {reader.fieldnames}")

            # Identify sample_id column
            id_col = None
            for candidate in ["sample_id", "id", "sample", "SampleID"]:
                for fn in reader.fieldnames:
                    if fn.strip().lower() == candidate:
                        id_col = fn
                        break
                if id_col:
                    break
            if not id_col:
                id_col = reader.fieldnames[0]

            # Identify latency column if available
            lat_col = None
            for candidate in ["latency_ms", "latency", "total_latency_ms", "time_ms"]:
                for fn in reader.fieldnames:
                    if fn.strip().lower() == candidate:
                        lat_col = fn
                        break
                if lat_col:
                    break

            for row_idx, row in enumerate(reader):
                sid = row[id_col].strip()
                val_str = row[score_col].strip()
                if not sid:
                    continue

                try:
                    score = float(val_str)
                    if math.isnan(score) or math.isinf(score):
                        raise ValueError("Score is NaN or Inf")
                except ValueError as e:
                    raise ValueError(f"Invalid score value '{val_str}' at row {row_idx+1} for sample '{sid}': {str(e)}")

                latency_val = None
                if lat_col and row.get(lat_col):
                    try:
                        latency_val = float(row[lat_col].strip())
                    except ValueError:
                        pass

                if sid in self.score_map:
                    raise ValueError(f"Duplicate sample_id '{sid}' in score CSV '{csv_path}' at row {row_idx+1}.")

                self.score_map[sid] = (score, latency_val, row)

        self._is_loaded = True

    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        if not self._is_loaded:
            raise RuntimeError("ScoreCSVAdapter must be loaded before calling predict().")

        t0 = time.perf_counter()
        if sample.sample_id not in self.score_map:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return ModelInferenceResult(
                sample_id=sample.sample_id,
                model_name=self.name,
                model_version=self.version,
                raw_score=None,
                score_direction=self.score_direction,
                status=SampleStatus.ERROR,
                prediction=PredictionOutcome.ERROR,
                error_type="MISSING_SCORE_IN_CSV",
                error_message=f"Sample '{sample.sample_id}' not found in score CSV.",
                timing=TimingRecord(total_latency_ms=elapsed_ms),
            )

        score, recorded_lat, extra = self.score_map[sample.sample_id]
        calc_lat_ms = (time.perf_counter() - t0) * 1000.0
        final_lat = recorded_lat if recorded_lat is not None else calc_lat_ms

        return ModelInferenceResult(
            sample_id=sample.sample_id,
            model_name=self.name,
            model_version=self.version,
            raw_score=score,
            score_direction=self.score_direction,
            status=SampleStatus.PASS,
            prediction=PredictionOutcome.ACCEPT if score >= 0.5 else PredictionOutcome.REJECT,
            timing=TimingRecord(total_latency_ms=final_lat),
            extra_metadata={"source_csv_row": extra},
        )
