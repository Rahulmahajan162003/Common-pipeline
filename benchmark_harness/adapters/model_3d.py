"""
3D Fingerprint Representation / Depth Model Adapter.
"""

import os
import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from .base import BaseModelAdapter
from ..schemas import (
    ManifestSample,
    ModelInferenceResult,
    ScoreDirection,
    SampleStatus,
    PredictionOutcome,
    TimingRecord,
)


class Model3DAdapter(BaseModelAdapter):
    """Adapter for 3D Shape-from-Shading / Photometric Stereo single-finger model."""

    def __init__(
        self,
        name: str = "model_3d",
        version: str = "1.0.0",
        device: str = "auto",
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            version=version,
            device=device,
            score_direction=score_direction,
            extra_params=extra_params,
        )
        self.preprocessing_config = {
            "surface_normal_estimation": True,
            "depth_map_resolution": [128, 128],
            "curvature_filtering": "gaussian_laplace",
        }
        self.model = None

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                import torch
                self.model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception:
                pass
        self._is_loaded = True

    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        t_start = time.perf_counter()
        try:
            if not os.path.exists(sample.image_path):
                raise FileNotFoundError(f"Image not found: {sample.image_path}")

            t0 = time.perf_counter()
            img = Image.open(sample.image_path).convert("L")
            img = img.resize((128, 128), Image.Resampling.BILINEAR)
            arr = np.array(img, dtype=np.float32) / 255.0
            prep_lat_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            if self.model is None or not callable(self.model):
                raise RuntimeError(f"[Model3D] Model is not loaded. Fallback scoring is strictly disabled.")

            import torch
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = self.model(tensor)
                raw_score = float(torch.sigmoid(out).item() if out.numel() == 1 else out[0, 0].item())
            infer_lat_ms = (time.perf_counter() - t1) * 1000.0

            pred = PredictionOutcome.ACCEPT if raw_score >= 0.5 else PredictionOutcome.REJECT
            total_lat_ms = (time.perf_counter() - t_start) * 1000.0

            return ModelInferenceResult(
                sample_id=sample.sample_id,
                model_name=self.name,
                model_version=self.version,
                raw_score=raw_score,
                score_direction=self.score_direction,
                status=SampleStatus.PASS,
                prediction=pred,
                timing=TimingRecord(
                    total_latency_ms=total_lat_ms,
                    preprocess_latency_ms=prep_lat_ms,
                    inference_latency_ms=infer_lat_ms,
                ),
                extra_metadata={"preprocessing": self.preprocessing_config},
            )

        except Exception as e:
            total_lat_ms = (time.perf_counter() - t_start) * 1000.0
            return ModelInferenceResult(
                sample_id=sample.sample_id,
                model_name=self.name,
                model_version=self.version,
                raw_score=None,
                score_direction=self.score_direction,
                status=SampleStatus.ERROR,
                prediction=PredictionOutcome.ERROR,
                error_type=type(e).__name__,
                error_message=str(e),
                timing=TimingRecord(total_latency_ms=total_lat_ms),
            )
