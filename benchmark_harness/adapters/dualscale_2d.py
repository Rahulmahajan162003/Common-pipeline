"""
DualScalePatchNet / 2D Signal Model Adapter for single-finger evaluation.
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


class DualScale2DAdapter(BaseModelAdapter):
    """Adapter for DualScalePatchNet / 2D single-finger liveness and matching models."""

    def __init__(
        self,
        name: str = "dualscale_2d",
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
            "resize": [224, 224],
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "color_format": "RGB",
            "patch_scales": [1.0, 0.5],
        }
        self.model = None
        self.torch_device = "cpu"

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Loads PyTorch model checkpoint or initializes model architecture."""
        import torch
        if self.device == "auto":
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.torch_device = self.device

        if not checkpoint_path:
            raise ValueError(f"[DualScale2D] 'checkpoint_path' must be provided. Fallback simulation is strictly disabled.")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[DualScale2D] Checkpoint not found: '{checkpoint_path}'. Fallback simulation is strictly disabled.")

        try:
            chk = torch.load(checkpoint_path, map_location=self.torch_device, weights_only=False)
            if isinstance(chk, dict) and "model_state_dict" in chk:
                self.model = chk["model_state_dict"]
            else:
                self.model = chk
            if hasattr(self.model, "eval"):
                self.model.eval()
            self._is_loaded = True
            print(f"[DualScale2D] Successfully loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            raise RuntimeError(f"[DualScale2D] Failed to load model from '{checkpoint_path}': {str(e)}") from e

    def warmup(self, warmup_runs: int = 3) -> None:
        """Warms up GPU/CPU caches."""
        try:
            import torch
            dummy = torch.randn(1, 3, 224, 224, device=self.torch_device)
            for _ in range(warmup_runs):
                if self.model is not None and hasattr(self.model, "forward"):
                    with torch.no_grad():
                        _ = self.model(dummy)
            if self.torch_device.startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:
            pass
        self._warmup_done = True

    def preprocess(self, sample: ManifestSample) -> Any:
        """Loads image, resizes, normalizes."""
        if not os.path.exists(sample.image_path):
            raise FileNotFoundError(f"Image not found: {sample.image_path}")
        img = Image.open(sample.image_path).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (arr - mean) / std
        return norm

    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        t_start = time.perf_counter()
        prep_lat_ms = 0.0
        infer_lat_ms = 0.0
        post_lat_ms = 0.0

        try:
            if self.model is None:
                raise RuntimeError(
                    f"[DualScale2D] Model '{self.name}' is not loaded. Fallback scoring is strictly disabled."
                )

            # 1. Preprocessing
            t0 = time.perf_counter()
            processed_data = self.preprocess(sample)
            prep_lat_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Inference
            t1 = time.perf_counter()
            import torch
            tensor = torch.from_numpy(processed_data).permute(2, 0, 1).unsqueeze(0).to(self.torch_device)
            with torch.no_grad():
                out = self.model(tensor)
                if hasattr(out, "logits"):
                    out = out.logits
                if isinstance(out, (list, tuple)):
                    out = out[0]

                if out.shape[-1] > 1:
                    probs = torch.softmax(out, dim=-1)
                    # Class 0: Live / Bonafide, Class 1: Attack / Spoof
                    live_prob = float(probs[0, 0].item())
                    spoof_prob = float(probs[0, 1].item())
                else:
                    sig = float(torch.sigmoid(out).item())
                    spoof_prob = sig
                    live_prob = 1.0 - sig

                if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    raw_score = live_prob
                else:
                    raw_score = spoof_prob

            if self.torch_device.startswith("cuda"):
                torch.cuda.synchronize()
            infer_lat_ms = (time.perf_counter() - t1) * 1000.0

            # 3. Postprocessing
            t2 = time.perf_counter()
            pred = PredictionOutcome.ACCEPT if (
                raw_score >= 0.5 if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT else raw_score <= 0.5
            ) else PredictionOutcome.REJECT
            post_lat_ms = (time.perf_counter() - t2) * 1000.0

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
                    postprocess_latency_ms=post_lat_ms,
                ),
                device=self.torch_device,
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
                device=self.torch_device,
            )
