"""
DualScalePatchNet Model Adapter for single-finger evaluation.
Uses native DualScalePatchNet architecture and exact preprocessing from training pipeline.
"""

import os
import sys
import time
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2
import torch
import torch.nn.functional as F

from .base import BaseModelAdapter
from ..schemas import (
    ManifestSample,
    ModelInferenceResult,
    ScoreDirection,
    SampleStatus,
    PredictionOutcome,
    TimingRecord,
)

FINE_SIZE = 96
MACRO_SIZE = 160


def _get_dualscale_model_cls():
    """Dynamically imports DualScalePatchNet class."""
    try:
        from backend.pad.models import DualScalePatchNet
        return DualScalePatchNet
    except ImportError:
        pass

    # Try reference_code path
    possible_dirs = [
        os.path.join(os.getcwd(), "backend"),
        os.path.join(os.getcwd(), "reference_code", "uidai-leader-work", "backend"),
    ]
    for p in possible_dirs:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from pad.models import DualScalePatchNet
        return DualScalePatchNet
    except ImportError as e:
        raise ImportError(f"[DualScale2D] Failed to import DualScalePatchNet from backend.pad.models: {e}") from e


class DualScale2DAdapter(BaseModelAdapter):
    """Adapter for DualScalePatchNet single-finger PAD evaluation."""

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
            "patch_fine": [FINE_SIZE, FINE_SIZE],
            "patch_macro": [MACRO_SIZE, MACRO_SIZE],
            "num_patches": 5,
            "channels": 6,  # RGB + YCrCb
            "normalization": {"mean": 0.50, "std": 0.25},
        }
        self.model = None
        self.torch_device = "cpu"

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Loads PyTorch DualScalePatchNet model and weights."""
        if self.device == "auto":
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.torch_device = self.device

        if not checkpoint_path:
            raise ValueError(f"[DualScale2D] 'checkpoint_path' must be provided. Fallback simulation is strictly disabled.")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[DualScale2D] Checkpoint not found: '{checkpoint_path}'. Fallback simulation is strictly disabled.")

        resolved_checkpoint = checkpoint_path

        try:
            ModelCls = _get_dualscale_model_cls()
            self.model = ModelCls(num_classes=3, in_channels=6).to(self.torch_device)
            chk = torch.load(resolved_checkpoint, map_location=self.torch_device, weights_only=False)
            state_dict = chk.get("model_state_dict", chk) if isinstance(chk, dict) else chk
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self._is_loaded = True
            print(f"[DualScale2D] Successfully loaded DualScalePatchNet weights from {resolved_checkpoint} onto {self.torch_device}")
        except Exception as e:
            raise RuntimeError(f"[DualScale2D] Failed to load DualScalePatchNet from '{resolved_checkpoint}': {str(e)}") from e

    def warmup(self, warmup_runs: int = 3) -> None:
        """Warms up GPU/CPU caches."""
        if self.model is None:
            return
        try:
            dummy_fine = torch.randn(5, 6, FINE_SIZE, FINE_SIZE, device=self.torch_device)
            dummy_macro = torch.randn(5, 6, FINE_SIZE, FINE_SIZE, device=self.torch_device)
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = self.model(dummy_fine, dummy_macro)
            if str(self.torch_device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:
            pass
        self._warmup_done = True

    def preprocess(self, sample: ManifestSample) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Exact KT Preprocessing from train_dualscale_patchnet.py:
        1. Read BGR image
        2. Convert to RGB and YCrCb, concatenate -> 6-channel
        3. Extract 5 deterministic dual patches (Fine: 96x96, Macro: 160x160 -> resized to 96x96)
        4. Normalize: (patch / 255.0 - 0.5) / 0.25
        """
        if not os.path.exists(sample.image_path):
            raise FileNotFoundError(f"Image not found: {sample.image_path}")

        img_bgr = cv2.imread(sample.image_path)
        if img_bgr is None:
            raise ValueError(f"cv2.imread failed to read image: {sample.image_path}")

        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        img_6c = np.concatenate([img_rgb, img_ycrcb], axis=2)

        coords = [
            (h // 2, w // 2),
            (int(h * 0.25), int(w * 0.25)),
            (int(h * 0.25), int(w * 0.75)),
            (int(h * 0.75), int(w * 0.25)),
            (int(h * 0.75), int(w * 0.75)),
        ]

        fines: List[torch.Tensor] = []
        macros: List[torch.Tensor] = []

        for cy, cx in coords:
            # 1. Fine patch (96x96)
            y1 = max(0, min(cy - FINE_SIZE // 2, h - FINE_SIZE))
            x1 = max(0, min(cx - FINE_SIZE // 2, w - FINE_SIZE))
            fine_patch = img_6c[y1:y1 + FINE_SIZE, x1:x1 + FINE_SIZE]
            if fine_patch.shape[:2] != (FINE_SIZE, FINE_SIZE):
                fine_patch = cv2.resize(fine_patch, (FINE_SIZE, FINE_SIZE))

            # 2. Context macro patch (160x160, resized to 96x96)
            my1 = max(0, min(cy - MACRO_SIZE // 2, h - MACRO_SIZE))
            mx1 = max(0, min(cx - MACRO_SIZE // 2, w - MACRO_SIZE))
            macro_patch = img_6c[my1:my1 + MACRO_SIZE, mx1:mx1 + MACRO_SIZE]
            macro_patch = cv2.resize(macro_patch, (FINE_SIZE, FINE_SIZE))

            # Normalize: (t - 0.5) / 0.25
            tf = (torch.from_numpy(fine_patch.transpose(2, 0, 1)).float() / 255.0 - 0.50) / 0.25
            tm = (torch.from_numpy(macro_patch.transpose(2, 0, 1)).float() / 255.0 - 0.50) / 0.25
            fines.append(tf)
            macros.append(tm)

        batch_fine = torch.stack(fines).to(self.torch_device)
        batch_macro = torch.stack(macros).to(self.torch_device)
        return batch_fine, batch_macro

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
            batch_fine, batch_macro = self.preprocess(sample)
            prep_lat_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Inference
            t1 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(batch_fine, batch_macro)
                probs = F.softmax(logits, dim=1).cpu().numpy()

                # 5-patch mean probabilities (Class 0: Live, 1: Screen replay, 2: Paper print)
                mean_probs = probs.mean(axis=0)
                p_live = float(mean_probs[0])
                p_screen = float(mean_probs[1])
                p_paper = float(mean_probs[2])

                # Live score = 1 - (P(screen) + P(paper))
                live_score = 1.0 - (p_screen + p_paper)
                attack_score = p_screen + p_paper

                if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    raw_score = live_score
                else:
                    raw_score = attack_score

            if str(self.torch_device).startswith("cuda"):
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
                device=str(self.torch_device),
                extra_metadata={
                    "p_live": round(p_live, 6),
                    "p_screen": round(p_screen, 6),
                    "p_paper": round(p_paper, 6),
                    "p_attack": round(attack_score, 6),
                    "preprocessing": self.preprocessing_config,
                },
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
                device=str(self.torch_device),
            )
