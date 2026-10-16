"""
ShuffleNetV2 PAD adapter (single finger) - mirrors the production app flow.

App flow (confirmed by the app developer):
1. Detect the finger with the finger detector (input 512, confidence >= 0.10).
2. Expand the best box: 8% of box width left/right, 18% of box height on top,
   8% of box height at the bottom (clipped to the image).
3. Resize the crop to 224x224, ImageNet normalisation (same as KT eval code).
4. One ShuffleNetV2 inference, softmax: index 0 = live, index 1 = spoof.
5. Live decision: live_score >= threshold (0.656550407409668 in production).
No retries, no patch aggregation.

If no finger is detected the sample is reported with status FAIL (NoFingerDetected):
a capture failure, counted separately. There is no whole-image fallback.
"""

import os
import time
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from .base import BaseModelAdapter
from ..schemas import (
    ManifestSample,
    ModelInferenceResult,
    ScoreDirection,
    SampleStatus,
    PredictionOutcome,
    TimingRecord,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TF_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

DEFAULT_DETECTOR_PATHS = [
    os.path.join("models", "best-new.pt"),
    os.path.join("backend", "best-new.pt"),
    os.path.join("backend", "models", "best-new.pt"),
    os.path.join("reference_code", "uidai-leader-work", "backend", "best-new.pt"),
]


class NoFingerDetected(Exception):
    """Raised when the detector finds no finger in the image."""


def build_shufflenet_pad() -> nn.Module:
    """Same architecture as KT build_pad_model('shufflenet_v2_x1_0'): fc -> 2 classes."""
    model = models.shufflenet_v2_x1_0(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


class ShuffleNetPadAdapter(BaseModelAdapter):
    """ShuffleNetV2 single-finger PAD with the app's finger crop."""

    def __init__(
        self,
        name: str = "shufflenet_pad",
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
        p = self.extra_params or {}
        self.finger_crop = str(p.get("finger_crop", "yolo")).lower().strip()
        if self.finger_crop not in ("yolo", "none"):
            raise ValueError(f"[ShuffleNetPAD] Invalid finger_crop '{self.finger_crop}'. Must be 'yolo' or 'none'.")
        self.detector_path = p.get("detector_path")
        self.detector_imgsz = int(p.get("detector_imgsz", 512))
        self.detector_conf = float(p.get("detector_conf", 0.10))
        self.margin_side = float(p.get("margin_side", 0.08))
        self.margin_top = float(p.get("margin_top", 0.18))
        self.margin_bottom = float(p.get("margin_bottom", 0.08))

        self.preprocessing_config = {
            "finger_crop": self.finger_crop,
            "detector_imgsz": self.detector_imgsz,
            "detector_conf": self.detector_conf,
            "margins": {"side": self.margin_side, "top": self.margin_top, "bottom": self.margin_bottom},
            "resize": [224, 224],
            "normalization": {"mean": list(MEAN), "std": list(STD)},
            "live_index": 0,
        }
        self.model: Optional[nn.Module] = None
        self.detector = None
        self.torch_device = "cpu"

    # ------------------------------------------------------------------ load
    def load(self, checkpoint_path: Optional[str] = None) -> None:
        self.torch_device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device

        if not checkpoint_path:
            raise ValueError("[ShuffleNetPAD] 'checkpoint_path' must be provided.")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[ShuffleNetPAD] Checkpoint not found: '{checkpoint_path}'.")

        try:
            model = build_shufflenet_pad()
            ckpt = torch.load(checkpoint_path, map_location=self.torch_device, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state)  # strict: any mismatch is an error
            self.model = model.to(self.torch_device).eval()
            print(f"[ShuffleNetPAD] Loaded weights from {checkpoint_path} onto {self.torch_device}")
        except Exception as e:
            raise RuntimeError(f"[ShuffleNetPAD] Failed to load '{checkpoint_path}': {e}") from e

        if self.finger_crop == "yolo":
            paths = [self.detector_path] if self.detector_path else DEFAULT_DETECTOR_PATHS
            found = next((w for w in paths if w and os.path.exists(w)), None)
            if found is None:
                raise FileNotFoundError(f"[ShuffleNetPAD] Finger detector not found in: {paths}. Hidden fallbacks are disabled.")
            try:
                from ultralytics import YOLO
                self.detector = YOLO(found)
                print(f"[ShuffleNetPAD] Loaded finger detector from {found}")
            except Exception as e:
                raise RuntimeError(f"[ShuffleNetPAD] Failed to load finger detector '{found}': {e}") from e

        self._is_loaded = True

    def warmup(self, warmup_runs: int = 3) -> None:
        if self.model is None:
            return
        dummy = torch.randn(1, 3, 224, 224, device=self.torch_device)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy)
        self._warmup_done = True

    # ------------------------------------------------------------ preprocess
    def crop_finger(self, im_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int], float]:
        """App crop: best detection, asymmetric margins, clipped to the image."""
        res = self.detector(im_bgr, imgsz=self.detector_imgsz, conf=self.detector_conf, verbose=False)
        boxes = res[0].boxes
        if boxes is None or len(boxes) == 0:
            raise NoFingerDetected("No finger detected")
        confs = boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[i].tolist()
        bw, bh = x2 - x1, y2 - y1
        h, w = im_bgr.shape[:2]
        cx1 = int(max(0, round(x1 - self.margin_side * bw)))
        cx2 = int(min(w, round(x2 + self.margin_side * bw)))
        cy1 = int(max(0, round(y1 - self.margin_top * bh)))
        cy2 = int(min(h, round(y2 + self.margin_bottom * bh)))
        crop = im_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            raise NoFingerDetected("Empty crop after detection")
        return crop, (cx1, cy1, cx2, cy2), float(confs[i])

    def score_crop(self, crop_bgr: np.ndarray) -> Tuple[float, float]:
        """Returns (p_live, p_spoof) for one crop."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        t = TF_224(Image.fromarray(rgb)).unsqueeze(0).to(self.torch_device)
        with torch.no_grad():
            probs = F.softmax(self.model(t), dim=1)[0].cpu().numpy()
        return float(probs[0]), float(probs[1])

    # --------------------------------------------------------------- predict
    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        t_start = time.perf_counter()
        try:
            if self.model is None:
                raise RuntimeError("[ShuffleNetPAD] Model is not loaded.")
            if not os.path.exists(sample.image_path):
                raise FileNotFoundError(f"Image not found: {sample.image_path}")
            im = cv2.imread(sample.image_path)
            if im is None:
                raise ValueError(f"cv2.imread failed to read image: {sample.image_path}")

            t0 = time.perf_counter()
            meta: Dict[str, Any] = {}
            if self.finger_crop == "yolo":
                crop, bbox, conf = self.crop_finger(im)
                meta.update({"crop_bbox": list(bbox), "detector_conf": round(conf, 3)})
            else:
                crop = im
            prep_ms = (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            p_live, p_spoof = self.score_crop(crop)
            infer_ms = (time.perf_counter() - t1) * 1000.0

            raw = p_live if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT else p_spoof
            pred = PredictionOutcome.ACCEPT if (
                raw >= 0.5 if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT else raw <= 0.5
            ) else PredictionOutcome.REJECT

            return ModelInferenceResult(
                sample_id=sample.sample_id,
                model_name=self.name,
                model_version=self.version,
                raw_score=raw,
                score_direction=self.score_direction,
                status=SampleStatus.PASS,
                prediction=pred,
                timing=TimingRecord(
                    total_latency_ms=(time.perf_counter() - t_start) * 1000.0,
                    preprocess_latency_ms=prep_ms,
                    inference_latency_ms=infer_ms,
                    postprocess_latency_ms=0.0,
                ),
                device=str(self.torch_device),
                extra_metadata={
                    "p_live": round(p_live, 6),
                    "p_spoof": round(p_spoof, 6),
                    "preprocessing": self.preprocessing_config,
                    **meta,
                },
            )
        except NoFingerDetected as e:
            # Capture failure, not a crash: in the app the user would simply retake.
            # Reported as FAIL so it is counted separately and never scored as live or fake.
            return ModelInferenceResult(
                sample_id=sample.sample_id,
                model_name=self.name,
                model_version=self.version,
                raw_score=None,
                score_direction=self.score_direction,
                status=SampleStatus.FAIL,
                prediction=PredictionOutcome.ERROR,
                error_type="NoFingerDetected",
                error_message=str(e),
                timing=TimingRecord(total_latency_ms=(time.perf_counter() - t_start) * 1000.0),
                device=str(self.torch_device),
            )
        except Exception as e:
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
                timing=TimingRecord(total_latency_ms=(time.perf_counter() - t_start) * 1000.0),
                device=str(self.torch_device),
            )
