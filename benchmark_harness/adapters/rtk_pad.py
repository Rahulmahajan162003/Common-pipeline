"""
RTK-PAD (GlobalLocalRTKPAD) Presentation Attack Detection Adapter.
Implements:
1. Slap and single-finger detection using YOLO (best-new.pt)
2. Global-Local Dual Attention scoring via GlobalLocalRTKPAD (pad_rtk_global_local_best.pt)
3. Zero-Tolerance / Weakest-Link slap aggregation: slap_live = min(finger_live_scores)
"""

import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from .base import BaseModelAdapter
from ..schemas import (
    ManifestSample,
    ModelInferenceResult,
    ScoreDirection,
    SampleStatus,
    PredictionOutcome,
    TimingRecord,
)

# Standard ImageNet transform for RTK-PAD 224x224 input
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
TF_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


def _get_rtk_model_cls():
    """Dynamically imports GlobalLocalRTKPAD class."""
    try:
        from backend.pad.models_rtk import GlobalLocalRTKPAD, build_rtk_pad_model
        return GlobalLocalRTKPAD, build_rtk_pad_model
    except ImportError:
        pass

    possible_dirs = [
        os.path.join(os.getcwd(), "backend"),
        os.path.join(os.getcwd(), "reference_code", "uidai-leader-work", "backend"),
    ]
    for p in possible_dirs:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)

    try:
        from pad.models_rtk import GlobalLocalRTKPAD, build_rtk_pad_model
        return GlobalLocalRTKPAD, build_rtk_pad_model
    except ImportError as e:
        raise ImportError(f"[RTK-PAD] Failed to import GlobalLocalRTKPAD: {e}") from e


def _get_yolo_detector():
    """Loads YOLO finger detector."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError(f"[RTK-PAD] ultralytics package is required for finger detection: {e}") from e

    possible_weights = [
        os.path.join("models", "best-new.pt"),
        os.path.join("backend", "best-new.pt"),
        os.path.join("backend", "models", "best-new.pt"),
        os.path.join("reference_code", "uidai-leader-work", "backend", "best-new.pt"),
    ]
    for w in possible_weights:
        if os.path.exists(w):
            return YOLO(w)
    raise FileNotFoundError(f"[RTK-PAD] YOLO detector weights 'best-new.pt' not found in: {possible_weights}")


class RtkPadAdapter(BaseModelAdapter):
    """Adapter for RTK-PAD GlobalLocal model on slap and single finger images."""

    def __init__(
        self,
        name: str = "rtk_pad",
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
            "normalization": {"mean": list(MEAN), "std": list(STD)},
            "color_format": "RGB",
            "slap_aggregation": "zero_tolerance_min_live",
        }
        self.model: Optional[torch.nn.Module] = None
        self.detector = None
        self.torch_device = "cpu"

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Loads GlobalLocalRTKPAD and YOLO finger detector."""
        if self.device == "auto":
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.torch_device = self.device

        if not checkpoint_path:
            raise ValueError(f"[RTK-PAD] 'checkpoint_path' must be provided. Fallback simulation is strictly disabled.")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"[RTK-PAD] Checkpoint not found: '{checkpoint_path}'. Fallback simulation is strictly disabled.")

        resolved_checkpoint = checkpoint_path

        try:
            _, build_fn = _get_rtk_model_cls()
            self.model = build_fn(num_classes=2, pretrained=False).to(self.torch_device)
            ckpt = torch.load(resolved_checkpoint, map_location=self.torch_device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"[RTK-PAD] Loaded GlobalLocalRTKPAD weights from {resolved_checkpoint} onto {self.torch_device}")
        except Exception as e:
            raise RuntimeError(f"[RTK-PAD] Failed to load GlobalLocalRTKPAD from '{resolved_checkpoint}': {e}") from e

        # Load YOLO detector
        try:
            self.detector = _get_yolo_detector()
            print("[RTK-PAD] Loaded YOLO finger detector (best-new.pt)")
        except Exception as e:
            print(f"[RTK-PAD] Warning: YOLO finger detector not loaded ({e}). Direct whole-image scoring fallback will be used.")

        self._is_loaded = True

    def warmup(self, warmup_runs: int = 3) -> None:
        """Warms up model and hardware caches."""
        if self.model is None:
            return
        dummy = torch.randn(1, 3, 224, 224, device=self.torch_device)
        try:
            with torch.no_grad():
                for _ in range(warmup_runs):
                    _ = self.model(dummy)
            if str(self.torch_device).startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:
            pass
        self._warmup_done = True

    def detect_slap_fingers(self, im_bgr: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        """
        Detects finger crops in slap image using YOLO detector.
        Applies multi-scale detection, NMS filtering, and sorts fingers left to right.
        """
        if self.detector is None:
            return [(im_bgr, (0, 0, im_bgr.shape[1], im_bgr.shape[0]), 1.0)]

        all_boxes = []
        for sz in [640, 800, 1024]:
            try:
                res = self.detector(im_bgr, imgsz=sz, conf=0.15, verbose=False)
                boxes = res[0].boxes
                if len(boxes) > 0:
                    for b, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
                        h, w = b[3] - b[1], b[2] - b[0]
                        if h > 50 and w > 50:
                            all_boxes.append((b, float(c)))
            except Exception:
                pass

        kept = []
        for b, c in sorted(all_boxes, key=lambda x: x[1], reverse=True):
            ax1, ay1, ax2, ay2 = b
            overlap = False
            for kb, kc in kept:
                kx1, ky1, kx2, ky2 = kb
                ix1, iy1 = max(ax1, kx1), max(ay1, ky1)
                ix2, iy2 = min(ax2, kx2), min(ay2, ky2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter = iw * ih
                union = (ax2 - ax1) * (ay2 - ay1) + (kx2 - kx1) * (ky2 - ky1) - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.35:
                    overlap = True
                    break
            if not overlap:
                kept.append((b, c))

        kept.sort(key=lambda x: (x[0][0] + x[0][2]) / 2.0)

        h_img, w_img = im_bgr.shape[:2]
        crops = []
        for b, conf in kept:
            x1, y1, x2, y2 = b.astype(int)
            margin = 15
            cx1 = max(0, x1 - margin)
            cy1 = max(0, y1 - margin)
            cx2 = min(w_img, x2 + margin)
            cy2 = min(h_img, y2 + margin)
            crop = im_bgr[cy1:cy2, cx1:cx2]
            if crop.size > 0:
                crops.append((crop, (cx1, cy1, cx2, cy2), conf))

        if not crops:
            crops.append((im_bgr, (0, 0, w_img, h_img), 0.0))

        return crops

    def score_single_crop(self, crop_bgr: np.ndarray) -> Tuple[float, float]:
        """Evaluates one crop with GlobalLocalRTKPAD. Returns (p_live, p_attack)."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb)
        t_224 = TF_224(im_pil).unsqueeze(0).to(self.torch_device)

        with torch.no_grad():
            out = self.model(t_224)
            p_attack = float(out["prob_attack"][0].cpu().item())
            p_live = 1.0 - p_attack

        return p_live, p_attack

    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        t_start = time.perf_counter()
        prep_lat_ms = 0.0
        infer_lat_ms = 0.0
        post_lat_ms = 0.0

        try:
            if self.model is None:
                raise RuntimeError("[RTK-PAD] Model weights are not loaded. Fallback scoring is strictly disabled.")

            if not os.path.exists(sample.image_path):
                raise FileNotFoundError(f"Image not found: {sample.image_path}")

            # 1. Preprocessing & Finger Detection
            t0 = time.perf_counter()
            img_bgr = cv2.imread(sample.image_path)
            if img_bgr is None:
                raise ValueError(f"cv2.imread failed to read image: {sample.image_path}")

            finger_crops = self.detect_slap_fingers(img_bgr)
            prep_lat_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Inference on each finger crop
            t1 = time.perf_counter()
            finger_scores: List[Dict[str, float]] = []
            for idx, (crop, bbox, conf) in enumerate(finger_crops, start=1):
                p_live, p_attack = self.score_single_crop(crop)
                finger_scores.append({
                    "finger_index": idx,
                    "yolo_conf": round(conf, 3),
                    "p_live": round(p_live, 6),
                    "p_attack": round(p_attack, 6),
                    "bbox": list(bbox),
                })

            if str(self.torch_device).startswith("cuda"):
                torch.cuda.synchronize()
            infer_lat_ms = (time.perf_counter() - t1) * 1000.0

            # 3. Slap Fusion Rule: Zero-Tolerance / Weakest-Link (min live score)
            t2 = time.perf_counter()
            if finger_scores:
                slap_live = min(f["p_live"] for f in finger_scores)
                slap_attack = max(f["p_attack"] for f in finger_scores)
            else:
                slap_live = 0.0
                slap_attack = 1.0

            if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                raw_score = slap_live
            else:
                raw_score = slap_attack

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
                    "slap_live_score": round(slap_live, 6),
                    "slap_attack_score": round(slap_attack, 6),
                    "fingers_detected": len(finger_crops),
                    "finger_breakdown": finger_scores,
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
