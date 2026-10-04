"""
LitMAS (Lightweight Multi-Modal Anti-Spoofing) Model Adapter.
Evaluates Vision Transformer (DeiT) based Mixture-of-Experts (MoE) PAD models on single-finger RGB images.
"""

import os
import sys
import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
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


class DeitWithClassifier(nn.Module):
    """
    LitMAS Mixture-of-Experts (MoE) Architecture with DeiT Backbone
    """
    def __init__(self, base_model):
        super(DeitWithClassifier, self).__init__()
        self.base_model = base_model
        self.iris_expert = nn.Linear(192, 512)
        self.face_expert = nn.Linear(192, 512)
        self.audio_expert = nn.Linear(192, 512)
        self.fingerprint_expert = nn.Linear(192, 512)
        self.classifier = nn.Linear(512, 2)

    def forward(self, pixel_values, modality_type="fingerprint"):
        outputs = self.base_model(pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        if modality_type == "fingerprint":
            embeddings = self.fingerprint_expert(cls_token)
        elif modality_type == "face":
            embeddings = self.face_expert(cls_token)
        elif modality_type == "iris":
            embeddings = self.iris_expert(cls_token)
        elif modality_type == "audio":
            embeddings = self.audio_expert(cls_token)
        else:
            embeddings = self.fingerprint_expert(cls_token)
            
        logits = self.classifier(embeddings)
        return logits, embeddings


class LitMasAdapter(BaseModelAdapter):
    """
    Adapter for LitMAS (Lightweight Multi-Modal Anti-Spoofing) Vision Transformer model.
    """

    def __init__(
        self,
        name: str = "litmas",
        version: str = "1.0.0",
        device: str = "auto",
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        checkpoint_path: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            name=name,
            version=version,
            device=device,
            score_direction=score_direction,
            extra_params=extra_params,
        )
        self.checkpoint_path = checkpoint_path or r"D:\LitMAS-main\LitMAS-main\checkpoints_colfispoof\litmas_colfispoof_best.pth"
        self.model: Optional[nn.Module] = None
        self.torch_device = torch.device("cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        if checkpoint_path:
            self.checkpoint_path = checkpoint_path

        # Resolve compute device
        if self.device == "cuda" and torch.cuda.is_available():
            self.torch_device = torch.device("cuda")
        elif self.device == "auto":
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.torch_device = torch.device("cpu")

        try:
            from transformers import DeiTModel
            base_deit = DeiTModel.from_pretrained("facebook/deit-tiny-distilled-patch16-224")
            self.model = DeitWithClassifier(base_model=base_deit)

            if self.checkpoint_path and os.path.exists(self.checkpoint_path):
                ckpt = torch.load(self.checkpoint_path, map_location=self.torch_device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)
                print(f"[LitMAS] Loaded weights successfully from {self.checkpoint_path}")
            else:
                print(f"[!] Warning: Checkpoint path not found ({self.checkpoint_path}), using random init")

            self.model.to(self.torch_device)
            self.model.eval()
            self._is_loaded = True

        except Exception as e:
            print(f"[!] Failed to initialize LitMAS model: {e}")
            raise

    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        if not self._is_loaded or self.model is None:
            self.load()

        t_start = time.perf_counter()
        try:
            if not os.path.exists(sample.image_path):
                raise FileNotFoundError(f"Image path does not exist: {sample.image_path}")

            # 1. Preprocessing
            t0 = time.perf_counter()
            with Image.open(sample.image_path) as img:
                img_rgb = img.convert("RGB")
                tensor = self.transform(img_rgb).unsqueeze(0).to(self.torch_device)
            prep_lat_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Inference
            t1 = time.perf_counter()
            with torch.no_grad():
                logits, _ = self.model(tensor, modality_type="fingerprint")
                probs = F.softmax(logits, dim=1)
                # In LitMAS colfispoof training: class 0 is Bonafide (Live), class 1 is Attack (Spoof)
                # ScoreDirection.HIGHER_IS_ACCEPT means live probability (class 0)
                live_prob = float(probs[0, 0].item())
                spoof_prob = float(probs[0, 1].item())

                if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    raw_score = live_prob
                else:
                    raw_score = spoof_prob

            infer_lat_ms = (time.perf_counter() - t1) * 1000.0
            total_lat_ms = (time.perf_counter() - t_start) * 1000.0

            # Default decision using 0.5 prior before harness calibration
            pred = PredictionOutcome.ACCEPT if raw_score >= 0.5 else PredictionOutcome.REJECT

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
                extra_metadata={
                    "live_probability": live_prob,
                    "spoof_probability": spoof_prob,
                    "modality": "fingerprint",
                    "device": str(self.torch_device),
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
            )
