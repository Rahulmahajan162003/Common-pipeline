"""
RTK-PAD (Real-Time Knowledge-guided Presentation Attack Detection) Model Adapter.
Evaluates MobileNetV3-based PAD models on single-finger RGB images.
"""

import os
import time
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
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


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.sigmoid = Hsigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            Hsigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        Hswish(),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        Hswish(),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup
        if inp == hidden_dim:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SELayer(hidden_dim) if use_se else nn.Identity(),
                Hswish() if use_hs else nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3PAD(nn.Module):
    def __init__(self, num_classes=2, width_mult=1.0):
        super(MobileNetV3PAD, self).__init__()
        self.cfgs = [
            # k, t, c, SE, HS, s
            [3,   1,  16, 0, 0, 1],
            [3,   4,  24, 0, 0, 2],
            [3,   3,  24, 0, 0, 1],
            [5,   3,  40, 1, 0, 2],
            [5,   3,  40, 1, 0, 1],
            [5,   3,  40, 1, 0, 1],
            [3,   6,  80, 0, 1, 2],
            [3, 2.5,  80, 0, 1, 1],
            [3, 2.3,  80, 0, 1, 1],
            [3, 2.3,  80, 0, 1, 1],
            [3,   6, 112, 1, 1, 1],
            [3,   6, 112, 1, 1, 1],
            [5,   6, 160, 1, 1, 2],
            [5,   6, 160, 1, 1, 1],
            [5,   6, 160, 1, 1, 1],
        ]
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = 1280
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            Hswish(),
            nn.Dropout(0.2),
        )
        self.linear = nn.Linear(output_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.linear(x)
        return x


class RtkPadAdapter(BaseModelAdapter):
    """Adapter for RTK-PAD MobileNetV3 presentation attack detection model."""

    DEFAULT_CHECKPOINTS = [
        r"D:\rtk_pad_colfispoof_weights.pth",
        r"D:\RTK-PAD-main\RTK-PAD-main\checkpoints\colfispoof\rtk_pad_colfispoof_best.pth",
    ]

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
            "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "color_format": "RGB",
        }
        self.model: Optional[nn.Module] = None
        self.torch_device = "cpu"

    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Loads RTK-PAD MobileNetV3 weights."""
        if self.device == "auto":
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.torch_device = self.device

        self.model = MobileNetV3PAD(num_classes=2)

        # Resolve checkpoint path
        target_path = checkpoint_path
        if not target_path or not os.path.exists(target_path):
            raise FileNotFoundError(
                f"[RTK-PAD] Required model checkpoint not found: '{checkpoint_path}'. "
                f"Automatic fallback scoring is strictly disabled."
            )

        try:
            ckpt = torch.load(target_path, map_location=self.torch_device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            print(f"[RTK-PAD] Loaded weights successfully from {target_path}")
        except Exception as e:
            raise RuntimeError(f"[RTK-PAD] Failed to load checkpoint from {target_path}: {e}") from e

        self.model.to(self.torch_device)
        self.model.eval()
        self._is_loaded = True

    def warmup(self, warmup_runs: int = 3) -> None:
        """Warm up PyTorch graph and hardware caches."""
        if self.model is None:
            return
        dummy = torch.randn(1, 3, 224, 224, device=self.torch_device)
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(dummy)
        if self.torch_device.startswith("cuda"):
            torch.cuda.synchronize()
        self._warmup_done = True

    def preprocess(self, sample: ManifestSample) -> torch.Tensor:
        """Loads image, resizes to (224, 224), normalizes with ImageNet parameters."""
        if not os.path.exists(sample.image_path):
            raise FileNotFoundError(f"Image not found: {sample.image_path}")

        img = Image.open(sample.image_path).convert("RGB")
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm = (arr - mean) / std
        # Convert HWC to CHW tensor
        tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(self.torch_device)
        return tensor

    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        t_start = time.perf_counter()
        prep_lat_ms = 0.0
        infer_lat_ms = 0.0
        post_lat_ms = 0.0

        try:
            if self.model is None:
                raise RuntimeError("[RTK-PAD] Model weights are not loaded. Fallback scoring is strictly disabled.")

            # 1. Preprocessing
            t0 = time.perf_counter()
            tensor = self.preprocess(sample)
            prep_lat_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Inference
            t1 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=-1)
                # Class 0 = Live (Bonafide), Class 1 = Spoof (Attack)
                live_prob = float(probs[0, 0].item())
                spoof_prob = float(probs[0, 1].item())

                if self.score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
                    raw_score = live_prob
                else:
                    raw_score = spoof_prob

            if self.torch_device.startswith("cuda"):
                torch.cuda.synchronize()
            infer_lat_ms = (time.perf_counter() - t1) * 1000.0

            # 3. Decision & Postprocessing
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
