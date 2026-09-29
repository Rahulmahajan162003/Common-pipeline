"""
Standardized image preprocessing pipelines and configurations.
Ensures identical transformations across benchmarking runs.
"""

import os
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass, field
import numpy as np
from PIL import Image


@dataclass
class StandardPreprocessConfig:
    target_size: Tuple[int, int] = (224, 224)
    color_mode: str = "RGB"  # "RGB", "L" (grayscale), "BGR"
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    normalize: bool = True
    crop_center: bool = False
    crop_size: Optional[Tuple[int, int]] = None
    pad_if_needed: bool = False


class ImagePreprocessor:
    """Deterministic image preprocessor with explicit configuration logging."""

    def __init__(self, config: Optional[StandardPreprocessConfig] = None):
        self.config = config or StandardPreprocessConfig()

    def process_file(self, image_path: str) -> np.ndarray:
        """Reads image file and applies transformations."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        img = Image.open(image_path)
        return self.process_image(img)

    def process_image(self, img: Image.Image) -> np.ndarray:
        """Applies configured transformations to a PIL Image."""
        # 1. Color mode conversion
        if self.config.color_mode == "RGB":
            img = img.convert("RGB")
        elif self.config.color_mode in ("L", "GRAY", "GRAYSCALE"):
            img = img.convert("L")

        # 2. Crop
        if self.config.crop_center and self.config.crop_size:
            w, h = img.size
            cw, ch = self.config.crop_size
            left = max(0, (w - cw) // 2)
            top = max(0, (h - ch) // 2)
            right = min(w, left + cw)
            bottom = min(h, top + ch)
            img = img.crop((left, top, right, bottom))

        # 3. Resize
        if self.config.target_size:
            img = img.resize(self.config.target_size, Image.Resampling.BILINEAR)

        # 4. Array conversion & normalization
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.config.normalize:
            if self.config.color_mode == "RGB":
                mean = np.array(self.config.mean, dtype=np.float32)
                std = np.array(self.config.std, dtype=np.float32)
                arr = (arr - mean) / std
            elif self.config.color_mode in ("L", "GRAY"):
                mean_val = float(np.mean(self.config.mean)) if self.config.mean else 0.5
                std_val = float(np.mean(self.config.std)) if self.config.std else 0.5
                arr = (arr - mean_val) / std_val

        return arr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_size": list(self.config.target_size),
            "color_mode": self.config.color_mode,
            "mean": self.config.mean,
            "std": self.config.std,
            "normalize": self.config.normalize,
            "crop_center": self.config.crop_center,
            "crop_size": list(self.config.crop_size) if self.config.crop_size else None,
        }


def load_and_preprocess_image(image_path: str, config: Optional[StandardPreprocessConfig] = None) -> np.ndarray:
    """Convenience helper for standard preprocessing."""
    preprocessor = ImagePreprocessor(config)
    return preprocessor.process_file(image_path)
