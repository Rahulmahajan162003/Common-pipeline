"""
Base adapter interface and registry for the Benchmark Harness.
All model implementations must conform to this contract.
"""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, List, Optional
import numpy as np

from ..schemas import (
    ManifestSample,
    ModelInferenceResult,
    ScoreDirection,
    SampleStatus,
    PredictionOutcome,
    TimingRecord,
)


class BaseModelAdapter(ABC):
    """Abstract base class for all model adapters."""

    def __init__(
        self,
        name: str,
        version: str = "1.0.0",
        device: str = "auto",
        score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
        extra_params: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.version = version
        self.device = device
        self.score_direction = score_direction
        self.extra_params = extra_params or {}
        self._is_loaded = False
        self._warmup_done = False
        self.preprocessing_config: Dict[str, Any] = {}

    @abstractmethod
    def load(self, checkpoint_path: Optional[str] = None) -> None:
        """Loads model weights, checkpoints, or resources into memory."""
        pass

    def warmup(self, warmup_runs: int = 3) -> None:
        """Runs warmup iterations to ensure steady-state latency profiling."""
        self._warmup_done = True

    def preprocess(self, sample: ManifestSample) -> Any:
        """Preprocesses a single sample into model input format."""
        return sample.image_path

    @abstractmethod
    def predict(self, sample: ManifestSample) -> ModelInferenceResult:
        """Executes inference on a single sample and returns a standardized ModelInferenceResult."""
        pass

    def predict_batch(self, samples: List[ManifestSample]) -> List[ModelInferenceResult]:
        """Default batch implementation; adapters can override for vectorization."""
        return [self.predict(s) for s in samples]

    def close(self) -> None:
        """Releases GPU or memory resources."""
        pass
