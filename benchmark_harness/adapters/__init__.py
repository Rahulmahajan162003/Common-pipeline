"""
Adapter registry and factory for the Benchmark Harness.
"""

from typing import Dict, Type, Any, Optional

from .base import BaseModelAdapter
from .score_csv import ScoreCSVAdapter
from .dualscale_2d import DualScale2DAdapter
from .model_3d import Model3DAdapter
from .flash_liveness import FlashLivenessAdapter
from ..schemas import ScoreDirection

ADAPTER_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {
    "score_csv": ScoreCSVAdapter,
    "csv": ScoreCSVAdapter,
    "dualscale_2d": DualScale2DAdapter,
    "dualscale": DualScale2DAdapter,
    "dual_scale_patchnet": DualScale2DAdapter,
    "model_3d": Model3DAdapter,
    "3d": Model3DAdapter,
    "flash_liveness": FlashLivenessAdapter,
    "flash": FlashLivenessAdapter,
}


def register_adapter(name: str, adapter_cls: Type[BaseModelAdapter]) -> None:
    """Registers a new model adapter with the harness."""
    ADAPTER_REGISTRY[name.lower()] = adapter_cls


def get_adapter(
    adapter_name: str,
    name: Optional[str] = None,
    version: str = "1.0.0",
    device: str = "auto",
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
    score_csv_path: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> BaseModelAdapter:
    """Factory to instantiate configured model adapters."""
    key = adapter_name.lower().strip()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter '{adapter_name}'. Registered adapters: {list(ADAPTER_REGISTRY.keys())}"
        )

    cls = ADAPTER_REGISTRY[key]
    if issubclass(cls, ScoreCSVAdapter):
        return cls(
            name=name or adapter_name,
            version=version,
            score_csv_path=score_csv_path,
            score_direction=score_direction,
            extra_params=extra_params,
        )

    return cls(
        name=name or adapter_name,
        version=version,
        device=device,
        score_direction=score_direction,
        extra_params=extra_params,
    )
