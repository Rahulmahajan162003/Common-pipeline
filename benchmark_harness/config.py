"""
Configuration parsing and validation for the Benchmark Harness.
"""

import os
import yaml
import json
import hashlib
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .schemas import ScoreDirection, ThresholdMethod, TaskType


@dataclass
class BenchmarkMetaConfig:
    name: str = "single_finger_benchmark"
    dataset_version: str = "v1"
    task_type: TaskType = TaskType.SINGLE_FINGER_AUTHENTICATION
    random_seed: int = 42
    description: str = "YellowSense Single-Finger Benchmark Run"


@dataclass
class DataConfig:
    manifest_path: str = ""
    calibration_manifest_path: Optional[str] = None
    calibration_split_name: str = "calibration"
    test_split_name: str = "test"
    check_image_existence: bool = True
    verify_hashes: bool = True
    allow_missing_images: bool = False
    base_image_dir: Optional[str] = None


@dataclass
class EvaluationConfig:
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT
    threshold_method: ThresholdMethod = ThresholdMethod.TARGET_FAR
    manual_threshold: Optional[float] = None
    target_far: float = 0.001  # 0.1% FAR
    target_frr: Optional[float] = None
    treat_error_as_reject: bool = False
    threshold_sweep_steps: int = 100


@dataclass
class LatencyConfig:
    warmup_runs: int = 3
    repetitions: int = 1
    thresholds_ms: List[float] = field(default_factory=lambda: [1000.0, 2000.0, 5000.0, 7000.0])
    track_cuda_events: bool = True


@dataclass
class ModelConfig:
    name: str = "dualscale_2d"
    version: str = "1.0.0"
    adapter: str = "dualscale_2d"
    checkpoint_path: Optional[str] = None
    score_csv_path: Optional[str] = None
    device: str = "auto"
    batch_size: int = 1
    score_direction: Optional[ScoreDirection] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputsConfig:
    root: str = "results"
    export_plots: bool = True
    generate_html: bool = True
    export_fusion_table: bool = True
    top_k_hard_cases: int = 20


@dataclass
class BenchmarkConfig:
    benchmark: BenchmarkMetaConfig = field(default_factory=BenchmarkMetaConfig)
    data: DataConfig = field(default_factory=DataConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    models: List[ModelConfig] = field(default_factory=list)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)
    raw_dict: Dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        """Computes deterministic SHA-256 hash of configuration."""
        raw_repr = json.dumps(self.raw_dict, sort_keys=True, default=str)
        return hashlib.sha256(raw_repr.encode("utf-8")).hexdigest()[:16]


def load_config(config_path: str) -> BenchmarkConfig:
    """Loads and validates a YAML or JSON benchmark configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            raw = yaml.safe_load(f) or {}
        elif config_path.endswith(".json"):
            raw = json.load(f) or {}
        else:
            raise ValueError(f"Unsupported config format for '{config_path}'. Must be YAML or JSON.")

    return parse_config_dict(raw)


def parse_config_dict(raw: Dict[str, Any]) -> BenchmarkConfig:
    """Parses dictionary into BenchmarkConfig dataclass with strict validation."""
    bm_raw = raw.get("benchmark", {})
    data_raw = raw.get("data", {})
    eval_raw = raw.get("evaluation", {})
    lat_raw = raw.get("latency", {})
    models_raw = raw.get("models", [])
    out_raw = raw.get("outputs", {})

    # Validate score direction
    score_dir_str = eval_raw.get("score_direction")
    if not score_dir_str:
        raise ValueError(
            "Evaluation config MUST explicitly define 'score_direction'. "
            "Options: 'higher_is_accept' or 'lower_is_accept'."
        )
    try:
        score_direction = ScoreDirection(score_dir_str)
    except ValueError:
        raise ValueError(
            f"Invalid score_direction '{score_dir_str}'. Must be 'higher_is_accept' or 'lower_is_accept'."
        )

    # Validate threshold method
    thresh_method_str = eval_raw.get("threshold_method", "target_far")
    try:
        threshold_method = ThresholdMethod(thresh_method_str)
    except ValueError:
        raise ValueError(f"Invalid threshold_method '{thresh_method_str}'.")

    # Validate task type
    task_type_str = bm_raw.get("task_type", "single_finger_authentication")
    try:
        task_type = TaskType(task_type_str)
    except ValueError:
        task_type = TaskType.SINGLE_FINGER_AUTHENTICATION

    benchmark_meta = BenchmarkMetaConfig(
        name=bm_raw.get("name", "single_finger_benchmark"),
        dataset_version=bm_raw.get("dataset_version", "v1"),
        task_type=task_type,
        random_seed=int(bm_raw.get("random_seed", 42)),
        description=bm_raw.get("description", "YellowSense Single-Finger Benchmark Run"),
    )

    data_cfg = DataConfig(
        manifest_path=data_raw.get("manifest_path", data_raw.get("manifest", "")),
        calibration_manifest_path=data_raw.get("calibration_manifest_path"),
        calibration_split_name=data_raw.get("calibration_split_name", "calibration"),
        test_split_name=data_raw.get("test_split_name", "test"),
        check_image_existence=data_raw.get("check_image_existence", True),
        verify_hashes=data_raw.get("verify_hashes", True),
        allow_missing_images=data_raw.get("allow_missing_images", False),
        base_image_dir=data_raw.get("base_image_dir"),
    )

    eval_cfg = EvaluationConfig(
        score_direction=score_direction,
        threshold_method=threshold_method,
        manual_threshold=eval_raw.get("manual_threshold", eval_raw.get("threshold")),
        target_far=float(eval_raw.get("target_far", 0.001)),
        target_frr=float(eval_raw["target_frr"]) if "target_frr" in eval_raw else None,
        treat_error_as_reject=bool(eval_raw.get("treat_error_as_reject", False)),
        threshold_sweep_steps=int(eval_raw.get("threshold_sweep_steps", 100)),
    )

    lat_cfg = LatencyConfig(
        warmup_runs=int(lat_raw.get("warmup_runs", 3)),
        repetitions=int(lat_raw.get("repetitions", 1)),
        thresholds_ms=[float(x) for x in lat_raw.get("thresholds_ms", [1000.0, 2000.0, 5000.0, 7000.0])],
        track_cuda_events=bool(lat_raw.get("track_cuda_events", True)),
    )

    parsed_models: List[ModelConfig] = []
    if isinstance(models_raw, list):
        for m in models_raw:
            m_sd_str = m.get("score_direction")
            m_score_direction = None
            if m_sd_str:
                try:
                    m_score_direction = ScoreDirection(str(m_sd_str).lower().strip())
                except ValueError:
                    raise ValueError(f"Invalid model score_direction '{m_sd_str}' for model '{m.get('name')}'. Must be 'higher_is_accept', 'lower_is_accept', or 'higher_is_reject'.")

            parsed_models.append(
                ModelConfig(
                    name=m.get("name", "unknown_model"),
                    version=m.get("version", "1.0.0"),
                    adapter=m.get("adapter", "score_csv"),
                    checkpoint_path=m.get("checkpoint_path", m.get("checkpoint")),
                    score_csv_path=m.get("score_csv_path", m.get("score_csv")),
                    device=m.get("device", "auto"),
                    batch_size=int(m.get("batch_size", 1)),
                    score_direction=m_score_direction,
                    extra_params=m.get("params", m.get("extra_params", {})),
                )
            )

    out_cfg = OutputsConfig(
        root=out_raw.get("root", "results"),
        export_plots=bool(out_raw.get("export_plots", True)),
        generate_html=bool(out_raw.get("generate_html", True)),
        export_fusion_table=bool(out_raw.get("export_fusion_table", True)),
        top_k_hard_cases=int(out_raw.get("top_k_hard_cases", 20)),
    )

    return BenchmarkConfig(
        benchmark=benchmark_meta,
        data=data_cfg,
        evaluation=eval_cfg,
        latency=lat_cfg,
        models=parsed_models,
        outputs=out_cfg,
        raw_dict=raw,
    )
