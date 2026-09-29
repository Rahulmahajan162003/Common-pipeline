"""
Data schemas, dataclasses, and enums for the YellowSense Fingerprint Benchmark Harness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ScoreDirection(str, Enum):
    HIGHER_IS_ACCEPT = "higher_is_accept"
    LOWER_IS_ACCEPT = "lower_is_accept"


class SampleStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


class PredictionOutcome(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    ERROR = "error"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    TEST = "test"


class GroundTruthLabel(str, Enum):
    LIVE = "live"
    BONAFIDE = "bonafide"
    SPOOF = "spoof"
    ATTACK = "attack"
    GENUINE = "genuine"
    IMPOSTOR = "impostor"

    @classmethod
    def is_positive(cls, label: Union[str, "GroundTruthLabel"]) -> bool:
        """Returns True if the label represents live/bonafide/genuine."""
        normalized = str(label).strip().lower()
        if normalized in ("live", "bonafide", "genuine", "1", "true"):
            return True
        elif normalized in ("spoof", "attack", "impostor", "0", "false", "fake"):
            return False
        raise ValueError(f"Unknown ground truth label: '{label}'. Must be live/bonafide/genuine or spoof/attack/impostor.")

    @classmethod
    def normalize_label(cls, label: Union[str, "GroundTruthLabel"]) -> str:
        """Normalizes label to 'live' or 'spoof' for consistent internal processing."""
        return "live" if cls.is_positive(label) else "spoof"


class TaskType(str, Enum):
    SINGLE_FINGER_AUTHENTICATION = "single_finger_authentication"
    SPOOF_DETECTION = "spoof_detection"
    SLAP_MATCHING = "slap_matching"
    GENERIC_BINARY_CLASSIFICATION = "generic_binary_classification"


class ThresholdMethod(str, Enum):
    MANUAL = "manual"
    EER = "eer"
    TARGET_FAR = "target_far"
    TARGET_FRR = "target_frr"
    MIN_BALANCED_ERROR = "min_balanced_error"


@dataclass
class ManifestSample:
    """Represents a single record in the benchmark manifest."""
    sample_id: str
    ground_truth: str
    image_path: str
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    finger_id: Optional[str] = None
    scenario: Optional[str] = None
    split: str = "test"
    device: Optional[str] = None
    lighting: Optional[str] = None
    capture_mode: Optional[str] = None
    spoof_type: Optional[str] = None
    material: Optional[str] = None
    dataset_version: Optional[str] = None
    source_dataset: Optional[str] = None
    notes: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def is_positive_ground_truth(self) -> bool:
        return GroundTruthLabel.is_positive(self.ground_truth)


@dataclass
class TimingRecord:
    """Latency timings for an inference sample in milliseconds."""
    total_latency_ms: float
    preprocess_latency_ms: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    postprocess_latency_ms: Optional[float] = None
    network_latency_ms: Optional[float] = None
    server_latency_ms: Optional[float] = None


@dataclass
class ModelInferenceResult:
    """Standardized output produced by a ModelAdapter."""
    sample_id: str
    model_name: str
    model_version: str
    raw_score: Optional[float]
    score_direction: ScoreDirection
    status: SampleStatus = SampleStatus.PASS
    prediction: PredictionOutcome = PredictionOutcome.REJECT
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    timing: TimingRecord = field(default_factory=lambda: TimingRecord(total_latency_ms=0.0))
    device: Optional[str] = None
    extra_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSampleRecord:
    """Final unified sample record after thresholding and metric evaluation."""
    sample_id: str
    ground_truth: str
    normalized_ground_truth: str
    model_name: str
    model_version: str
    raw_score: Optional[float]
    score_direction: str
    threshold: float
    prediction: str  # accept / reject / error
    status: str      # PASS / FAIL / ERROR / TIMEOUT / SKIPPED
    is_correct: Optional[bool]
    is_false_accept: bool
    is_false_reject: bool
    total_latency_ms: float
    preprocess_latency_ms: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    postprocess_latency_ms: Optional[float] = None
    network_latency_ms: Optional[float] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    subject_id: Optional[str] = None
    scenario: Optional[str] = None
    device: Optional[str] = None
    spoof_type: Optional[str] = None
    material: Optional[str] = None
    split: str = "test"


@dataclass
class ValidationError:
    """Represents a dataset or configuration validation error."""
    category: str
    message: str
    sample_id: Optional[str] = None
    row_index: Optional[int] = None
    details: Optional[Dict[str, Any]] = None
