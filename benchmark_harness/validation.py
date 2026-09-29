"""
Validation and data-integrity verification engine for manifests, configs, and splits.
"""

import os
import hashlib
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter

from .schemas import ValidationError, GroundTruthLabel
from .manifest import BenchmarkManifest
from .config import BenchmarkConfig


def compute_file_sha256(filepath: str) -> Optional[str]:
    """Computes SHA-256 hash of a file's binary content."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BenchmarkValidator:
    """Validates manifests, configurations, and splits for scientific integrity."""

    @staticmethod
    def validate_manifest(
        manifest: BenchmarkManifest,
        check_images: bool = True,
        verify_hashes: bool = False,
        allow_missing_images: bool = False
    ) -> List[ValidationError]:
        """Performs full integrity check on a single manifest."""
        errors: List[ValidationError] = []

        if len(manifest.samples) == 0:
            errors.append(ValidationError(
                category="EMPTY_MANIFEST",
                message="The manifest contains 0 samples."
            ))
            return errors

        seen_sample_ids: Set[str] = set()
        duplicate_sample_ids: Set[str] = set()

        seen_image_hashes: Dict[str, str] = {}  # hash -> first sample_id

        for idx, sample in enumerate(manifest.samples):
            # 1. Unique Sample ID check
            if not sample.sample_id or not sample.sample_id.strip():
                errors.append(ValidationError(
                    category="EMPTY_SAMPLE_ID",
                    message=f"Sample ID is empty at row {idx+1}.",
                    row_index=idx
                ))
            elif sample.sample_id in seen_sample_ids:
                duplicate_sample_ids.add(sample.sample_id)
                errors.append(ValidationError(
                    category="DUPLICATE_SAMPLE_ID",
                    message=f"Duplicate sample_id '{sample.sample_id}' detected.",
                    sample_id=sample.sample_id,
                    row_index=idx
                ))
            else:
                seen_sample_ids.add(sample.sample_id)

            # 2. Ground Truth Label Validity
            try:
                GroundTruthLabel.is_positive(sample.ground_truth)
            except ValueError as e:
                errors.append(ValidationError(
                    category="INVALID_GROUND_TRUTH",
                    message=f"Invalid ground truth label '{sample.ground_truth}' for sample '{sample.sample_id}': {str(e)}",
                    sample_id=sample.sample_id,
                    row_index=idx
                ))

            # 3. Image file existence check
            if check_images and not allow_missing_images:
                if not sample.image_path:
                    errors.append(ValidationError(
                        category="MISSING_IMAGE_PATH",
                        message=f"Missing image_path for sample '{sample.sample_id}'.",
                        sample_id=sample.sample_id,
                        row_index=idx
                    ))
                elif not os.path.exists(sample.image_path):
                    errors.append(ValidationError(
                        category="IMAGE_FILE_NOT_FOUND",
                        message=f"Image file does not exist: '{sample.image_path}' for sample '{sample.sample_id}'.",
                        sample_id=sample.sample_id,
                        row_index=idx,
                        details={"path": sample.image_path}
                    ))
                elif verify_hashes:
                    # Check for duplicate image content within the manifest
                    img_hash = compute_file_sha256(sample.image_path)
                    if img_hash:
                        if img_hash in seen_image_hashes:
                            prev_id = seen_image_hashes[img_hash]
                            errors.append(ValidationError(
                                category="DUPLICATE_IMAGE_CONTENT",
                                message=f"Sample '{sample.sample_id}' has identical image content (hash: {img_hash[:10]}...) to sample '{prev_id}'.",
                                sample_id=sample.sample_id,
                                row_index=idx,
                                details={"duplicate_with": prev_id, "hash": img_hash}
                            ))
                        else:
                            seen_image_hashes[img_hash] = sample.sample_id

        # Summary check: class distribution
        has_invalid_gt = any(e.category == "INVALID_GROUND_TRUTH" for e in errors)
        if not has_invalid_gt and len(manifest.samples) > 0:
            live_count = sum(1 for s in manifest.samples if s.is_positive_ground_truth())
            spoof_count = len(manifest.samples) - live_count
            if live_count == 0:
                errors.append(ValidationError(
                    category="SINGLE_CLASS_DATASET",
                    message="The manifest contains ONLY spoof/negative samples (0 genuine samples)."
                ))
            elif spoof_count == 0:
                errors.append(ValidationError(
                    category="SINGLE_CLASS_DATASET",
                    message="The manifest contains ONLY live/genuine samples (0 spoof samples)."
                ))

        return errors

    @staticmethod
    def validate_split_leakage(
        calibration_manifest: BenchmarkManifest,
        test_manifest: BenchmarkManifest,
        verify_hashes: bool = True
    ) -> List[ValidationError]:
        """Checks for subject leakage, duplicate samples, and duplicate images between calibration and test sets."""
        errors: List[ValidationError] = []

        calib_sample_ids = set(calibration_manifest.sample_ids)
        test_sample_ids = set(test_manifest.sample_ids)

        # 1. Sample ID intersection
        shared_ids = calib_sample_ids.intersection(test_sample_ids)
        if shared_ids:
            for sid in shared_ids:
                errors.append(ValidationError(
                    category="SPLIT_SAMPLE_OVERLAP",
                    message=f"Sample '{sid}' appears in both calibration and test sets.",
                    sample_id=sid
                ))

        # 2. Subject ID leakage
        calib_subjects = set(s.subject_id for s in calibration_manifest.samples if s.subject_id)
        test_subjects = set(s.subject_id for s in test_manifest.samples if s.subject_id)
        shared_subjects = calib_subjects.intersection(test_subjects)
        if shared_subjects:
            for subj in shared_subjects:
                errors.append(ValidationError(
                    category="SUBJECT_LEAKAGE",
                    message=f"Subject '{subj}' appears in both calibration and test splits. "
                            f"Subject-level partitioning is violated.",
                    details={"subject_id": subj}
                ))

        # 3. Image path and content leakage
        calib_paths = set(os.path.abspath(s.image_path) for s in calibration_manifest.samples if s.image_path)
        test_paths = set(os.path.abspath(s.image_path) for s in test_manifest.samples if s.image_path)
        shared_paths = calib_paths.intersection(test_paths)
        if shared_paths:
            for p in shared_paths:
                errors.append(ValidationError(
                    category="SPLIT_IMAGE_PATH_OVERLAP",
                    message=f"Exact image path '{p}' appears in both calibration and test sets.",
                    details={"path": p}
                ))

        if verify_hashes:
            calib_hashes: Dict[str, str] = {}
            for s in calibration_manifest.samples:
                if s.image_path and os.path.exists(s.image_path):
                    h = compute_file_sha256(s.image_path)
                    if h:
                        calib_hashes[h] = s.sample_id

            for s in test_manifest.samples:
                if s.image_path and os.path.exists(s.image_path):
                    h = compute_file_sha256(s.image_path)
                    if h and h in calib_hashes:
                        errors.append(ValidationError(
                            category="SPLIT_IMAGE_HASH_LEAKAGE",
                            message=f"Test sample '{s.sample_id}' has identical image content hash ({h[:10]}...) "
                                    f"to calibration sample '{calib_hashes[h]}'.",
                            sample_id=s.sample_id,
                            details={"calibration_sample_id": calib_hashes[h], "hash": h}
                        ))

        return errors

    @staticmethod
    def validate_config(config: BenchmarkConfig) -> List[ValidationError]:
        """Validates configuration settings."""
        errors: List[ValidationError] = []

        if not config.data.manifest_path and not any(m.score_csv_path for m in config.models):
            errors.append(ValidationError(
                category="INVALID_CONFIG",
                message="Either 'data.manifest_path' or model 'score_csv_path' must be specified."
            ))

        if config.evaluation.target_far <= 0 or config.evaluation.target_far >= 1.0:
            errors.append(ValidationError(
                category="INVALID_CONFIG",
                message=f"target_far must be strictly between 0 and 1. Found: {config.evaluation.target_far}"
            ))

        if not config.models:
            errors.append(ValidationError(
                category="NO_MODELS_SPECIFIED",
                message="Configuration must specify at least one model in 'models' list."
            ))

        return errors


def format_validation_errors(errors: List[ValidationError]) -> str:
    """Formats list of ValidationErrors into a clean readable string."""
    if not errors:
        return "All validation checks passed successfully."

    lines = [f"Found {len(errors)} validation issue(s):"]
    for i, err in enumerate(errors, 1):
        sample_info = f" [Sample: {err.sample_id}]" if err.sample_id else ""
        row_info = f" [Row: {err.row_index+1}]" if err.row_index is not None else ""
        lines.append(f"  {i}. [{err.category}]{sample_info}{row_info}: {err.message}")
    return "\n".join(lines)
