"""
Manifest loading, parsing, and management for the Benchmark Harness.
"""

import os
import csv
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict

from .schemas import ManifestSample, GroundTruthLabel, ValidationError


class BenchmarkManifest:
    """Container and manager for benchmark dataset samples."""

    def __init__(self, samples: List[ManifestSample], manifest_path: Optional[str] = None):
        self.samples = samples
        self.manifest_path = manifest_path

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, idx: int) -> ManifestSample:
        return self.samples[idx]

    @property
    def sample_ids(self) -> List[str]:
        return [s.sample_id for s in self.samples]

    @property
    def subjects(self) -> List[str]:
        return list(set(s.subject_id for s in self.samples if s.subject_id))

    def get_split(self, split_name: str) -> "BenchmarkManifest":
        """Returns a new BenchmarkManifest containing only samples in the given split."""
        filtered = [s for s in self.samples if s.split.lower() == split_name.lower()]
        return BenchmarkManifest(filtered, self.manifest_path)

    def get_live_samples(self) -> List[ManifestSample]:
        """Returns all genuine/live/bonafide samples."""
        return [s for s in self.samples if s.is_positive_ground_truth()]

    def get_spoof_samples(self) -> List[ManifestSample]:
        """Returns all spoof/attack/impostor samples."""
        return [s for s in self.samples if not s.is_positive_ground_truth()]

    def compute_manifest_hash(self) -> str:
        """Computes a deterministic SHA-256 hash of the manifest content."""
        hasher = hashlib.sha256()
        # Sort sample records by sample_id to ensure order independence
        sorted_samples = sorted(self.samples, key=lambda s: s.sample_id)
        for s in sorted_samples:
            record_str = f"{s.sample_id}|{s.ground_truth}|{s.image_path}|{s.subject_id}|{s.split}|{s.scenario}"
            hasher.update(record_str.encode("utf-8"))
        return hasher.hexdigest()

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates dataset summary metrics."""
        total = len(self.samples)
        live_count = len(self.get_live_samples())
        spoof_count = len(self.get_spoof_samples())
        subjects = set(s.subject_id for s in self.samples if s.subject_id)
        splits = set(s.split for s in self.samples)
        devices = set(s.device for s in self.samples if s.device)
        scenarios = set(s.scenario for s in self.samples if s.scenario)
        spoof_types = set(s.spoof_type for s in self.samples if s.spoof_type)

        return {
            "total_samples": total,
            "live_samples": live_count,
            "spoof_samples": spoof_count,
            "live_ratio": (live_count / total) if total > 0 else 0.0,
            "spoof_ratio": (spoof_count / total) if total > 0 else 0.0,
            "unique_subjects": len(subjects),
            "splits": list(splits),
            "devices": list(devices),
            "scenarios": list(scenarios),
            "spoof_types": list(spoof_types),
            "manifest_hash": self.compute_manifest_hash() if total > 0 else "empty",
        }

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        base_image_dir: Optional[str] = None
    ) -> "BenchmarkManifest":
        """Loads a manifest from a CSV file."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Manifest CSV not found: {csv_path}")

        samples: List[ManifestSample] = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return cls([], csv_path)

            for row in reader:
                # Strip keys and values
                clean_row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
                sample_id = clean_row.get("sample_id") or clean_row.get("id") or clean_row.get("SampleID")
                gt = clean_row.get("ground_truth") or clean_row.get("label") or clean_row.get("GroundTruth")
                img_path = clean_row.get("image_path") or clean_row.get("path") or clean_row.get("ImagePath")

                if not sample_id or not gt:
                    continue  # Will be caught or flagged during strict validation

                # Resolve image path
                if not img_path:
                    resolved_path = ""
                elif os.path.isabs(img_path):
                    resolved_path = os.path.normpath(img_path)
                elif base_image_dir:
                    resolved_path = os.path.normpath(os.path.join(base_image_dir, img_path))
                elif os.path.exists(img_path):
                    resolved_path = os.path.normpath(img_path)
                else:
                    manifest_dir = os.path.dirname(os.path.abspath(csv_path))
                    alt_path = os.path.normpath(os.path.join(manifest_dir, img_path))
                    if os.path.exists(alt_path):
                        resolved_path = alt_path
                    else:
                        resolved_path = os.path.normpath(img_path)

                sample = ManifestSample(
                    sample_id=str(sample_id),
                    ground_truth=str(gt),
                    image_path=str(resolved_path),
                    subject_id=clean_row.get("subject_id") or clean_row.get("subject"),
                    session_id=clean_row.get("session_id") or clean_row.get("session"),
                    finger_id=clean_row.get("finger_id") or clean_row.get("finger"),
                    scenario=clean_row.get("scenario"),
                    split=clean_row.get("split", "test").lower(),
                    device=clean_row.get("device"),
                    lighting=clean_row.get("lighting"),
                    capture_mode=clean_row.get("capture_mode"),
                    spoof_type=clean_row.get("spoof_type"),
                    material=clean_row.get("material"),
                    dataset_version=clean_row.get("dataset_version"),
                    source_dataset=clean_row.get("source_dataset"),
                    notes=clean_row.get("notes"),
                )
                samples.append(sample)

        return cls(samples, csv_path)

    def to_csv(self, output_path: str) -> None:
        """Saves manifest to a CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if not self.samples:
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                f.write("sample_id,ground_truth,image_path,subject_id,split\n")
            return

        fieldnames = [
            "sample_id", "subject_id", "session_id", "finger_id", "scenario",
            "split", "ground_truth", "image_path", "device", "lighting",
            "capture_mode", "spoof_type", "material", "dataset_version",
            "source_dataset", "notes"
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in self.samples:
                writer.writerow(asdict(s))
