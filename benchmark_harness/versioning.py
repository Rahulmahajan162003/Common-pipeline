"""
Run metadata extraction and environment versioning for the Benchmark Harness.
"""

import os
import sys
import platform
import subprocess
import datetime
import socket
from typing import Dict, Any, Optional

from .schemas import ScoreDirection
from .config import BenchmarkConfig
from .manifest import BenchmarkManifest


def get_git_metadata(repo_dir: Optional[str] = None) -> Dict[str, Any]:
    """Extracts git commit, branch, and dirty status if in a git repository."""
    git_info = {
        "git_commit": "unknown",
        "git_branch": "unknown",
        "git_dirty": False,
        "is_git_repo": False,
    }
    cwd = repo_dir or os.getcwd()
    try:
        # Check if git repo
        is_repo = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False
        )
        if is_repo.returncode == 0 and is_repo.stdout.strip() == "true":
            git_info["is_git_repo"] = True

            # Commit hash
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False
            )
            if commit.returncode == 0:
                git_info["git_commit"] = commit.stdout.strip()

            # Branch
            branch = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False
            )
            if branch.returncode == 0:
                git_info["git_branch"] = branch.stdout.strip()

            # Dirty state
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False
            )
            if status.returncode == 0:
                git_info["git_dirty"] = len(status.stdout.strip()) > 0
    except Exception:
        pass

    return git_info


def get_hardware_environment() -> Dict[str, Any]:
    """Gathers CPU, memory, OS, and GPU environment specs."""
    env = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_compiler": platform.python_compiler(),
        "python_executable": sys.executable,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
    }

    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device_count"] = torch.cuda.device_count()
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
            env["cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    # Package versions
    pkg_versions = {}
    for pkg in ["torch", "torchvision", "numpy", "pandas", "scipy", "sklearn", "PIL", "cv2", "yaml"]:
        try:
            mod = __import__(pkg)
            pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkg_versions[pkg] = "not_installed"
    env["package_versions"] = pkg_versions

    return env


def build_run_metadata(
    config: BenchmarkConfig,
    manifest: Optional[BenchmarkManifest] = None,
    calibration_manifest: Optional[BenchmarkManifest] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Assembles a full reproducibility metadata dictionary."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_meta = get_git_metadata()
    hw_meta = get_hardware_environment()

    metadata = {
        "benchmark_name": config.benchmark.name,
        "benchmark_version": "1.0.0",
        "dataset_version": config.benchmark.dataset_version,
        "task_type": config.benchmark.task_type.value,
        "timestamp_utc": timestamp,
        "random_seed": config.benchmark.random_seed,
        "config_hash": config.config_hash,
        "git": git_meta,
        "environment": hw_meta,
        "evaluation_settings": {
            "score_direction": config.evaluation.score_direction.value,
            "threshold_method": config.evaluation.threshold_method.value,
            "target_far": config.evaluation.target_far,
            "target_frr": config.evaluation.target_frr,
            "manual_threshold": config.evaluation.manual_threshold,
            "treat_error_as_reject": config.evaluation.treat_error_as_reject,
        },
        "latency_settings": {
            "warmup_runs": config.latency.warmup_runs,
            "repetitions": config.latency.repetitions,
            "thresholds_ms": config.latency.thresholds_ms,
        },
    }

    if model_name:
        metadata["model"] = {
            "name": model_name,
            "version": model_version or "unknown",
        }

    if manifest:
        metadata["test_dataset_summary"] = manifest.get_summary_stats()

    if calibration_manifest:
        metadata["calibration_dataset_summary"] = calibration_manifest.get_summary_stats()

    return metadata
