"""
Plot generation for ROC, DET, Score Distributions, and Latency Profiles.
"""

import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering
import matplotlib.pyplot as plt

from ..schemas import ScoreDirection, GroundTruthLabel


def plot_roc_curve(
    scores: List[float],
    ground_truths: List[str],
    output_path: str,
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
    eer: Optional[float] = None,
    target_far: Optional[float] = None,
) -> None:
    """Plots standard Receiver Operating Characteristic (ROC) curve."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    labels = np.array([GroundTruthLabel.is_positive(gt) for gt in ground_truths], dtype=bool)
    score_arr = np.array(scores, dtype=np.float64)

    pos_scores = score_arr[labels]
    neg_scores = score_arr[~labels]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return

    thresholds = np.linspace(np.min(score_arr) - 0.01, np.max(score_arr) + 0.01, 500)
    far_list = []
    tar_list = []

    for th in thresholds:
        if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
            far = np.mean(neg_scores >= th)
            tar = np.mean(pos_scores >= th)
        else:
            far = np.mean(neg_scores <= th)
            tar = np.mean(pos_scores <= th)
        far_list.append(far)
        tar_list.append(tar)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.plot(far_list, tar_list, color="#1a73e8", lw=2.2, label="ROC Curve")
    ax.plot([0, 1], [0, 1], color="#9aa0a6", linestyle="--", lw=1.2, label="Random Chance")

    if eer is not None:
        ax.scatter([eer], [1.0 - eer], color="#ea4335", s=60, zorder=5, label=f"EER = {eer*100:.2f}%")

    if target_far is not None and target_far > 0:
        ax.axvline(x=target_far, color="#34a853", linestyle=":", lw=1.5, label=f"Target FAR ({target_far*100:.3f}%)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Accept Rate (FAR / APCER)", fontsize=11, fontweight="medium")
    ax.set_ylabel("True Accept Rate (TAR / GAR)", fontsize=11, fontweight="medium")
    ax.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_det_curve(
    scores: List[float],
    ground_truths: List[str],
    output_path: str,
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
) -> None:
    """Plots Detection Error Tradeoff (DET) curve (FAR vs FRR)."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    labels = np.array([GroundTruthLabel.is_positive(gt) for gt in ground_truths], dtype=bool)
    score_arr = np.array(scores, dtype=np.float64)

    pos_scores = score_arr[labels]
    neg_scores = score_arr[~labels]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return

    thresholds = np.linspace(np.min(score_arr) - 0.01, np.max(score_arr) + 0.01, 500)
    far_list = []
    frr_list = []

    for th in thresholds:
        if score_direction == ScoreDirection.HIGHER_IS_ACCEPT:
            far = np.mean(neg_scores >= th)
            frr = np.mean(pos_scores < th)
        else:
            far = np.mean(neg_scores <= th)
            frr = np.mean(pos_scores > th)
        far_list.append(max(far, 1e-5))
        frr_list.append(max(frr, 1e-5))

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.plot(far_list, frr_list, color="#d93025", lw=2.2, label="DET Curve (FAR vs FRR)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("False Accept Rate (FAR / APCER) [Log Scale]", fontsize=11, fontweight="medium")
    ax.set_ylabel("False Reject Rate (FRR / BPCER) [Log Scale]", fontsize=11, fontweight="medium")
    ax.set_title("Detection Error Tradeoff (DET)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_score_distributions(
    scores: List[float],
    ground_truths: List[str],
    output_path: str,
    threshold: Optional[float] = None,
) -> None:
    """Plots Genuine vs Impostor/Spoof score probability distributions."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    labels = np.array([GroundTruthLabel.is_positive(gt) for gt in ground_truths], dtype=bool)
    score_arr = np.array(scores, dtype=np.float64)

    pos_scores = score_arr[labels]
    neg_scores = score_arr[~labels]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    bins = np.linspace(
        min(np.min(score_arr), 0.0),
        max(np.max(score_arr), 1.0),
        40
    )

    if len(pos_scores) > 0:
        ax.hist(
            pos_scores,
            bins=bins,
            alpha=0.6,
            color="#1e8e3e",
            density=True,
            edgecolor="#137333",
            label=f"Bona Fide / Genuine (N={len(pos_scores)})",
        )

    if len(neg_scores) > 0:
        ax.hist(
            neg_scores,
            bins=bins,
            alpha=0.6,
            color="#d93025",
            density=True,
            edgecolor="#a50e0e",
            label=f"Attack / Spoof (N={len(neg_scores)})",
        )

    if threshold is not None:
        ax.axvline(
            x=threshold,
            color="#202124",
            linestyle="--",
            lw=2.0,
            label=f"Decision Threshold ({threshold:.4f})",
        )

    ax.set_xlabel("Raw Model Score", fontsize=11, fontweight="medium")
    ax.set_ylabel("Density", fontsize=11, fontweight="medium")
    ax.set_title("Genuine vs Spoof Score Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_latency_distribution(
    latencies_ms: List[float],
    output_path: str,
    p50_ms: Optional[float] = None,
    p95_ms: Optional[float] = None,
    p99_ms: Optional[float] = None,
) -> None:
    """Plots inference latency distribution and highlight percentile markers."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if not latencies_ms:
        return

    arr = np.array(latencies_ms, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

    ax.hist(arr, bins=35, color="#1a73e8", alpha=0.7, edgecolor="#174ea6", density=False)

    if p50_ms is not None:
        ax.axvline(x=p50_ms, color="#34a853", linestyle="--", lw=1.8, label=f"P50 (Median): {p50_ms:.1f}ms")
    if p95_ms is not None:
        ax.axvline(x=p95_ms, color="#f9ab00", linestyle="--", lw=1.8, label=f"P95: {p95_ms:.1f}ms")
    if p99_ms is not None:
        ax.axvline(x=p99_ms, color="#ea4335", linestyle="--", lw=2.0, label=f"P99: {p99_ms:.1f}ms")

    # 7-second marker callout
    if np.max(arr) >= 5000:
        ax.axvline(x=7000.0, color="#9334e6", linestyle=":", lw=2.0, label="7000ms SLA Target")

    ax.set_xlabel("Latency (ms)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Sample Count", fontsize=11, fontweight="medium")
    ax.set_title("End-to-End Inference Latency Distribution", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_plots(
    scores: List[float],
    ground_truths: List[str],
    latencies_ms: List[float],
    output_dir: str,
    score_direction: ScoreDirection = ScoreDirection.HIGHER_IS_ACCEPT,
    threshold: Optional[float] = None,
    eer: Optional[float] = None,
    target_far: Optional[float] = None,
    p50_ms: Optional[float] = None,
    p95_ms: Optional[float] = None,
    p99_ms: Optional[float] = None,
) -> Dict[str, str]:
    """Generates all standard benchmark diagnostic plots and returns their filepaths."""
    os.makedirs(output_dir, exist_ok=True)
    saved_plots: Dict[str, str] = {}

    roc_path = os.path.join(output_dir, "roc.png")
    plot_roc_curve(scores, ground_truths, roc_path, score_direction, eer, target_far)
    saved_plots["roc"] = roc_path

    det_path = os.path.join(output_dir, "det.png")
    plot_det_curve(scores, ground_truths, det_path, score_direction)
    saved_plots["det"] = det_path

    dist_path = os.path.join(output_dir, "score_distribution.png")
    plot_score_distributions(scores, ground_truths, dist_path, threshold)
    saved_plots["score_distribution"] = dist_path

    lat_path = os.path.join(output_dir, "latency_distribution.png")
    plot_latency_distribution(latencies_ms, lat_path, p50_ms, p95_ms, p99_ms)
    saved_plots["latency_distribution"] = lat_path

    return saved_plots
