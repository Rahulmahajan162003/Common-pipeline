"""
DualScalePatchNet Parity Verification Script:
Compares scores on 20 images (10 genuine live + 10 screen replay attacks)
between the original KT standalone evaluation script and the Benchmark Harness adapter.
Pass criterion: Max score discrepancy |Delta| < 0.001 across all 20 images.
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Ensure repo root and backend are in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKEND_DIR = os.path.join(REPO_ROOT, "backend")
for p in [REPO_ROOT, BACKEND_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_harness.adapters.dualscale_2d import DualScale2DAdapter
from benchmark_harness.manifest import BenchmarkManifest
from benchmark_harness.schemas import ScoreDirection
from backend.pad.models import DualScalePatchNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = os.path.join(REPO_ROOT, "backend", "pad", "weights", "dual_scale_patchnet_best.pth")
MANIFEST_PATH = os.path.join(REPO_ROOT, "datasets", "manifests", "pad_v0", "pad_v0_single_finger.csv")
MANIFEST_DIR = os.path.dirname(MANIFEST_PATH)
FINE_SIZE = 96
MACRO_SIZE = 160


def run_original_kt_scoring(model, img_path):
    """Exact standalone KT evaluation pipeline from train_dualscale_patchnet.py / SDK spec."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    img_6c = np.concatenate([img_rgb, img_ycrcb], axis=2)

    coords = [
        (h // 2, w // 2),
        (int(h * 0.25), int(w * 0.25)),
        (int(h * 0.25), int(w * 0.75)),
        (int(h * 0.75), int(w * 0.25)),
        (int(h * 0.75), int(w * 0.75)),
    ]
    fines, macros = [], []
    for cy, cx in coords:
        y1 = max(0, min(cy - FINE_SIZE // 2, h - FINE_SIZE))
        x1 = max(0, min(cx - FINE_SIZE // 2, w - FINE_SIZE))
        fine_patch = img_6c[y1:y1 + FINE_SIZE, x1:x1 + FINE_SIZE]
        if fine_patch.shape[:2] != (FINE_SIZE, FINE_SIZE):
            fine_patch = cv2.resize(fine_patch, (FINE_SIZE, FINE_SIZE))

        my1 = max(0, min(cy - MACRO_SIZE // 2, h - MACRO_SIZE))
        mx1 = max(0, min(cx - MACRO_SIZE // 2, w - MACRO_SIZE))
        macro_patch = img_6c[my1:my1 + MACRO_SIZE, mx1:mx1 + MACRO_SIZE]
        macro_patch = cv2.resize(macro_patch, (FINE_SIZE, FINE_SIZE))

        tf = (torch.from_numpy(fine_patch.transpose(2, 0, 1)).float() / 255.0 - 0.50) / 0.25
        tm = (torch.from_numpy(macro_patch.transpose(2, 0, 1)).float() / 255.0 - 0.50) / 0.25
        fines.append(tf)
        macros.append(tm)

    batch_fine = torch.stack(fines).to(DEVICE)
    batch_macro = torch.stack(macros).to(DEVICE)

    with torch.no_grad():
        logits = model(batch_fine, batch_macro)
        probs = F.softmax(logits, dim=1).cpu().numpy().mean(axis=0)

    p_live = float(probs[0])
    p_screen = float(probs[1])
    p_paper = float(probs[2])
    p_attack = float(p_screen + p_paper)
    live_score = 1.0 - p_attack
    return p_live, p_screen, p_paper, p_attack, live_score


def main():
    print(f"Loading reference model from {WEIGHTS_PATH}...")
    ref_model = DualScalePatchNet(num_classes=3, in_channels=6).to(DEVICE)
    ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
    ref_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    ref_model.eval()

    print(f"Initializing Harness DualScale2DAdapter...")
    adapter = DualScale2DAdapter(
        name="dualscale_2d",
        device=str(DEVICE),
        score_direction=ScoreDirection.HIGHER_IS_ACCEPT,
        extra_params={"patch_aggregation": "mean", "finger_crop": "none"},
    )
    adapter.load(WEIGHTS_PATH)
    adapter.warmup()

    # Load 10 live + 10 screen replay images
    manifest = BenchmarkManifest.from_csv(MANIFEST_PATH)
    live_samples = [s for s in manifest.samples if s.ground_truth == "live"][:10]
    spoof_samples = [s for s in manifest.samples if s.ground_truth == "spoof"][:10]
    selected_20 = live_samples + spoof_samples

    print(f"Selected {len(live_samples)} live samples and {len(spoof_samples)} screen replay samples.")

    rows = []
    max_diff = 0.0

    for idx, s in enumerate(selected_20, start=1):
        # 1. Original KT script
        ref_p_live, ref_p_scr, ref_p_pap, ref_p_atk, ref_live_score = run_original_kt_scoring(
            ref_model, s.image_path
        )

        # 2. Benchmark Harness adapter
        res = adapter.predict(s)
        tool_live_score = res.raw_score
        tool_p_live = res.extra_metadata["p_live"]
        tool_p_scr = res.extra_metadata["p_screen"]
        tool_p_pap = res.extra_metadata["p_paper"]
        tool_p_atk = res.extra_metadata["p_attack"]

        diff = abs(ref_live_score - tool_live_score)
        if diff > max_diff:
            max_diff = diff

        rows.append({
            "No.": idx,
            "Sample ID": s.sample_id,
            "Ground Truth": s.ground_truth,
            "KT Script Live Score": f"{ref_live_score:.6f}",
            "Tool Live Score": f"{tool_live_score:.6f}",
            "Diff (|Delta|)": f"{diff:.8f}",
            "P(Screen)": f"{tool_p_scr:.4f}",
            "P(Paper)": f"{tool_p_pap:.4f}",
            "P(Attack)": f"{tool_p_atk:.4f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("DUALSCALE PATCHNET 20-IMAGE BALANCED PARITY TABLE (10 Live + 10 Screen Replay)")
    print("=" * 96)
    print(df.to_string(index=False))
    print("=" * 96)
    print(f"Maximum Absolute Discrepancy: {max_diff:.10f}")
    passed = max_diff < 0.001
    print(f"Parity Test Status (|Delta| < 0.001): {'PASSED' if passed else 'FAILED'}")
    print("=" * 96)

    out_csv = os.path.join(REPO_ROOT, "scores", "dualscale_20_image_parity.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved updated 20-image parity results to: {out_csv}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
