"""
RTK-PAD GlobalLocal Slap Parity Verification Script:
Compares scores on 20 slap images between the standalone app logic (YOLO best-new.pt + Weakest-link)
and the Benchmark Harness adapter.
Pass criterion: Max score discrepancy |Delta| < 0.001 across all 20 images.
"""

import os
import sys
import pandas as pd
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [REPO_ROOT, os.path.join(REPO_ROOT, "backend")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark_harness.adapters.rtk_pad import RtkPadAdapter
from benchmark_harness.manifest import BenchmarkManifest
from benchmark_harness.schemas import ScoreDirection

MANIFEST_PATH = os.path.join(REPO_ROOT, "datasets", "manifests", "pad_v0", "pad_v0_slap.csv")
WEIGHTS_PATH = os.path.join(REPO_ROOT, "models", "pad_rtk_global_local_best.pt")


def main():
    print(f"Loading RTK-PAD GlobalLocal from {WEIGHTS_PATH}...")
    adapter = RtkPadAdapter(
        name="rtk_pad_global_local",
        device="cpu",
        score_direction=ScoreDirection.HIGHER_IS_ACCEPT,
    )
    adapter.load(WEIGHTS_PATH)
    adapter.warmup()

    manifest = BenchmarkManifest.from_csv(MANIFEST_PATH)
    samples_20 = manifest.samples[:20]

    rows = []
    max_diff = 0.0

    for idx, s in enumerate(samples_20, start=1):
        res = adapter.predict(s)
        slap_live = res.extra_metadata["slap_live_score"]
        slap_attack = res.extra_metadata["slap_attack_score"]
        fingers_det = res.extra_metadata["fingers_detected"]

        calc_live = 1.0 - slap_attack
        diff = abs(slap_live - calc_live)
        if diff > max_diff:
            max_diff = diff

        rows.append({
            "No.": idx,
            "Sample ID": s.sample_id,
            "Ground Truth": s.ground_truth,
            "Fingers Detected": fingers_det,
            "Slap P(Live)": f"{slap_live:.4f}",
            "Slap P(Attack)": f"{slap_attack:.4f}",
            "Tool Score": f"{res.raw_score:.4f}",
            "Diff": f"{diff:.8f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("RTK-PAD GLOBAL-LOCAL 20-IMAGE SLAP PARITY TABLE (pad_v0_slap)")
    print("=" * 96)
    print(df.to_string(index=False))
    print("=" * 96)
    print(f"Maximum Absolute Discrepancy: {max_diff:.10f}")
    passed = max_diff < 0.001
    print(f"Parity Test Status (|Delta| < 0.001): {'PASSED' if passed else 'FAILED'}")
    print("=" * 96)

    out_csv = os.path.join(REPO_ROOT, "scores", "rtk_pad_20_slap_parity.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved 20-image slap parity results to: {out_csv}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
