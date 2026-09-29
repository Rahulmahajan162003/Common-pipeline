# Benchmark Results Directory

Every benchmark execution automatically generates an isolated, versioned directory to ensure reproducibility:

```
results/
    <dataset_version>/
        run_<YYYY-MM-DD_HHMMSS>_<run_hash>/
            config.yaml              # Exact frozen configuration used
            metadata.json            # Machine, environment, git commit, and run parameters
            predictions.csv          # Per-sample predictions, scores, latencies, and metadata
            raw_scores.csv           # Preserved raw scores & execution statuses
            threshold_analysis.csv   # Comprehensive threshold sweep evaluation table
            metrics.json             # Structured JSON metrics (Classification, Biometric, PAD, Latency)
            latency.csv              # Granular per-sample latency breakdown
            report.md                # Human-readable markdown evaluation report
            report.html              # Standalone interactive HTML dashboard
            fusion_scores.csv        # Multi-signal aligned scores ready for fusion (if multi-model)
            plots/
                roc.png                  # Receiver Operating Characteristic curve
                det.png                  # Detection Error Tradeoff curve
                score_distribution.png   # Bona fide vs Attack score distribution
                latency_distribution.png # Latency histogram & tail percentiles
```

## Reproducing a Run
To inspect or compare previous runs, use the CLI:

```bash
python -m benchmark_harness compare --runs results/single_finger_test_v1/runA results/single_finger_test_v1/runB
```
