# Pipeline benchmark summary

Median over repeats. Speedup is vs `baseline` on the same dataset / workers / batch / device.

**Read the spread column before trusting a single-digit-percent difference.** It's `(max - min) / median` over the repeats in that cell — a cell with 2 repeats and a wide spread cannot distinguish a real effect from run-to-run noise.

Known limitations of this harness (not fixed by more repeats):
- **n=3 repeats minimum** in this run.
- **Warm-up is not excluded.** The first epoch of each cell (cuDNN autotune, CUDA context init, allocator/page-lock warmup) is timed like any other; on short runs this can look like a per-batch regression that a longer run would amortize away.
- **Single point in batch-size space**: only batch_size=8 tested. `pin_memory`'s benefit scales with transfer size — a variant that loses here might win at a batch size this sweep never tried.
- **6 epochs per cell** — mechanisms that amortize a one-time cost across epochs (e.g. `persistent_workers` avoiding worker respawn) get a shorter horizon to pay off than a real training run would give them.
- See `results/equivalence.md` for whether variants in the same dataset actually trained on the same data, `results/correctness.md` for whether they trained to the same best_val_metric, and `results/drift.md` for whether run order correlates with the measured speed.

| dataset | workers | variant | n | atoms/s | spread | vs baseline | loader ms | step ms | unpack ms | h2d ms | serialize ms | peak GB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| carbon | 0 | baseline | 3 | 779 | 6% | 1.00 | 11.49 | 29.6 | 1.26 | 2.01 | 2.03 | — |
| carbon | 0 | everything | 3 | 761 | 4% | 0.98 | 10.49 | 31.6 | 1.23 | 2.00 | 0.41 | 1.76 |
| carbon | 0 | persistent | 3 | 742 | 4% | 0.95 | 11.86 | 31.3 | 1.28 | 2.01 | 2.07 | 1.76 |
| carbon | 0 | pinned | 3 | 687 | 2% | 0.88 | 12.19 | 33.9 | 1.29 | 2.04 | 2.06 | 1.76 |
| carbon | 0 | timed | 3 | 725 | 11% | 0.93 | 12.17 | 32.0 | 1.26 | 2.01 | 2.06 | 1.76 |
| carbon | 0 | transport | 3 | 815 | 8% | 1.05 | 10.63 | 29.2 | 1.23 | 1.99 | 0.41 | 1.76 |
| qm9 | 0 | baseline | 3 | 1843 | 6% | 1.00 | 13.02 | 27.7 | 1.08 | 2.10 | 1.88 | — |
| qm9 | 0 | everything | 3 | 1751 | 4% | 0.95 | 10.57 | 32.6 | 1.18 | 2.07 | 0.45 | 1.75 |
| qm9 | 0 | persistent | 3 | 1690 | 10% | 0.92 | 11.82 | 32.6 | 1.10 | 2.11 | 1.99 | 1.75 |
| qm9 | 0 | pinned | 3 | 1674 | 6% | 0.91 | 11.91 | 33.0 | 1.10 | 2.08 | 1.95 | 1.75 |
| qm9 | 0 | timed | 3 | 1753 | 3% | 0.95 | 11.98 | 31.0 | 1.12 | 2.10 | 1.97 | 1.75 |
| qm9 | 0 | transport | 3 | 1793 | 11% | 0.97 | 10.14 | 32.0 | 1.20 | 2.05 | 0.41 | 1.75 |
| si216 | 0 | baseline | 3 | 16614 | 1% | 1.00 | 26.42 | 77.6 | 7.83 | 2.20 | 9.39 | — |
| si216 | 0 | everything | 3 | 18757 | 2% | 1.13 | 18.10 | 74.3 | 2.21 | 2.14 | 0.69 | 1.75 |
| si216 | 0 | persistent | 3 | 16020 | 1% | 0.96 | 28.01 | 79.9 | 7.86 | 2.19 | 10.13 | 1.75 |
| si216 | 0 | pinned | 3 | 15264 | 1% | 0.92 | 31.62 | 81.6 | 9.29 | 2.24 | 12.64 | 1.74 |
| si216 | 0 | timed | 3 | 16021 | 1% | 0.96 | 27.92 | 79.5 | 7.91 | 2.19 | 9.82 | 1.75 |
| si216 | 0 | transport | 3 | 18568 | 1% | 1.12 | 17.67 | 75.3 | 2.23 | 2.16 | 0.59 | 1.76 |
| si_large | 0 | baseline | 3 | 21662 | 0% | 1.00 | 73.37 | 295.6 | 33.20 | 2.75 | 36.39 | — |
| si_large | 0 | everything | 3 | 25355 | 1% | 1.17 | 47.06 | 268.5 | 7.68 | 2.48 | 3.30 | 1.80 |
| si_large | 0 | persistent | 3 | 21584 | 1% | 1.00 | 74.92 | 295.9 | 33.06 | 2.73 | 37.22 | 1.82 |
| si_large | 0 | pinned | 3 | 20726 | 0% | 0.96 | 86.62 | 299.4 | 37.28 | 2.72 | 44.00 | 1.82 |
| si_large | 0 | timed | 3 | 21604 | 0% | 1.00 | 75.06 | 295.6 | 33.32 | 2.72 | 37.16 | 1.82 |
| si_large | 0 | transport | 3 | 26093 | 0% | 1.20 | 38.63 | 267.8 | 6.31 | 2.70 | 1.92 | 1.79 |
