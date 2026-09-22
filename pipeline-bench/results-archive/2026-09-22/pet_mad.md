# PET-MAD fine-tuning benchmark

Fine-tunes a pretrained PET-MAD checkpoint (not a small model trained from scratch) on the same datasets `results/summary.md` uses, with the same per-stage timing. Tests whether the data-loading changes this harness benchmarks help or hurt at PET-MAD's much larger model size (102 atomic types, energy + non-conservative force + stress heads) and richer per-batch compute, compared to the small energy-only model everything else in this harness trains from scratch.

| dataset | workers | variant | status | atoms/s | vs baseline | best_val_metric | vs baseline |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| qm9 | 0 | baseline | ok | 1383 | ref | 3.9166 | ref |
| qm9 | 0 | everything | ok | 1419 | 1.03 | 3.9664 | 1% |
| qm9 | 0 | persistent | ok | 1360 | 0.98 | 3.9664 | 1% |
| qm9 | 0 | pinned | ok | 1334 | 0.96 | 3.9664 | 1% |
| qm9 | 0 | timed | ok | 1311 | 0.95 | 3.9664 | 1% |
| qm9 | 0 | transport | ok | 1434 | 1.04 | 3.9664 | 1% |
| qm9 | 1 | baseline | ok | 1789 | ref | 3.9423 | ref |
| qm9 | 1 | everything | ok | 1812 | 1.01 | 3.7578 | 5% |
| qm9 | 1 | persistent | ok | 2022 | 1.13 | 3.7578 | 5% |
| qm9 | 1 | pinned | ok | 1870 | 1.05 | 3.7578 | 5% |
| qm9 | 1 | timed | ok | 1939 | 1.08 | 3.7578 | 5% |
| qm9 | 1 | transport | ok | 1873 | 1.05 | 3.7578 | 5% |

No cell crashed.
