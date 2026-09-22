# FlashMD benchmark

FlashMD's trainer has no per-stage timing instrumentation (unlike PET's), and none of the perf/data-loading branches this harness compares touch its code at all — confirmed by diff against the stacked PRs. So a difference between variants here cannot be attributed to those changes; it says something about the variant's branch/environment more generally.

| workers | variant | status | wall_s | structures/s | best_val_metric | vs baseline |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 0 | baseline | ok | 4.7 | 42.6 | 0.029255 | ref |
| 0 | everything | ok | 4.4 | 45.5 | 0.028559 | 2% |
| 0 | persistent | ok | 4.5 | 44.4 | 0.028559 | 2% |
| 0 | pinned | ok | 4.7 | 42.6 | 0.028559 | 2% |
| 0 | timed | ok | 4.4 | 45.5 | 0.028559 | 2% |
| 0 | transport | ok | 4.3 | 46.5 | 0.028559 | 2% |
| 1 | baseline | ok | 6.5 | 30.8 | 0.028452 | ref |
| 1 | everything | ok | 4.3 | 46.5 | 0.028557 | 0% |
| 1 | persistent | ok | 4.1 | 48.8 | 0.028557 | 0% |
| 1 | pinned | ok | 4.2 | 47.6 | 0.028557 | 0% |
| 1 | timed | ok | 4.2 | 47.6 | 0.028557 | 0% |
| 1 | transport | ok | 4.1 | 48.8 | 0.028557 | 0% |

No cell crashed.
