# Checkpoint save/resume check

Per (variant, workers): train straight through vs. train, checkpoint, restart via the real `mtt train --restart` path, and finish. A CRASH means resume itself is broken for that variant/worker count — the actual regression this check exists to catch. A drift flag on a completed run is expected background noise (the checkpoint doesn't carry RNG state) unless it's large or one-sided across variants.

| variant | dataset | workers | status | continuous best | resumed best | drift | resume overhead (s) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| baseline | qm9 | 0 | ok | 0.001305 | 0.001163 | 11% | -1.12 |
| baseline | qm9 | 1 | ok | 0.001161 | 0.001351 | 14% | -0.72 |
| everything | qm9 | 0 | ok | 0.0012 | 0.001239 | 3% | -1.39 |
| everything | qm9 | 1 | ok | 0.001487 | 0.00123 | 17% | -0.81 |
| persistent | qm9 | 0 | ok | 0.0012 | 0.001239 | 3% | -1.48 |
| persistent | qm9 | 1 | ok | 0.001487 | 0.00123 | 17% | -1.03 |
| pinned | qm9 | 0 | ok | 0.0012 | 0.001239 | 3% | -1.42 |
| pinned | qm9 | 1 | ok | 0.001487 | 0.00123 | 17% | -1.03 |
| timed | qm9 | 0 | ok | 0.0012 | 0.001239 | 3% | -1.34 |
| timed | qm9 | 1 | ok | 0.001487 | 0.00123 | 17% | -0.78 |
| transport | qm9 | 0 | ok | 0.0012 | 0.001239 | 3% | -1.38 |
| transport | qm9 | 1 | ok | 0.001487 | 0.00123 | 17% | -1.04 |

All 12 cell(s) completed the checkpoint save/restart round trip without error.
