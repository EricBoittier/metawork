<!--
Draft for a comment on metatensor/metatrain#1275. NOT POSTED — review and
post by hand (gh pr comment 1275 --body-file ..., or the web UI).
Written from pipeline-bench's results, cosmopc27 (1x RTX 4070 Ti SUPER),
2026-09-18 (full matrix + first correctness run) and 2026-09-22 (correctness
rerun that reproduced it, plus the two checks below).
-->

## Independent benchmark: input-pipeline harness results on a single consumer GPU

Ran the stacked branches here through a separate Snakemake harness
([pipeline-bench](../../../pipeline-bench), not part of this repo) on a
single RTX 4070 Ti SUPER — a stand-in for the common case of one
workstation/consumer GPU rather than a cluster node. Two independent
things came out of it: a throughput result that's more conditional than
the PR description's headline number, and a correctness check that found
`baseline`'s reported metrics don't consistently match the rest of the
stack.

### Throughput: the win tracks structure size, not worker count

6 variants (`baseline` = pre-#1268, `persistent` = #1268-#1271,
`timed` = #1274, `pinned` = #1274 + #1273, `transport` = the tensor-transport
change (not yet opened on its own), `everything` = the full stack here) x
4 datasets x `num_workers` in `{0, 1, 4}` x batch_size=8 x 6 epochs x 2
repeats.

- **Small structures (qm9 ~10 atoms/structure, carbon ~4, both 100
  structures): a regression once a worker is added.** At `workers=1`,
  `baseline` already overlaps loading with compute well on this GPU;
  `everything`/`transport` cost 17-26%, `persistent`/`timed`/`pinned` cost
  5-15%.
- **216-atom periodic silicon (64 structures): wins only at `workers=0`**
  (+9-10% atoms/s for `everything`/`transport`), close to flat or a loss
  once a worker is added.
- **1000-atom periodic silicon (128 structures): wins outright, at every
  worker count** — `everything`/`transport` +9-20% over `baseline`
  regardless of `num_workers`. Mechanism is `unpack`, not `loader`:
  ~33-37 ms/batch on `baseline` regardless of worker count (fixed
  per-batch tensor work extra workers can't overlap away), cut to
  ~6-8 ms/batch by the transport branches.
- `persistent`/`pinned`/`timed` stay within 0.95-1.00x of `baseline` on
  the 1000-atom set at every worker count — the loader/worker-lifecycle
  overhead they target was never the bottleneck once structures are
  large; only the unpack/serialize path the transport branches touch
  pays off there.

Net: the PR's own +15.8% number (216-atom Si, 4 workers) is real in the
regime it was measured in, but doesn't generalize to `num_workers>=1` on
molecule-scale structures, where this matrix shows a 3-26% regression
instead. Worth stating explicitly in the PR description rather than as an
unconditional win. (Full write-up with per-cell numbers: pipeline-bench's
`notes/medium-hardware-run.md`.)

### Correctness: `baseline` doesn't train to the same place as the rest of the stack

None of the individual PRs report whether their `benchmark_pipeline.py`
still trains to the same result as the branch it's stacked on, only how
fast it runs. Added `--seed` plus reporting `best_val_metric` and
epoch-1 train/val loss to each variant's benchmark script (needs
`fix/pipeline-benchmark-val-split` / `fix/report-best-metric*` merged in
per variant — see pipeline-bench's README for the exact refs) and ran 3
fixed seeds x 4 datasets x 6 variants.

- `persistent`/`timed`/`pinned`/`transport`/`everything` agree with each
  other **bit-for-bit**, every dataset, every seed. No disagreement among
  the five newer branches anywhere.
- `baseline` diverges from all five, and the pattern is structure-size
  dependent: 0-1% on the two bulk Si datasets, **6-27%** on
  `qm9`/`carbon` (one cell — `qm9` seed 0 `epoch1_val_loss` — crossed 50%
  outright).
- Ruled out the data-loading changes as the cause: also ran this on
  FlashMD (`experimental.flashmd`), a different architecture whose
  trainer is byte-identical across all six variants (confirmed by diff —
  none of these branches touch it). `baseline` still diverges from the
  other five by ~2% there too, so whatever's behind it is something about
  that branch or its build/dependency environment, not the pipeline
  changes themselves. Still open which — flagging here in case it rings
  a bell, since `baseline` is also this stack's control for every other
  number in this PR.

### Reproducibility

Re-ran the correctness check today against the same fix branches (72
cells, 6 variants x 4 datasets x 3 seeds) after finding and fixing an
unrelated bug in the harness's own worktree-merge tooling (not in any of
this PR's branches). `results/correctness.md` came back **byte-for-byte
identical** to the run above — same bit-for-bit agreement among the five
newer branches, same 6-27%/0-1% baseline divergence pattern. The
throughput/regression numbers above are from the original run only
(`workers=1`/`4` weren't re-swept today, just `workers=0`); the
correctness finding is the one independently reproduced today.

### Caveats

Small datasets (100-128 structures), 6 epochs, batch_size=8 only, single
consumer GPU — not this repo's cluster hardware, and not the scale users
training large models will actually see. Treat the throughput percentages
above as directional. Happy to share the harness / raw results if useful
for a larger run.
