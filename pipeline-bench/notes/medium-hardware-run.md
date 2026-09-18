# Full matrix on medium hardware (cosmopc27)

Single RTX 4070 Ti SUPER, 6 variants x 3 datasets x workers `[0, 1, 4]` x
batch 8 x cuda x 6 epochs x 2 repeats, from `config.yaml`'s default sweep.
116 Snakemake jobs, ~16 min wall. Stands in for the common case of one
consumer/workstation GPU rather than a cluster node — small datasets
(qm9/carbon: 100 structures; si216: 64 structures x 216 atoms), so each
step is ~27-80 ms of GPU time and the numbers mostly expose CPU-side
loader/collate overhead, not raw model throughput.

Full table: `results/summary.md` (median of 2 repeats per cell).

## The transport/batching win is real, but conditional

`everything` and `transport` (the tensor-transport branches) cut
`serialize` from ~9.7 ms to ~0.6 ms per call on si216 (216 atoms/structure,
the heaviest per-item payload) and shrink the `loader` wait by ~30%. That
turns into a genuine +9-10% atoms/s over `baseline` on si216 — but only at
`num_workers=0`, where the main process does the collate/serialize work
itself. On qm9/carbon (much smaller per-structure payload), the same
branches are net *losses* at `workers=0` (qm9 -18%, carbon -3 to -5%): the
extra transform/transport bookkeeping doesn't pay for itself when there's
little to serialize.

## Regression once a worker is added

The realistic default is `num_workers>=1`. There, `baseline`'s plain
pipeline already overlaps loading with compute well on this GPU, and the
perf-branch stack becomes a straightforward regression:

- qm9, workers=1: baseline 2885 atoms/s vs `everything` 2144 (-26%), vs
  `transport` 2210 (-23%)
- carbon, workers=1: baseline 1166 vs `everything` 967 (-17%)
- si216, workers=1: baseline 22333 vs `everything` 22783 (+2%, roughly flat)

`persistent`, `timed`, and `pinned` sit between baseline and
`everything`/`transport` at workers>=1 — consistently 5-15% under
baseline on qm9/carbon, roughly flat on si216. `workers=4` narrows the gap
but never closes it for qm9/carbon; si216 is the one dataset where
`everything` edges back past baseline (+7%).

## Read for the PR stack

These branches look tuned for a regime where the loader is the actual
bottleneck: many workers, large structures, likely multi-GPU contention.
On one modest GPU with `num_workers=0` and the largest per-structure
dataset, the win shows up exactly as advertised. Everywhere else in this
matrix — which is most of it, since `num_workers>=1` is the realistic
setting — the added machinery currently costs more than it saves. Worth
raising on #1275 before merging the whole stack: the benefit is real but
narrower than "everything" implies, and it doesn't hold on the class of
hardware a lot of users actually train on.

## Caveats

- `baseline`'s `benchmark_pipeline.py` (the oldest branch in the stack)
  doesn't print `n_train`/`n_val`/peak-GPU-memory the way the later
  branches do, so `results/summary.md` shows `—` for baseline's peak GB —
  that's a report-format gap between branches, not zero memory use.
- 2 repeats per cell; treat the percentages above as directional, not
  noise-free.
