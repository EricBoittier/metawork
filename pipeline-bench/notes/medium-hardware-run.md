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

## Larger dataset: the win holds at every worker count

The matrix above tops out at si216 (64 structures x 216 atoms) — still
tiny. Added `si_large`: 128 structures x 1000 atoms (`make_si_bulk.py
--n-cells 5 --n-structures 128`, ~4.6x the atoms/structure and 2x the
structures of si216), same sweep. Once structures are actually large,
`everything`/`transport` win outright, at every `num_workers`:

| workers | baseline atoms/s | everything | vs baseline | transport | vs baseline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 21788 | 25195 | +16% | 25819 | +19% |
| 1 | 26708 | 29774 | +11% | 29041 | +9% |
| 4 | 26800 | 29747 | +11% | 29159 | +9% |

The mechanism is `unpack` (a substage of `step`, not `loader`): baseline
spends 33-37 ms/call unpacking the batch back into structures regardless
of worker count, since that's fixed per-batch tensor work, not something
extra workers can overlap away. `everything`/`transport` cut it to
6-8 ms/call — a real ~5x reduction that shows up as lower `step` time at
every worker count, not just `workers=0`. `serialize` at `workers=0`
drops even harder, ~38-44 ms to ~2-4 ms.

`persistent`, `pinned`, and `timed` are flat against baseline here too
(0.95-1.00x at every worker count) — they target loader/worker-lifecycle
overhead, which was never the bottleneck once atoms/structure gets large;
the unpack/serialize path only the transport branches touch is what
actually costs on bigger structures.

## Read for the PR stack

The two behaviors together sharpen the picture: `everything`/`transport`'s
win tracks per-structure payload size, not worker count. On tiny molecules
(qm9/carbon, ~20-40 atoms) the added bookkeeping is pure overhead and
costs 3-26% once a worker is added; on 1000-atom structures the same code
path saves a fixed ~25-30 ms/batch of unpack/serialize work that no amount
of worker overlap removes, and it wins by 9-19% everywhere. So the
regression isn't really a "medium hardware" problem — it's a "this
benchmark's default structures are too small" problem. Worth raising on
#1275: the perf claim holds for the workloads metatrain users training on
bulk/large-cell systems will actually see, but the stack should not be
sold as an unconditional win — on molecular-scale structures with
`num_workers>=1`, it's currently a regression, and `persistent`/`pinned`/
`timed` don't clearly pay for their own complexity in this matrix at all.

## Caveats

- `baseline`'s `benchmark_pipeline.py` (the oldest branch in the stack) is
  not an equivalent workload to the others, not just a reporting gap: its
  validation set was the first 8 structures *of the training set itself*
  (no disjoint split), while every later branch trains on a proper 80/20
  split. That's why `results/summary.md` also showed `—` for baseline's
  `n_train`/`n_val`/peak-GPU-memory — its header line didn't match the
  format the parser expects from the later branches, so those fields never
  even got extracted. Doesn't bias the throughput numbers here (validation
  sits outside every timed region), but it means baseline was never a
  clean, isolated control. Fixed on metatrain's
  `fix/pipeline-benchmark-val-split` (branched off
  `perf/1-pipeline-benchmark`); results above predate that fix.
- No correctness/equivalence check exists between variants anywhere in this
  harness — every number here is wall-clock speed. A branch that's faster
  but silently wrong (drops a structure, corrupts a batch) would look
  identical to a real win in this data. `results/equivalence.md` (added
  after this run) catches a workload-shape mismatch like the one above; it
  does not check model correctness.
- 2 repeats per cell; treat the percentages above as directional, not
  noise-free.
