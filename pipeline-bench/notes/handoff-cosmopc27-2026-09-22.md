# Handoff: cosmopc27 session -> kuma session

Context for whoever's running the kuma-side work: this is what a parallel
session did on the single-GPU workstation (cosmopc27, RTX 4070 Ti SUPER)
today. Net take: useful as an independent cross-check, but the `workers>1`
performance question turned into a dead end here — kuma's larger-scale run
is the one making real progress on it.

## What got verified (still holds)

Picked up `bench/medium-hardware` cold and re-ran `--config correctness=true`
(72 cells: 6 variants x 4 datasets x 3 seeds) against the same fix-branch
refs the original Sep 18 run used. `results/correctness.md` came back
**byte-for-byte identical** to the Sep 18 version — the one non-noisy check
in this harness (fixed-seed training, not wall-clock). `persistent`/`timed`/
`pinned`/`transport`/`everything` still agree bit-for-bit with each other at
every dataset/seed; `baseline` still diverges 6-27% on `qm9`/`carbon`,
0-1% on `si216`/`si_large`. `resume.md`, `pet_mad.md`, `flashmd.md` matched
the README's qualitative claims within normal run-to-run spread. So: the
harness and its central findings are trustworthy, independent of whatever
else happened today.

## Bug found and fixed: `make_worktree.py`'s merge fallback

`scripts/make_worktree.py`'s `keep_both_sides()` conflict resolver is meant
for add/add conflicts (two branches inserting an unrelated kwarg at the same
site) but gets applied to *any* conflict, checked only for "no `<<<<<<<`
left" — not for valid Python. Hit a real modify/modify conflict rebuilding
the `everything` worktree (`perf/5-batch-transport` + its
`fix/report-best-metric-everything` fix branch) where both sides rewrote the
same `with Trainer(...)` statement differently; the naive concatenation
produced an `IndentationError` that got silently `git commit`ed as a clean
merge. Fixed the worktree by hand and hardened the script to `py_compile`
every `.py` file it auto-resolves and abort loudly instead of committing
garbage — this is a real, useful, not-yet-committed fix
(`pipeline-bench/scripts/make_worktree.py`), worth pulling into any other
checkout that rebuilds these worktrees.

## `origin/main` has moved, not rebased (by request)

4 commits ahead of local `main` that `bench/medium-hardware` doesn't have:
a `distributed:` sweep (real multi-GPU DDP via `srun --ntasks`, an
OMol25-proxy dataset generator for size-diverse structures up to 512
atoms/structure, sbatch templates under `slurm/`), a `SLURM_NTASKS`-leak fix
that same sweep needed, and — independently — the identical
`/home/ericb` -> `/home/boittier` path fix this branch also made on its
own. If kuma's session already has that distributed-sweep infrastructure
(sounds like it, from the earlier transcript about 144 cells across ~32
GPUs), it's ahead of what's on this branch; reconciling later should be
low-conflict since neither branch touches the same files, except that one
duplicated path fix.

## The `workers>=1` investigation: three ideas tried, all dead ends

Starting question: throughput matrix (this session's + the earlier
medium-hardware run) shows `everything`/`transport` winning over `baseline`
at `workers=0` but flat-to-regressing once a worker is added, except on the
largest structures. Wanted to know why.

Confirmed with real measurement (not just theory) that `loader` wait is
already ~0ms for *both* variants once `workers=1` — so there's no
loader-side wait left for the transport/pinning branches to reclaim, which
explains why they don't help there. But `everything` is also consistently
15-30% slower in `step` regardless of worker count, concentrated in
`backward` (10.6-12.0ms baseline vs 13.5-16.8ms everything, reproducible
across 9+ repeats) — while a `torch.profiler` comparison
(`results/profile/{baseline,everything}_qm9_w1.{trace.json,table.txt}`,
~460MB each, view in ui.perfetto.dev not chrome://tracing) shows *total* GPU
kernel time is essentially identical between the two (248.9ms vs 248.2ms
self CUDA over the whole run). That combination — same total GPU work, but
wall-clock `backward` consistently higher — is the actual puzzle.

Diffed every file that differs between the `baseline` and `everything`
worktrees: `trainer.py` and every PET model file are byte-identical except
one harmless no-op cast (`centers.long()` in
`pet/modules/structures.py` — this independently confirms the dtype
compatibility bug your kuma session found; it's a no-op on this machine's
torch version but presumably not on kuma's, which matches your report of 5/6
variants crashing there). Several other files differ only by a pure rename
(`transform` -> named functions like `divide_by_scale`, for the profiler's
per-transform labels, PR #1272) — no functional change.

That leaves `unpack_batch` in `utils/data/dataset.py` as the only real
functional difference on the hot path. Found and confirmed by direct tensor
inspection (not guessing) that `everything`'s reconstructed
`System.positions`/`types` are `torch.split()` *views* sharing the entire
batch's storage (111 elements backing a 15-element system) instead of an
owned allocation, and the neighbor-list `values`/`samples` that feed
`PET::backend::compute_features`/`preprocess` directly (the two dominant
kernels, ~85% of GPU time) have the same pattern, worse — 7.7x oversized
backing storage (462 elements for 60 elements of real data). Both are real,
verifiable structural differences from `baseline`'s `load_system_buffer`
path, which produces fully independent tensors.

Tested fixing both with `.clone()` (verified the storage anomaly was
actually gone at the tensor level both times) and re-measured with the same
3-repeat protocol each time: **no improvement either time** — the
neighbor-list fix actually made `backward` measurably worse (+5.35ms vs
baseline, worse than before the fix). Reverted both; nothing left uncommitted
in the worktrees.

## Where this leaves the `backward` gap

Unexplained. `unpack_batch`'s own instrumented sub-stages sum to ~1.2ms,
nowhere near the multi-ms gap, and the two concrete hypotheses that looked
most promising (view/storage overhead reaching into forward/backward) are
now ruled out by direct experiment. The remaining candidates are inside
PET's C++/CUDA backend itself (`compute_features`/`preprocess`) —
`with_stack=True` profiling or reading that backend's actual implementation
would be the next step, and that's a bigger, less certain effort than
anything tried today. If kuma's profiling setup (mentioned
`results/profile/{baseline,everything}_qm9.trace.json/table.txt` at
`workers=0`) has more GPU headroom or better tooling for that kind of dive,
it's probably the better place to keep going rather than repeating this
here.

## State of this checkout

Uncommitted: `pipeline-bench/scripts/make_worktree.py` (the real hardening
fix — worth keeping), plus notes files under `pipeline-bench/notes/`.
`config.yaml` and all worktrees are back to their clean, checked-in state
(fix-branch refs were only ever added locally/temporarily to run these
checks, per this harness's existing convention). Nothing pushed anywhere.
`results/` is gitignored, so nothing there survives except what's on this
disk — if kuma's session has a fuller `workers=[0,1,4]` matrix at real
scale, that supersedes the `workers=0`-only data this session was left
with.
