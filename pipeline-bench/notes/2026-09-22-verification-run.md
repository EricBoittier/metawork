# Verification run on cosmopc27 (no rebase, compared against the Sep 18 results)

Picking this harness back up cold, without re-reading old chat history: the
repo state (git log, `README.md`, `results/*.md`) is the only source of
truth, and `results/` is gitignored, so nothing in it survives except what's
physically still on this disk. Two things came out of just trying to run it
again: a real bug in the worktree-merge tooling, and confirmation that
`origin/main` has moved with unrelated pipeline-bench work this branch
doesn't have. Neither is fixed here beyond what's noted below — no rebase
was done, on request.

## `origin/main` has moved, `bench/medium-hardware` does not need it

Local `main` hasn't advanced past the commit `bench/medium-hardware`
branched from, but `origin/main` has 4 newer commits, all pipeline-bench
work done independently on that branch:

- `2652855` fixes the same `/home/ericb` -> `/home/boittier` path issue this
  branch's `0137bf0` fixed on its own, and adds a `distributed:` sweep
  target (multi-GPU DDP via `srun --ntasks`, an OMol25-proxy dataset
  generator, reusable sbatch templates under `slurm/`)
- `db10c96` fixes a real bug that sweep hit: `SLURM_NTASKS` leaking from an
  enclosing multi-task allocation into a `world_size=1` cell, hanging for a
  601s rendezvous timeout
- `57eb567`, `b186543` — unrelated submodule/sbatch-rendering fixes

None of these touch the correctness/resume/pet_mad/flashmd instrumentation
this branch added, so a future rebase should be low-conflict, but the two
branches indepedently fixed the same path issue — worth reconciling by hand
rather than trusting a mechanical merge. **Not done here**, per instruction.

## Bug found and fixed: `make_worktree.py`'s conflict fallback silently commits broken code

Tried to rebuild the `everything` worktree (`perf/5-batch-transport` merged
with its `fix/report-best-metric-everything` fix branch, same as this
harness has always done to get `--seed` and the resume/pet_mad/flashmd
scripts onto a variant) and hit an `IndentationError` at import time.

Root cause: `scripts/make_worktree.py`'s `keep_both_sides()` conflict
fallback is designed for *add/add* conflicts — two independent branches
each inserting an unrelated kwarg or line at the same spot, where keeping
both is the right composition. It's applied unconditionally to *any* git
merge conflict, though, and the only correctness check afterward was "no
`<<<<<<<` markers left" — a syntactic check that says nothing about whether
the concatenation makes sense. Here, both branches had rewritten the same
`with Trainer(...)` statement in `benchmark_pipeline.py` differently (one
added `capture_epoch_metrics()` to the `with`, one hoisted
`Trainer(...)` out of it) — a real modify/modify conflict — and
`keep_both_sides` concatenated both versions of the line into invalid
Python, which then got `git commit`ed as though it were a clean resolution:

```python
with TemporaryDirectory() as checkpoint_dir, monitor_memory() as stats:
trainer = Trainer(hypers["training"])
with TemporaryDirectory() as checkpoint_dir, monitor_memory() as stats, \
        capture_epoch_metrics() as epoch_metrics:
    ...
```

Fixed two ways:

1. **`scripts/make_worktree.py`** now `py_compile`s every `.py` file it
   auto-resolves and aborts the merge loudly (`SystemExit`, no commit)
   instead of silently committing something that doesn't parse. This is a
   real hardening, not a one-off patch — it will catch the same failure
   mode on any future rebuild of `everything` (or `pinned`, which goes
   through the identical fallback) without needing to be re-discovered.
2. The worktree itself: removed the redundant first `with` line, keeping
   the branch that already covers both `TemporaryDirectory`/
   `monitor_memory` plus `capture_epoch_metrics`. Confirmed with
   `py_compile` and by running the smoke cell, the resume check, and the
   FlashMD cell against the repaired worktree — all exit 0.

This is a live bug in local harness tooling (not the upstream metatrain
PRs), and it was hit fresh this session — the merge commit that produced
the broken file is timestamped today, so it did not affect any of the
already-recorded `results/*.md` from Sep 18, which were built before this
particular conflict shape existed on `fix/report-best-metric-everything`.
`config.yaml` was reverted back to its clean base-refs-only state afterward
(the fix-branch refs were only added locally, temporarily, to run this
verification — same convention the README already documents for how the
Sep 18 results were produced).

## Side effect: `results/resume.md`, `results/pet_mad.md`, `results/flashmd.md` got clobbered mid-session

Before finding the bug above, an initial `--config smoke=true` run (scoped
to the single `everything` variant) executed successfully far enough to
overwrite these three files with 2-row, single-variant content, since
`results/` is gitignored and there's no git history to recover the
original 12-row versions from. They're back to full 6-variant, 12-row
content now (see below), but the exact Sep 18 byte content of those three
specific files is gone. `cells.csv`, `stages.csv`, `summary.md`,
`correctness.md`, `equivalence.md`, `drift.md` were not affected — the
`aggregate` rule they depend on requires every variant's cells and never
ran during that partial smoke attempt, so those were backed up intact
before the fuller rerun below and are compared directly.

## Re-ran `--config correctness=true` (72 cells, all 6 variants, all always-on checks): confirms the Sep 18 finding

Same command as the historical run, same fix branches merged in (this time
recovering cleanly from the worktree bug above, mid-merge, once the
`py_compile` guard caught it rather than papering over it). 26 min wall
across two Snakemake invocations (the first hit the worktree bug at 78%
and stopped there per `keep-going: true`; the second, after the worktree
was repaired, finished the remaining `everything` cells plus
aggregate/checks in another ~4 min).

**`results/correctness.md` is byte-for-byte identical to the Sep 18
version.** This is the one file that matters most for trusting the
harness's central finding — it's not a wall-clock measurement (noisy by
nature) but a training-to-a-fixed-seed comparison, and it reproduced
exactly: `persistent`/`timed`/`pinned`/`transport`/`everything` still agree
bit-for-bit with each other at every dataset and seed, `baseline` still
diverges 6-27% on `qm9`/`carbon` and 0-1% on `si216`/`si_large`, same as
documented in the README's "What the correctness check already found"
table. The baseline-divergence question is still exactly as open as the
README says — this run didn't narrow it, just re-confirmed it holds.

**`results/equivalence.md`**: same OK verdicts, same `n_train`/`n_val`
numbers for every dataset. The only diff was a trailing sentence of static
script output text (updated in a commit between the two runs, not a data
change).

**`results/resume.md`**: all 12 cells still `ok`, no crashes — matches "all
12 cells completed without error" from before. Overhead/drift numbers per
cell differ slightly run to run (expected: the checkpoint doesn't carry RNG
state across the restart, so this was always documented as background
noise, not a stable number).

**`results/pet_mad.md`**: same qualitative pattern as the README describes
— `everything` still wins at `workers=0` (today: +3%, README: +7% — single-seed,
single-repeat measurement, some run-to-run spread is expected) and is
roughly flat at `workers=1` (+1% today vs. 0.99x before). `baseline`'s
`best_val_metric` gap is 1% at `workers=0` and 5% at `workers=1`, squarely
inside the README's "1-5%" range.

**`results/flashmd.md`**: same pattern — `baseline` differs from the other
five by 2% at `workers=0` (README: ~2%) and 0% at `workers=1`, still
byte-identical across the five non-baseline variants at both worker counts.

**`results/summary.md`**: numeric throughput values differ from the Sep 18
backup by a few percent per cell (e.g. si_large/everything atoms/s 25240 ->
25355, +17% vs baseline either way) — this is exactly the run-to-run wall-clock
noise the `spread` column and multi-repeat design already exist to capture,
not a regression. The qualitative story (structure size decides the win,
not worker count) is unchanged.

**`results/drift.md` changed** in a way that's an artifact of *how* this
particular run was executed, not a new physical finding: the first
Snakemake invocation ran `pinned` -> `timed` -> `persistent` -> `baseline`
-> `transport` in sequence before hitting the worktree bug, and the second
invocation (after the fix) ran `everything` alone, ~4 minutes later and
entirely after the other five. That's a much more lopsided chronological
ordering than Snakemake's normal single-pass scheduling produces, so the
newly-flagged global Spearman rho (0.70, up from 0.21) is confounded with
"which variant got shoved to the very end by the bug-fix retry," not
independent evidence of GPU thermal drift. Treat this run's `drift.md` as
uninformative for that question — a clean single-pass rerun would be needed
to say anything about drift.

## Bottom line

The harness itself is trustworthy: the one deterministic, non-noisy check
(`correctness.md`) reproduced exactly. The bug found and fixed
(`make_worktree.py` silently committing a broken merge) was in local
tooling, caught before it could taint any recorded result, and is now
guarded against for future worktree rebuilds. Nothing here narrows the
still-open question of *why* `baseline` diverges — that remains the one
item the README already flags as unresolved.
