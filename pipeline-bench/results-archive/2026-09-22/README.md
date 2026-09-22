# Archived results, 2026-09-22

`pipeline-bench/results/` is gitignored (see `pipeline-bench/.gitignore`) so
it can be regenerated freely, but that also means nothing in it survives
across sessions or machines unless copied out by hand — this bit twice in
one session (an in-progress smoke test overwrote `resume.md`/`pet_mad.md`/
`flashmd.md` with no way to recover the originals). This directory is a
manual snapshot of the small, durable aggregate reports from that point in
time, not a replacement for the gitignored `results/` working directory.

Context: `notes/2026-09-22-verification-run.md` and
`notes/handoff-cosmopc27-2026-09-22.md` in this same `notes/` sibling
directory.

## What's here

- `summary.md`, `correctness.md`, `equivalence.md`, `drift.md`, `resume.md`,
  `pet_mad.md`, `flashmd.md` — the aggregate `.md` reports as they stood
  after re-running `--config correctness=true` (all 6 variants, fix
  branches merged in) on cosmopc27 (1x RTX 4070 Ti SUPER). `correctness.md`
  is byte-for-byte identical to the original Sep 18 run; the rest matched
  its qualitative claims within normal run-to-run spread — see the notes
  above for the full comparison.
- `cells.csv`, `stages.csv` — the tidy per-cell tables `summary.md` is built
  from. Covers `workers=0`, `batch_size=8` only (the `correctness=true`
  scope) — **not** the full `workers=[0,1,4]` matrix `summary.md` in the
  original Sep 18 results described; that fuller sweep's raw JSON still
  exists under the (gitignored) `results/cells/*/w1_*`/`w4_*` files on
  cosmopc27 as of this writing, but was not re-aggregated here.
- `profile/{baseline,everything}_qm9_w1.table.txt` — `torch.profiler`
  kernel-level summaries (workers=1, qm9) from the `backward`-timing
  investigation in the handoff note. The matching ~460MB `.trace.json`
  files are **not** archived here (too large for git); they exist locally
  under `pipeline-bench/results/profile/` on cosmopc27 if needed again,
  best viewed at ui.perfetto.dev.

## What's not here

Per-cell JSON/log files (`results/cells/`, `results/resume_cells/`,
`results/pet_mad_cells/`, `results/flashmd_cells/`) — many small files, and
the aggregates above are the values that actually matter; regenerate them
with Snakemake if the raw per-cell data is needed again.
