# Run-order drift check

`relative` = atoms/s divided by the median atoms/s for that exact (dataset, workers, batch, device) combination — 1.0 means typical for its own config, so this is comparable across configs and variants.

**Global** (all 72 cells, chronological): Spearman rho = 0.70
  - **FLAGGED**: cells run later in the matrix trend faster relative to their own config's median, independent of variant. That's consistent with thermal drift, background load, or a warming/cooling GPU over the run — treat variant-vs-baseline deltas for whichever variants ran at the extremes of the run with extra skepticism.

**Per-variant** (order within that variant's own block):
- `baseline`: rho = -0.57, n = 12 — **FLAGGED**
- `everything`: rho = -0.14, n = 12
- `persistent`: rho = -0.07, n = 12
- `pinned`: rho = 0.41, n = 12 — **FLAGGED**
- `timed`: rho = 0.17, n = 12
- `transport`: rho = -0.15, n = 12

A flagged rho is a correlation, not a diagnosis — it says the data is consistent with an order effect, not that one is proven. Rerunning the flagged variant's cells interleaved with another variant's (or just rerunning the matrix and checking whether the flag persists) is the actual follow-up, not adjusting the numbers based on this alone.
