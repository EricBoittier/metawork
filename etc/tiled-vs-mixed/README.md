# tiled vs mixed Ewald batching

Benchmarks two ways of batching the long-range (Ewald) branch of a PET+LR model across
structures of different size for one JIT-compiled train step:
`iris.pet.batching.TiledEwaldLR` (sum-padded atoms, tile-dispatched reciprocal sum) vs.
`mixed_lr.MixedEwaldLR` (max-padded rectangular k-space) -- does tiling let more structures
fit in one GPU batch, and how much faster is it per sample?

## Original design vs. what actually ran here

`env.sh`/`run_full.sh`/`verify.sh` are written for a **remote CSCS SLURM cluster**: they
`source`/`srun --environment=petlr`, read datasets from `/capstor/store/cscs/swissai/...`,
and request 4 GPUs. None of that exists on this machine, which is why this directory
originally had scripts but **no generated results** -- the benchmark had never actually been
run here, only staged/copied down.

`REPORT.md` and `results.json` in this directory now hold a **real, locally-executed**
regeneration, adapted to run on one workstation GPU:

- `prep_local_bio_dataset.py` builds a small `marathon.grain.DataSource` dataset (400
  periodic structures, energy+forces, no stress) from this project's own `bio_dimers`
  dataset (`lorem-tmlr-archive/datasets/bio_dimers_train.xyz`), standing in for the
  unavailable MAD/OC25 datasets.
- `bench.py` has a new `"local_bio"` entry in `CONFIGS` pointing at that dataset, with
  `num_k` reduced from 4096 to 1024 (bio_dimers' 30 A cells don't need MAD's resolution).
- Two extra dependencies had to be installed into `.venv-lorem-jax`: `petjax` (from its
  public repo, `github.com/lab-cosmo/pet-jax`) and `iris` (editable install of this
  workspace's own `iris-infra` submodule) -- plus a CUDA-enabled `jaxlib` (the venv only had
  a CPU build).
- `report.py` had two real bugs fixed while adapting it: the loss-weights line in the
  generated report was a hardcoded string (always showed the `mad` config's weights,
  regardless of which config actually ran) and the title/production-run labels had no entry
  for a non-`mad`/`oc25` config. Both are now derived from the actual config, and a
  `local_bio`-specific Caveats paragraph and "Reproducing" snippet were added.

**Headline result** (see `REPORT.md` for the full writeup): on this dataset's narrow
structure-size spread (mean 18.7, max 25 atoms/structure) and a 128-structure pool, `tiled`
and `mixed` are statistically indistinguishable in speed and memory at every batch size
tested (within ~1%), and neither backend ever OOMed -- the pool was simply too small and too
size-uniform to expose the gap the original benchmark (presumably run against MAD's much
more size-diverse structures) was designed to find. The padding-waste mechanism is real
(tiled wastes 1.09x vs. mixed's 1.35x at the largest batch tested) but too small here to
show up in wall-clock time once diluted by the SR trunk.

## Rerunning

```bash
export SCRATCH=/path/to/scratch
export DATASETS=/path/to/scratch/tvm_local_datasets
python prep_local_bio_dataset.py
python bench.py prep --config local_bio --pool 128 --workers 4 --cache "$SCRATCH/tvm_pool.pkl"
python sweep.py --config local_bio --cache "$SCRATCH/tvm_pool.pkl" --pool 128 \
    --out results.json --steps 8 --warmup 2 --bisect 4
python report.py --results results.json --out REPORT.md
```

Needs `.venv-lorem-jax` (or an equivalent env) with `jax[cuda12]`, `flax`, `optax`,
`marathon-train`, `petjax` (`pip install git+https://github.com/lab-cosmo/pet-jax`), and
`iris` (`pip install -e <path-to-iris-infra>`).
