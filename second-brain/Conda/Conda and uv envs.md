---
tags: [conda, uv, python, env]
---
# Conda and uv envs

## Miniforge (conda)
Lives in `~/miniforge3`, **not on default PATH** on cosmopc7.
```bash
source ~/miniforge3/etc/profile.d/conda.sh       # make `conda` available
# install from scratch:
curl -fsSL -o /tmp/Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
bash /tmp/Miniforge3.sh -b -p $HOME/miniforge3
```

## conda essentials
```bash
conda env list
conda create -n NAME -c conda-forge python=3.12 pkg1 pkg2 -y
conda activate NAME
conda install -n NAME -c conda-forge pkg
conda update -n NAME pkg
conda env export -n NAME --from-history > env.yml   # portable spec
conda env create -f env.yml
conda remove -n NAME --all                          # delete env
conda clean --all                                   # free disk
```

## conda-smithy env (for feedstocks)
```bash
conda create -n conda-smithy -c conda-forge conda-smithy -y
conda update -n conda-smithy conda-smithy
```
Used by [[Conda feedstock protocol]].

## uv (metawork venv)
Shared venv: `~/Documents/metawork/.venv` (Python 3.12), built by `etc/setup-metawork.sh`.
```bash
uv venv                                   # create .venv here
source .venv/bin/activate
uv pip install -e .                       # editable install
uv pip install -e ./python/               # repos with python/ subdir
uv pip install -e '.[torch]'              # with extras
uv run ruff format path/file.py           # run a tool in the project env
uv run tox -e docs
```

## torch / CUDA mismatch
Symptom: after `uv pip install -e pkg[torch]`, torch got replaced by PyPI's default cu130 wheel the driver can't run.
```bash
bash ~/Documents/metawork/etc/fix-torch-cuda.sh   # restores a working wheel + rebuilds *-torch packages
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Known conflicts
- `metatomic[torchsim]` pins `vesin<0.6`, `metatomic-ase` wants `vesin>=0.6` → don't install both extras together.
