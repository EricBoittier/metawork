---
tags: [tools, python, uv]
---
# uv

Installed: `~/.local/bin/uv` (0.12.x). Update: `uv self update`. Envs overview incl. conda: [[Conda and uv envs]].

## Projects (pyproject.toml + uv.lock)
```bash
uv init mypkg [--lib]            # new project
uv add numpy "torch>=2.5"        # add dependency (updates pyproject + lock + .venv)
uv add --dev pytest ruff         # dev group
uv add --optional cuda cupy      # extras
uv remove numpy
uv lock                          # resolve without installing; `uv lock --upgrade-package torch`
uv sync                          # make .venv match the lock exactly
uv sync --extra torch --group dev
uv run pytest -x                 # run in the project env (syncs first)
uv run --with ipython ipython    # one-off extra package
uv tree                          # dependency tree
uv export --format requirements-txt > requirements.txt
```

## pip-style (what metawork uses)
```bash
uv venv [--python 3.12] [.venv]  # create venv
source .venv/bin/activate
uv pip install -e .              # editable
uv pip install -e '.[torch]'
uv pip install -r requirements.txt
uv pip install --no-build-isolation -e ./python/metatomic_torch   # build against installed torch
uv pip list / uv pip freeze / uv pip show torch
uv pip uninstall pkg
uv pip compile requirements.in -o requirements.txt
uv pip install --system pkg      # into the non-venv python (avoid)
```

## PyTorch index
```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121   # kuma shared venv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell / GH200 (aarch64)
```
In `pyproject.toml`:
```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
metatomic = { path = "../metatomic/python/metatomic_torch", editable = true }
```
Pitfall: `uv pip install -e pkg[torch]` can silently swap torch for PyPI's default CUDA build → `etc/fix-torch-cuda.sh`. Pin with `-c pipeline-bench/slurm/torch-constraints.txt`.

## Python versions
```bash
uv python list
uv python install 3.12
uv python pin 3.12               # writes .python-version
```

## Tools (like pipx)
```bash
uvx ruff check .                 # run a tool without installing
uv tool install tox              # put on PATH (~/.local/bin)
uv tool list / uv tool upgrade --all
```
Scripts with inline deps:
```python
# /// script
# dependencies = ["ase", "numpy"]
# ///
```
`uv run script.py` builds a throwaway env.

## Cache / disk
```bash
uv cache dir / uv cache clean / uv cache prune
export UV_CACHE_DIR=$SCRATCH/.uv-cache     # on clusters with small home
export UV_LINK_MODE=copy                   # when cache and venv are on different filesystems
```

Related: [[Python]], [[Conda and uv envs]]
