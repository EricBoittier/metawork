---
tags: [python, lang]
---
# Python

Envs: see [[Conda and uv envs]]. Tests/lint: [[Testing, lint and docs]].

## Packaging
- `pyproject.toml` is the source of truth; build backends: `setuptools`, `hatchling`, `scikit-build-core` (CMake), `maturin` (Rust). metatensor packages ship custom backends under `build-backend/`.
```bash
uv pip install -e .                  # editable
python -m build                      # sdist + wheel into dist/
pip wheel . -w dist --no-deps
unzip -l dist/*.whl                  # check what got packaged
python -c "import pkg, sys; print(pkg.__file__)"   # which copy is imported?
pip show -f pkg                      # installed files
```

## Debugging
```python
breakpoint()                         # pdb; `PYTHONBREAKPOINT=0` disables
```
```bash
python -X faulthandler script.py     # traceback on segfault
python -m pdb -c continue script.py  # post-mortem on crash
py-spy dump --pid PID                # where is a hung process stuck
py-spy record -o prof.svg -- python script.py   # flamegraph, no code change
python -m cProfile -s cumtime script.py | head -40
```

## Performance reminders
- Vectorise with numpy/torch; Python loops over atoms/frames are the usual bottleneck.
- `functools.lru_cache` / `cache` for pure functions.
- Measure with `timeit` (`python -m timeit -s "setup" "stmt"`), not one `time.time()` delta.
- Multiprocessing for CPU-bound work (GIL); threads only help for I/O or C code that releases the GIL.

## Idioms
```python
from pathlib import Path
paths = sorted(Path("data").glob("*.xyz"))
by_key = {k: [x for x in xs if x.key == k] for k in {x.key for x in xs}}
first, *rest = items
with open(p) as f: lines = f.read().splitlines()
```
Typing: `from __future__ import annotations`; `X | None` instead of `Optional[X]`.

Related: [[PyTorch]], [[JAX]]
