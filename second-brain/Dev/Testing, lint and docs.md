---
tags: [dev, testing]
---
# Testing, lint and docs

Most metatensor repos use tox.
```bash
tox -l                                  # list envs
tox -e lint                             # ruff/format checks
tox -e format                           # auto-format (where defined)
tox -e docs                             # build sphinx docs
tox -e ase-tests -- tests/calculator.py::test_model_with_extensions   # one test via pytest args
uv run tox -e docs                      # if tox isn't installed globally
```
Plain pytest in the venv:
```bash
pytest path/test_file.py::test_name -x -q
pytest -k "keyword" -x
```
Formatting a single file: `uv run ruff format file.py` / `ruff check --fix file.py`.
