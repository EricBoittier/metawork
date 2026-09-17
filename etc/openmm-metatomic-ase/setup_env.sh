#!/bin/bash
# Build a throwaway venv that can `mtt export` a HF checkpoint and run it
# through openmm-ml's "ase" potential with metatomic_ase.MetatomicCalculator.
#
# Reuses whatever ase/torch/metatomic/metatomic_ase/vesin are already on the
# system (--system-site-packages), and adds openmm + the two vendored repos
# (editable, --no-deps to skip their metatomic-torch/-ase version pins, which
# are older than the dev builds in this workspace).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install openmm
.venv/bin/pip install --no-deps -e ../../openmm-ml
.venv/bin/pip install --no-deps -e ../../metatrain
.venv/bin/pip install jsonschema pydantic omegaconf python-hostlist tqdm typing_extensions \
    metatensor-learn metatensor-operations

echo "done: .venv/bin/python, .venv/bin/mtt"
