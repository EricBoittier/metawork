#!/bin/bash
# Rebuild site/*.html from the notebooks in this directory.
# Run from /home/boittier/analysis. Does NOT execute any notebook -- it only
# renders whatever outputs are already stored in each .ipynb. To refresh a
# notebook's own outputs first:
#   .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
#       --ExecutePreprocessor.kernel_name=python3 <notebook>.ipynb
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY=/home/boittier/metawork/.venv/bin/python
mkdir -p site pdf

for nb in *.ipynb; do
  [ "$nb" = "Untitled.ipynb" ] && continue
  "$PY" -m jupyter nbconvert --to html --output-dir=site "$nb"
  "$PY" -m jupyter nbconvert --to webpdf --output-dir=pdf "$nb"
done

echo "Rebuilt. Open site/index.html (HTML) or pdf/*.pdf"
