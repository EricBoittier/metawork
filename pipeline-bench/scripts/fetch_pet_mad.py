"""Download a PET-MAD checkpoint from HuggingFace, once, to a local cache path."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


HF_PATH = (
    "https://huggingface.co/lab-cosmo/upet/resolve/main/models/"
    "pet-mad-{size}-v{version}.ckpt"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, help="xs, s, or m")
    parser.add_argument("--version", required=True, help="e.g. 1.6.0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    url = HF_PATH.format(size=args.size, version=args.version)
    urlretrieve(url, args.out)


if __name__ == "__main__":
    main()
