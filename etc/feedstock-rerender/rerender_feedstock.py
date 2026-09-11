"""conda-smithy rerender that does not explode unused pinning keys.

Plain `conda-smithy rerender` on lammps-metatomic cartesian-products every
multi-value pin (python, ROOT, Arrow, ...) times kokkos x mpi (~360k variants)
and hangs. Collapsing unused keys leaves the used CUDA/kokkos/mpi/libtorch
matrix intact.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

KEEP_MULTI = {
    "c_compiler",
    "c_compiler_version",
    "c_stdlib",
    "c_stdlib_version",
    "channel_sources",
    "channel_targets",
    "cuda_compiler",
    "cuda_compiler_version",
    "cxx_compiler",
    "cxx_compiler_version",
    "docker_image",
    "extend_keys",
    "fortran_compiler_version",
    "kokkos_arch",
    "libtorch",
    "llvm_openmp",
    "mpi",
    "mpich",
    "openmpi",
    "pin_run_as_build",
    "pytorch",
    "replacements",
    "target_platform",
    "zip_keys",
}


def _require_feedstock(root: Path) -> None:
    if not (root / "conda-forge.yml").is_file():
        sys.exit(f"error: {root} is not a feedstock (no conda-forge.yml)")
    if not (root / "recipe" / "meta.yaml").is_file():
        sys.exit(f"error: {root} has no recipe/meta.yaml")
    nested = root.parent.name == "metawork" and root.name.endswith("-feedstock")
    if nested:
        sys.exit(
            f"error: {root} looks like a nested clone inside metawork.\n"
            f"Use the sibling checkout instead, e.g.\n"
            f"  {root.parent.parent / root.name}\n"
            f"Pass --force if you really mean this directory."
        )


def _patch_explode(root: Path) -> None:
    from conda_build.utils import ensure_list
    from conda_build.variants import explode_variants as orig_explode
    from conda_build.variants import find_used_variables_in_text
    import conda_build.variants as variants_mod

    recipe_text = (root / "recipe" / "meta.yaml").read_text(encoding="utf-8")

    def collapse_unused(spec: dict) -> dict:
        used = set(find_used_variables_in_text(tuple(sorted(spec)), recipe_text))
        used.update(KEEP_MULTI)
        for group in spec.get("zip_keys") or []:
            group = list(group)
            if any(k in used for k in group):
                used.update(group)
        collapsed = dict(spec)
        for key, value in spec.items():
            if key in used or isinstance(value, dict):
                continue
            values = ensure_list(value)
            if len(values) > 1:
                collapsed[key] = values[:1]
        return collapsed

    def explode_variants(spec):
        return orig_explode(collapse_unused(spec))

    variants_mod.explode_variants = explode_variants
    variants_mod.dict_of_lists_to_list_of_dicts = explode_variants


def _restore_readme(root: Path) -> None:
    subprocess.run(
        ["git", "restore", "--staged", "README.md"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["git", "restore", "README.md"],
        cwd=root,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "feedstock",
        nargs="?",
        default=".",
        help="feedstock directory (default: cwd)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow a nested metawork clone",
    )
    args = parser.parse_args(argv)
    root = Path(args.feedstock).resolve()
    if args.force:
        if not (root / "conda-forge.yml").is_file():
            sys.exit(f"error: {root} is not a feedstock (no conda-forge.yml)")
    else:
        _require_feedstock(root)

    os.chdir(root)
    print(f"rerendering {root}", flush=True)
    _patch_explode(root)

    from conda_smithy.cli import main as smithy_main

    sys.argv = ["conda-smithy", "rerender"]
    smithy_main()
    _restore_readme(root)
    print(
        "\nREADME.md restored if smithy touched it. Review with:\n"
        f"  git -C {root} status\n"
        f"  git -C {root} diff",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
