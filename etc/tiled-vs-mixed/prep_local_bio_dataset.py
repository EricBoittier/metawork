"""Build a local marathon.grain DataSource folder from bio_dimers_train.xyz,
standing in for the (locally unavailable) MAD/oc25 datasets that
tiled_vs_mixed/bench.py's CONFIGS were written for.
"""
import ase.io

from marathon.grain.data_source.prepare import prepare

SRC = "/home/boittier/metawork/lorem-tmlr-archive/datasets/bio_dimers_train.xyz"
OUT = "/tmp/claude-342316/-home-boittier-metawork/c18ff8ed-781b-4c12-8468-2336bf0996b1/scratchpad/tvm_local_datasets/local/bio_dimers"
N = 400

PROPERTIES = {
    "energy": {"shape": (1,), "storage": "atoms.calc"},
    "forces": {"shape": ("atom", 3), "storage": "atoms.calc"},
}


def main():
    print(f"reading {N} frames from {SRC}")
    atoms_list = ase.io.read(SRC, index=f"0:{N}")
    print(f"read {len(atoms_list)} frames; writing marathon dataset to {OUT}")
    prepare(atoms_list, folder=OUT, properties=PROPERTIES, samples_per_composition=25)
    print("done")


if __name__ == "__main__":
    main()
