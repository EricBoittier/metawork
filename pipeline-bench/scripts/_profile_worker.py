"""Profile one PET training run with torch.profiler, against one variant worktree.

Run this with `<worktree>/src` on PYTHONPATH (same convention as run_cell.py),
so it profiles whichever variant's code is shadowing the run env's metatrain
install. Mirrors benchmark_pipeline.py's dataset/model setup exactly (same
build_dataset, DatasetInfo, get_default_hypers("pet"), Trainer.train call
signature -- confirmed identical across every variant's benchmark_pipeline.py)
so the two profiles differ only in what the variant's code path actually does.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile

from metatrain.pet import PET, Trainer
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import Dataset, DatasetInfo, get_atomic_types
from metatrain.utils.data.readers import read_systems, read_targets
from metatrain.utils.hypers import init_with_defaults
from metatrain.utils.loss import LossSpecification


def build_dataset(path: str, key: str) -> Tuple[Dataset, Dict[str, Any]]:
    conf = {
        "energy": {
            "quantity": "energy",
            "read_from": path,
            "reader": "ase",
            "key": key,
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "num_subtargets": 1,
            "forces": False,
            "stress": False,
            "virial": False,
        }
    }
    targets, target_info = read_targets(OmegaConf.create(conf))
    systems = read_systems(path)
    dataset = Dataset.from_dict({"system": systems, "energy": targets["energy"]})
    return dataset, target_info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-trace", type=Path, required=True)
    parser.add_argument("--out-table", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset, target_info = build_dataset(args.dataset, args.key)
    dataset_info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=get_atomic_types(dataset),
        targets=target_info,
    )

    loss = OmegaConf.create({"energy": init_with_defaults(LossSpecification)})
    OmegaConf.resolve(loss)

    hypers = get_default_hypers("pet", base_precision=32)
    hypers["training"].update(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        loss=loss,
    )
    model = PET(hypers["model"], dataset_info)

    val_size = max(1, round(args.val_fraction * len(dataset)))
    split = len(dataset) - val_size
    train_dataset = torch.utils.data.Subset(dataset, range(split))
    val_dataset = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    trainer = Trainer(hypers["training"])

    activities = [ProfilerActivity.CPU]
    if args.device == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with TemporaryDirectory() as checkpoint_dir:
        with profile(
            activities=activities,
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
        ) as prof:
            trainer.train(
                model=model,
                dtype=torch.float32,
                devices=[torch.device(args.device)],
                train_datasets=[train_dataset],
                val_datasets=[val_dataset],
                checkpoint_dir=checkpoint_dir,
            )

    args.out_trace.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.out_trace))
    table = prof.key_averages().table(
        sort_by="cuda_time_total" if args.device == "cuda" else "cpu_time_total",
        row_limit=40,
    )
    args.out_table.write_text(table + "\n")
    print(f"best_val_metric {trainer.best_metric:.6f} at epoch {trainer.best_epoch}")
    print(f"trace -> {args.out_trace}")
    print(f"table -> {args.out_table}")


if __name__ == "__main__":
    main()
