"""Parse `benchmark_pipeline.py` stdout into a dict of stages and metadata."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional


HEADER = re.compile(
    r"(?P<architecture>\S+), (?P<n_train>\d+) train \+ (?P<n_val>\d+) validation "
    r"structures, batch_size=(?P<batch_size>\d+), "
    r"num_workers=(?P<num_workers>\d+), "
    r"device=(?P<device>\S+), epochs=(?P<epochs>\d+), "
    r"(?P<wall_s>[\d.]+) s wall"
)
MEMORY = re.compile(
    r"memory (?P<peak_gb>[\d.]+) GB peak, (?P<added_gb>[\d.]+) GB added "
    r"over a (?P<baseline_gb>[\d.]+) GB baseline"
)
BEST_METRIC = re.compile(
    r"best_val_metric (?P<best_val_metric>[\d.eE+-]+) "
    r"\((?P<best_model_metric>\S+)\) at epoch (?P<best_epoch>\d+)"
)
EPOCH1_METRICS = re.compile(
    r"epoch1_metrics train_loss=(?P<epoch1_train_loss>[\d.eE+-]+) "
    r"val_loss=(?P<epoch1_val_loss>[\d.eE+-]+)"
)
STAGE = re.compile(
    r"^(?P<stage>\S+)\s+(?P<calls>\d+)\s+(?P<ms_per_call>[\d.]+)\s+"
    r"(?P<pct>[\d.]+)%\s*$"
)
THROUGHPUT = re.compile(r"^(?P<name>atoms|structures)/s\s+(?P<value>[\d.]+)\s*$")
TRANSPORT = re.compile(
    r"^(?P<name>CollateFn \+ unpack_batch|collate_batch \(no blob\))"
    r"\s+(?P<ms>[\d.]+) ms\s*$"
)


def parse_report(text: str) -> Dict[str, Any]:
    """Pull header, stages, throughput and transport rows out of stdout."""
    header: Dict[str, Any] = {}
    match = HEADER.search(text)
    if match:
        header = {
            "architecture": match["architecture"],
            "n_train": int(match["n_train"]),
            "n_val": int(match["n_val"]),
            "batch_size": int(match["batch_size"]),
            "num_workers": int(match["num_workers"]),
            "device": match["device"],
            "epochs": int(match["epochs"]),
            "wall_s": float(match["wall_s"]),
        }
    memory = MEMORY.search(text)
    if memory:
        header.update({k: float(memory[k]) for k in memory.groupdict()})

    best = BEST_METRIC.search(text)
    if best:
        header["best_val_metric"] = float(best["best_val_metric"])
        header["best_model_metric"] = best["best_model_metric"]
        header["best_epoch"] = int(best["best_epoch"])

    epoch1 = EPOCH1_METRICS.search(text)
    if epoch1:
        header["epoch1_train_loss"] = float(epoch1["epoch1_train_loss"])
        header["epoch1_val_loss"] = float(epoch1["epoch1_val_loss"])

    stages = {}
    throughput = {}
    transport = {}
    in_table = False
    for line in text.splitlines():
        if line.startswith("stage") and "ms/call" in line:
            in_table = True
            continue
        if in_table:
            row = STAGE.match(line)
            if row:
                stages[row["stage"]] = {
                    "calls": int(row["calls"]),
                    "ms_per_call": float(row["ms_per_call"]),
                    "pct": float(row["pct"]),
                }
                continue
            rate = THROUGHPUT.match(line)
            if rate:
                throughput[rate["name"] + "/s"] = float(rate["value"])
                continue
            if line.strip() == "" or line.startswith("breakdown") or line.startswith(
                "inside"
            ) or line.startswith("other:"):
                continue
            in_table = False
        moved = TRANSPORT.match(line)
        if moved:
            transport[moved["name"]] = float(moved["ms"])

    return {
        "header": header,
        "stages": stages,
        "throughput": throughput,
        "transport": transport,
    }


def stage_ms(parsed: Dict[str, Any], name: str) -> Optional[float]:
    stage = parsed.get("stages", {}).get(name)
    return None if stage is None else stage["ms_per_call"]


_SAMPLE = """
PET, 51 train + 13 validation structures, batch_size=8, num_workers=0, device=cpu, epochs=2, 12.3 s wall (incl. validation), memory 1.54 GB peak, 0.42 GB added over a 1.12 GB baseline

best_val_metric 0.001200 (mae_prod) at epoch 1
epoch1_metrics train_loss=9.691435 val_loss=2.348874

stage               calls   ms/call  % of step
loader                 14     30.10      17.2%
step                   14    145.00      82.8%

breakdown of `step`:
unpack                 14      8.11       4.6%
h2d                    14      3.20       1.8%

inside the `loader` wait (only recorded with num_workers=0):
group_and_join         18      1.60       0.9%
transforms             18     12.50       7.1%
transforms/neighbor_lists     18      5.90       3.4%
serialize              18     10.20       5.8%

atoms/s                         11781.0
structures/s                       54.5

transport of a 8-structure batch, no transformations:
CollateFn + unpack_batch        2.90 ms
collate_batch (no blob)         0.40 ms
"""


if __name__ == "__main__":
    parsed = parse_report(_SAMPLE)
    assert parsed["header"]["n_train"] == 51
    assert parsed["header"]["peak_gb"] == 1.54
    assert parsed["header"]["best_val_metric"] == 0.0012
    assert parsed["header"]["best_model_metric"] == "mae_prod"
    assert parsed["header"]["best_epoch"] == 1
    assert parsed["header"]["epoch1_train_loss"] == 9.691435
    assert parsed["header"]["epoch1_val_loss"] == 2.348874
    assert parsed["stages"]["unpack"]["ms_per_call"] == 8.11
    assert parsed["stages"]["transforms/neighbor_lists"]["ms_per_call"] == 5.90
    assert parsed["throughput"]["atoms/s"] == 11781.0
    assert parsed["transport"]["collate_batch (no blob)"] == 0.40
    print("ok")
