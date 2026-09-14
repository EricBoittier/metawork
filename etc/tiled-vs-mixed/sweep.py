"""Drive bench.py trials in subprocesses: double until OOM, then bisect."""

import os
import sys

import argparse
import json
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))

OOM_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "Out of memory",
    "out of memory",
    "OOM when allocating",
)


def classify(text):
    if any(m in text for m in OOM_MARKERS):
        return "oom"
    return "error"


def run_trial(backend, batch_size, args):
    cmd = [
        sys.executable,
        os.path.join(HERE, "bench.py"),
        "trial",
        "--backend",
        backend,
        "--batch-size",
        str(batch_size),
        "--cache",
        args.cache,
        "--steps",
        str(args.steps),
        "--warmup",
        str(args.warmup),
        "--config",
        args.config,
    ]
    if args.sr_only:
        cmd.append("--sr-only")
    if args.compilation_cache:
        cmd += ["--compilation-cache", args.compilation_cache]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    wall = time.perf_counter() - t0
    blob = proc.stdout + proc.stderr

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            out = json.loads(line[len("RESULT ") :])
            out["trial_wall_s"] = wall
            print(
                f"  {backend if not args.sr_only else 'sr':>5} S={batch_size:<5} ok  "
                f"step {out['step_s_mean'] * 1e3:8.1f} ms  "
                f"{out['s_per_sample'] * 1e3:7.3f} ms/sample  "
                f"peak {out['peak_gb']:5.1f} GB  atoms {out['real_atoms']}",
                flush=True,
            )
            return out

    kind = classify(blob)
    tail = "\n".join(blob.strip().splitlines()[-4:])
    print(
        f"  {backend if not args.sr_only else 'sr':>5} S={batch_size:<5} "
        f"{kind.upper()}  ({wall:.0f}s)",
        flush=True,
    )
    if kind == "error":
        print("    " + tail.replace("\n", "\n    "), flush=True)
    return {
        "backend": backend,
        "batch_size": batch_size,
        "ok": False,
        "failure": kind,
        "returncode": proc.returncode,
        "tail": tail,
        "trial_wall_s": wall,
    }


def sweep_backend(backend, args, results):
    ok, fail = {}, {}
    size = args.start

    while size <= args.pool:
        r = run_trial(backend, size, args)
        results.append(r)
        save(args, results)
        if r["ok"]:
            ok[size] = r
            size *= 2
        else:
            fail[size] = r
            break

    if not ok:
        return
    lo = max(ok)
    hi = min(fail) if fail else None
    if hi is None:
        return

    for _ in range(args.bisect):
        if hi - lo <= max(1, lo // 8):
            break
        mid = (lo + hi) // 2
        r = run_trial(backend, mid, args)
        results.append(r)
        save(args, results)
        if r["ok"]:
            lo = mid
        else:
            hi = mid


def save(args, results):
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="mad")
    p.add_argument("--cache", default=os.environ["SCRATCH"] + "/tvm_pool.pkl")
    p.add_argument("--out", default=os.environ["HOME"] + "/tiled_vs_mixed_results.json")
    p.add_argument("--pool", type=int, default=1024)
    p.add_argument("--start", type=int, default=4)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--bisect", type=int, default=4)
    p.add_argument("--compilation-cache", default="")
    p.add_argument("--backends", default="tiled,mixed")
    p.add_argument("--sizes", default="")
    p.add_argument("--sr-only", action="store_true")
    p.add_argument("--append", action="store_true")
    args = p.parse_args()

    results = []
    if args.append and os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)
        print(f"appending to {len(results)} existing records", flush=True)

    for backend in args.backends.split(","):
        print(f"== {backend}{' (sr-only)' if args.sr_only else ''} ==", flush=True)
        if args.sizes:
            for size in [int(x) for x in args.sizes.split(",")]:
                r = run_trial(backend, size, args)
                results.append(r)
                save(args, results)
        else:
            sweep_backend(backend, args, results)
    save(args, results)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
