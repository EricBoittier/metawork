"""Execute and export every lesson; keep a machine-readable completion record."""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "exports"
    output.mkdir(exist_ok=True)
    results = []
    for notebook in sorted((ROOT / "notebooks").glob("*.py")):
        destination = output / f"{notebook.stem}.html"
        log = output / f"{notebook.stem}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "marimo",
                    "export",
                    "html",
                    str(notebook),
                    "--include-code",
                    "--force",
                    "-o",
                    str(destination),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        passed = result.returncode == 0 and destination.is_file()
        results.append({"notebook": notebook.name, "passed": passed, "log": log.name})
        print(f"{'PASS' if passed else 'FAIL'} {notebook.name}", flush=True)
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "python": sys.version.split()[0],
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
    return 0 if results and all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
