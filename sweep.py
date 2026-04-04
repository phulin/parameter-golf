"""
Sweep a single env var over a list of values, running train_hybrid.py for each.

Usage:
    python sweep.py VAR_NAME val1 val2 val3 ...
"""

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path


def parse_metrics(line: str) -> dict[str, float]:
    """Extract named float metrics from a log line (e.g. 'train_loss:1.2345')."""
    return {k: float(v) for k, v in re.findall(r"(\w+):(\d+\.\d+)", line)}


def run_one(var_name: str, value: str) -> dict[str, float]:
    """Run train_hybrid.py with the given env var override, stream output, return best metrics."""
    env = {**os.environ, var_name: value}
    best: dict[str, float] = {}

    proc = subprocess.Popen(
        [sys.executable, "train_hybrid.py"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    stderr = proc.stderr

    def drain_stderr():
        for line in stderr:
            print(line, end="", file=sys.stderr, flush=True)

    stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
    stderr_thread.start()

    for line in proc.stdout:
        print(line, end="", flush=True)
        metrics = parse_metrics(line)
        for key in ("train_loss", "val_loss", "val_bpb"):
            if key in metrics:
                if key not in best or metrics[key] < best[key]:
                    best[key] = metrics[key]

    stderr_thread.join()
    proc.wait()
    if proc.returncode != 0:
        print(
            f"train_hybrid.py exited with code {proc.returncode} for {var_name}={value}",
            file=sys.stderr,
        )
    return best


def main():
    if len(sys.argv) < 3:
        print("Usage: python sweep.py VAR_NAME val1 val2 ...", file=sys.stderr)
        sys.exit(1)

    var_name = sys.argv[1]
    values = sys.argv[2:]
    results: list[tuple[str, dict[str, float]]] = []

    for value in values:
        print(f"\n=== {var_name}={value} ===\n", flush=True)
        best = run_one(var_name, value)
        results.append((value, best))

    # Summary table
    metrics = ["train_loss", "val_loss", "val_bpb"]
    col = 14
    header = f"\n{'value':<{col}}" + "".join(f"{m:<{col}}" for m in metrics)
    print(header)
    print("-" * len(header.rstrip()))
    for value, best in results:
        row = f"{value:<{col}}"
        for m in metrics:
            row += f"{best[m]:<{col}.4f}" if m in best else f"{'—':<{col}}"
        print(row)

    # Best per metric
    print()
    best_per_metric = {}
    for m in metrics:
        candidates = [(v, b[m]) for v, b in results if m in b]
        if candidates:
            best_val, best_score = min(candidates, key=lambda x: x[1])
            best_per_metric[m] = {"value": best_val, "score": best_score}
            print(f"best {m}: {var_name}={best_val} ({best_score:.4f})")

    # Save to file
    n = 1
    while (out_path := Path(f"sweep-{n}-{var_name}.json")).exists():
        n += 1
    out_path.write_text(json.dumps({
        "var": var_name,
        "results": [{"value": v, "best": b} for v, b in results],
        "best_per_metric": best_per_metric,
    }, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
