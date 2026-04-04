"""
Sweep a single env var over a list of values, running train_hybrid.py for each.

Usage:
    python sweep.py VAR_NAME val1 val2 val3 ...
"""

import os
import subprocess
import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: python sweep.py VAR_NAME val1 val2 ...", file=sys.stderr)
        sys.exit(1)

    var_name = sys.argv[1]
    values = sys.argv[2:]

    for value in values:
        env = {**os.environ, var_name: value}
        print(f"\n=== {var_name}={value} ===\n", flush=True)
        result = subprocess.run(
            [sys.executable, "train_hybrid.py"],
            env=env,
        )
        if result.returncode != 0:
            print(
                f"train_hybrid.py exited with code {result.returncode} for {var_name}={value}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
