"""Deprecated: use scripts/generate_benchmark_dataset.py for GT benchmark generation."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    script = ROOT / "scripts" / "generate_benchmark_dataset.py"
    print("note: generate_synthetic_samples.py now delegates to generate_benchmark_dataset.py")
    raise SystemExit(subprocess.call([sys.executable, str(script), *sys.argv[1:]]))


if __name__ == "__main__":
    main()
