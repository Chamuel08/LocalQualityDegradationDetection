#!/usr/bin/env python3
"""Generate GT benchmark dataset (synthetic_benchmark) via ~/data/degradation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(os.environ.get("LQDD_DATA_DIR", Path.home() / "data"))
DEFAULT_INPUT = Path(os.environ.get("LQDD_SOURCE_DIR", DEFAULT_DATA / "source_frames"))
DEFAULT_OUTPUT = DEFAULT_DATA / "synthetic_benchmark"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lqdd GT benchmark (synthetic_benchmark)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="clean source frames")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples-per-type", type=int, default=5)
    parser.add_argument("--clean-count", type=int, default=8)
    parser.add_argument("--use-lqdd-scan", action="store_true", default=True)
    parser.add_argument("--no-lqdd-scan", action="store_true")
    parser.add_argument("--attach-mos", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not (data_dir / "degradation" / "synthesize.py").is_file():
        print(f"error: degradation package not found under {data_dir}", file=sys.stderr)
        raise SystemExit(1)
    if not args.input.is_dir():
        print(f"error: source frames not found: {args.input}", file=sys.stderr)
        raise SystemExit(1)

    env = os.environ.copy()
    env["LQDD_REPO"] = str(REPO)
    use_scan = args.use_lqdd_scan and not args.no_lqdd_scan

    cmd = [
        sys.executable,
        "-m",
        "degradation.synthesize",
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--samples-per-type",
        str(args.samples_per_type),
        "--clean-count",
        str(args.clean_count),
    ]
    if use_scan:
        cmd.append("--use-lqdd-scan")
    if args.attach_mos:
        cmd.append("--attach-mos")

    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(data_dir), env=env)
    if rc != 0:
        raise SystemExit(rc)

    manifest = args.output / "manifest.json"
    print(f"\nBenchmark manifest: {manifest}")
    print("Evaluate with:")
    print(f"  python benchmark/run_eval.py --manifest {manifest} --output benchmark/runs/results.json")


if __name__ == "__main__":
    main()
