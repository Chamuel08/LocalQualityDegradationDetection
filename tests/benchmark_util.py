from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = Path(os.environ.get("LQDD_DATA_DIR", Path.home() / "data"))


def benchmark_manifest_path() -> Path:
    env = os.environ.get("LQDD_BENCHMARK_MANIFEST")
    if env:
        return Path(env)
    manifest = DEFAULT_DATA / "synthetic_benchmark" / "manifest.json"
    if manifest.is_file():
        return manifest
    return DEFAULT_DATA / "synthetic_benchmark" / "manifest.json"


def benchmark_image_for_type(primary_type: str) -> Path | None:
    manifest = benchmark_manifest_path()
    if not manifest.is_file():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    samples = data.get("samples", data)
    root = manifest.parent
    for sample in samples:
        ptype = sample.get("primary_type")
        if not ptype and sample.get("artifacts"):
            ptype = sample["artifacts"][0]["type"]
        if ptype == primary_type or (primary_type == "clean" and sample.get("is_clean")):
            return (root / sample["image"]).resolve()
    return None
