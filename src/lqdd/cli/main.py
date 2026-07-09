from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import jsonschema
import numpy as np

from lqdd.config.loader import load_config
from lqdd.models.inputs import BadcaseMetadata, SingleFrameInput, SourceInfo
from lqdd.models.report import report_to_dict
from lqdd.pipeline.agent_pipeline import AgentPipeline
from lqdd.pipeline.fast_pipeline import FastPipeline
from lqdd.report.html_renderer import render_html_report

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_V01 = REPO_ROOT / "specs" / "001-v0-fast-mvp" / "contracts" / "quality-report.schema.json"
SCHEMA_V1 = REPO_ROOT / "specs" / "002-v1-agent-layer" / "contracts" / "quality-report.v1.schema.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def _load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(f"cannot read image: {path}")
    return img


def _load_metadata(path: Path) -> tuple[BadcaseMetadata | None, SourceInfo | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    meta = BadcaseMetadata(
        frame_id=raw.get("frame_id", ""),
        resolution=tuple(raw["resolution"]) if raw.get("resolution") else None,
        bitrate_kbps=raw.get("bitrate_kbps"),
        codec=raw.get("codec"),
        has_overlay=bool(raw.get("has_overlay", False)),
        overlay_types=raw.get("overlay_types"),
    )
    source = SourceInfo(
        resolution=tuple(raw["resolution"]) if raw.get("resolution") else (0, 0),
        codec=raw.get("codec"),
    )
    return meta, source


def _load_ignore_regions(path: Path) -> list[tuple[int, int, int, int]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    regions = raw.get("regions", [])
    return [tuple(int(v) for v in r) for r in regions]


def _validate_report(data: dict, use_v1: bool = False) -> None:
    schema_path = SCHEMA_V1 if use_v1 else SCHEMA_V01
    if not schema_path.is_file():
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.validate(instance=data, schema=schema)


def _output_format(output: Path | None, output_dir: Path | None, batch: bool) -> str:
    if batch and output_dir and output and output.suffix.lower() == ".html":
        return "html"
    if output and output.suffix.lower() == ".html":
        return "html"
    return "json"


def _write_report(
    report_data: dict,
    report_obj,
    frame_bgr: np.ndarray,
    stem: str,
    fmt: str,
    output: Path | None,
    output_dir: Path | None,
    batch: bool,
) -> None:
    if batch and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = ".html" if fmt == "html" else ".json"
        out_path = output_dir / f"{stem}{ext}"
    elif output and str(output) != "-":
        out_path = output
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        print(json.dumps(report_data, ensure_ascii=False, indent=2))
        return

    if fmt == "html":
        render_html_report(report_obj, out_path, frame_bgr)
    else:
        out_path.write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detect.py", description="Local Quality Degradation Detection v0.1")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--image", type=Path, help="Single image file")
    group.add_argument("--image-dir", type=Path, help="Directory of images (non-recursive)")
    parser.add_argument("--mode", default="fast", help="Detection mode (v0.1: fast only)")
    parser.add_argument("--output", type=Path, default=Path("-"), help="Output file (- for stdout JSON)")
    parser.add_argument("--output-dir", type=Path, help="Batch output directory")
    parser.add_argument("--metadata", type=Path, help="Metadata JSON sidecar")
    parser.add_argument("--ignore-regions", type=Path, help="Ignore regions JSON sidecar")
    parser.add_argument("--config", type=Path, help="YAML config path")
    parser.add_argument("--frame-id", type=str, help="Override frame_id for single image")
    parser.add_argument("--legacy-fixed", action="store_true", help="Use v0.1 fixed pipeline (no Agent)")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.image and not args.image_dir:
        print("error: specify --image or --image-dir", file=sys.stderr)
        return 2
    if args.mode == "deep":
        print("error: deep mode not implemented in 002; use --mode fast", file=sys.stderr)
        return 2
    if args.mode != "fast":
        print(f"error: unsupported mode for v0.1: {args.mode}", file=sys.stderr)
        return 2

    config = load_config(args.config)
    use_agent = config.agent.enabled and not args.legacy_fixed
    if use_agent:
        pipeline = AgentPipeline(config)
    else:
        config.report.system_version = "0.1.0"
        pipeline = FastPipeline(config)

    metadata: BadcaseMetadata | None = None
    source_info: SourceInfo | None = None
    if args.metadata:
        if not args.metadata.is_file():
            print(f"error: image not found: {args.metadata}", file=sys.stderr)
            return 1
        try:
            metadata, source_info = _load_metadata(args.metadata)
        except ValueError as exc:
            print(f"error: failed to parse {args.metadata}: {exc}", file=sys.stderr)
            return 1

    ignore_regions: list[tuple[int, int, int, int]] | None = None
    if args.ignore_regions:
        if not args.ignore_regions.is_file():
            print(f"error: image not found: {args.ignore_regions}", file=sys.stderr)
            return 1
        try:
            ignore_regions = _load_ignore_regions(args.ignore_regions)
        except ValueError as exc:
            print(f"error: failed to parse {args.ignore_regions}: {exc}", file=sys.stderr)
            return 1

    batch = args.image_dir is not None
    fmt = _output_format(args.output, args.output_dir, batch)

    if args.image:
        if not args.image.is_file():
            print(f"error: image not found: {args.image}", file=sys.stderr)
            return 1
        images = [args.image]
    else:
        if not args.image_dir.is_dir():
            print(f"error: image not found: {args.image_dir}", file=sys.stderr)
            return 1
        images = _collect_images(args.image_dir)
        if not images:
            print(f"error: image not found: {args.image_dir}", file=sys.stderr)
            return 1

    successes = 0
    failures = 0
    for image_path in images:
        try:
            frame = _load_image(image_path)
            frame_id = args.frame_id or (metadata.frame_id if metadata and metadata.frame_id else image_path.stem)
            frame_input = SingleFrameInput(
                frame=frame,
                frame_id=frame_id,
                mode="fast",
                source_info=source_info,
                metadata=metadata,
                ignore_regions=ignore_regions,
            )
            report = pipeline.run(frame_input)
            data = report_to_dict(report)
            _validate_report(data, use_v1=use_agent)
            _write_report(
                data,
                report,
                frame,
                image_path.stem,
                fmt,
                args.output,
                args.output_dir,
                batch,
            )
            successes += 1
            if args.verbose:
                print(f"ok: {image_path}", file=sys.stderr)
        except OSError as exc:
            failures += 1
            print(f"warning: skipping {image_path}: {exc}", file=sys.stderr)
        except Exception as exc:
            failures += 1
            print(f"warning: skipping {image_path}: {exc}", file=sys.stderr)

    if successes == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
