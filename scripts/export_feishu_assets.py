#!/usr/bin/env python3
"""Export PNG/HTML assets for Feishu portfolio (local only, gitignored)."""

from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lqdd.config.loader import load_config
from lqdd.models.inputs import SingleFrameInput
from lqdd.pipeline.agent_pipeline import AgentPipeline
from lqdd.pipeline.fast_pipeline import FastPipeline
from lqdd.report.html_renderer import render_html_report
from lqdd.report.viz_variants import render_mask_overlay
from lqdd.models.report import report_to_dict

OUT = REPO / "benchmark" / "runs" / "feishu_export"
DATA = Path("/Users/wendong_zhangwendong/data")

SCENARIO_A = [
    ("clean_source", DATA / "taolive_frames/343275048276_10001737.png", "干净源帧"),
    ("synth_000000", DATA / "synthetic_taolive/images/000000.png", "合成 compression_hf"),
    (
        "real_crf36",
        DATA / "compare/frames_cache/343275048276_10001737_h264_5s_CRF_36.png",
        "真实 CRF36",
    ),
]


def _load(path: Path) -> cv2.Mat:
    img = cv2.imread(str(path))
    if img is None:
        raise OSError(path)
    return img


def _save_overlay(frame, report, path: Path) -> None:
    cv2.imwrite(str(path), render_mask_overlay(frame, report, style="contour_fill"))


def _extract_b64_from_html(html_path: Path, png_path: Path) -> bool:
    text = html_path.read_text(encoding="utf-8")
    if "data:image/png;base64," not in text:
        return False
    b64 = text.split("data:image/png;base64,", 1)[1].split('"', 1)[0]
    png_path.write_bytes(base64.b64decode(b64))
    return True


def export_scenario_a(config) -> list[dict]:
    dest = OUT / "scenario_a"
    dest.mkdir(parents=True, exist_ok=True)
    pipe = FastPipeline(config)
    rows = []
    for key, img, title in SCENARIO_A:
        if not img.is_file():
            continue
        frame = _load(img)
        report = pipe.run(SingleFrameInput(frame=frame, frame_id=key, mode="fast"))
        shutil.copy2(img, dest / f"{key}_input.png")
        _save_overlay(frame, report, dest / f"{key}_overlay.png")
        html = dest / f"{key}.html"
        render_html_report(report, html, frame_bgr=frame)
        rows.append(
            {
                "key": key,
                "title": title,
                "mos": report.overall_mos,
                "count": len(report.degradations),
                "types": [d.degradation_type for d in report.degradations],
                "files": [f"{key}_input.png", f"{key}_overlay.png", f"{key}.html"],
            }
        )
    (dest / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def export_scenario_d(config) -> dict:
    dest = OUT / "scenario_d"
    dest.mkdir(parents=True, exist_ok=True)
    img = DATA / "synthetic_taolive/images/000000.png"
    frame = _load(img)
    fi = SingleFrameInput(frame=frame, frame_id="000000", mode="fast")

    legacy_pipe = FastPipeline(config)
    legacy_report = legacy_pipe.run(fi)
    agent_report = AgentPipeline(config).run(fi)

    render_html_report(legacy_report, dest / "legacy_v01.html", frame_bgr=frame)
    render_html_report(agent_report, dest / "agent_vlm.html", frame_bgr=frame)
    _save_overlay(frame, legacy_report, dest / "legacy_v01_overlay.png")
    _save_overlay(frame, agent_report, dest / "agent_vlm_overlay.png")

    leg = report_to_dict(legacy_report)
    ag = report_to_dict(agent_report)
    (dest / "legacy_v01.json").write_text(json.dumps(leg, ensure_ascii=False, indent=2), encoding="utf-8")
    (dest / "agent_vlm.json").write_text(json.dumps(ag, ensure_ascii=False, indent=2), encoding="utf-8")

    vlm_t = next((t for t in ag.get("decision_trace", []) if t["stage"] == "vlm_confirm"), {})
    judge_t = next((t for t in ag.get("decision_trace", []) if t["stage"] == "judge"), {})
    compare = {
        "legacy_confidence": leg["degradations"][0]["confidence"] if leg["degradations"] else None,
        "agent_confidence": ag["degradations"][0]["confidence"] if ag["degradations"] else None,
        "vlm_decision": vlm_t.get("decision"),
        "vlm_output": vlm_t.get("output_summary"),
        "vlm_reasoning": (ag["degradations"][0].get("vlm_reasoning") if ag["degradations"] else None),
        "judge_decision": judge_t.get("decision"),
        "judge_output": judge_t.get("output_summary"),
        "agent_meta": ag.get("agent_meta"),
    }
    (dest / "compare.json").write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")
    return compare


def copy_summaries() -> None:
    dest = OUT / "summaries"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ["scenario_b/summary.txt", "RUN_SUMMARY.md"]:
        src = REPO / "benchmark" / "runs" / name
        if src.is_file():
            shutil.copy2(src, dest / Path(name).name)


def main() -> None:
    config = load_config(REPO / "config.yaml" if (REPO / "config.yaml").is_file() else REPO / "config.example.yaml")
    OUT.mkdir(parents=True, exist_ok=True)
    print("scenario A ...")
    export_scenario_a(config)
    print("scenario D compare ...")
    cmp = export_scenario_d(config)
    print("vlm", cmp.get("vlm_decision"), "conf", cmp.get("legacy_confidence"), "->", cmp.get("agent_confidence"))
    copy_summaries()
    print(f"done: {OUT}")


if __name__ == "__main__":
    main()
