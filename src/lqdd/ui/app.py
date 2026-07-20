"""LQDD 图形界面入口（Gradio + pywebview）。

复用现有 pipeline 与 mask 叠加渲染，不改动任何单帧/视频内部接口。

启动：
    lqdd-gui                 # pywebview 原生窗口（默认）
    lqdd-gui --browser       # 浏览器回退
    python -m lqdd.ui.app    # 等价

frozen 模式（PyInstaller 打包后）下从 bundle 内解析 config.example.yaml。
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _bundled_config_path() -> Path | None:
    """frozen 模式下返回 bundle 内的 config.example.yaml 路径。"""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(".")))
        candidate = base / "config.example.yaml"
        if candidate.is_file():
            return candidate
    return None


def _resolve_config(config_path: str | None) -> "Any":
    from lqdd.config.loader import load_config

    p = (config_path or "").strip()
    if p and Path(p).is_file():
        return load_config(Path(p))
    # 默认：让 load_config 找 cwd 的 config.yaml / config.example.yaml
    cfg = load_config()
    # frozen 兜底：若 cwd 没有任何配置，用 bundle 内的
    if getattr(sys, "frozen", False) and not Path("config.yaml").is_file() and not Path("config.example.yaml").is_file():
        bundled = _bundled_config_path()
        if bundled:
            return load_config(bundled)
    return cfg


def _build_pipeline(config: Any, use_agent: bool):
    """按模式构造 pipeline。use_agent=True 用 AgentPipeline，否则 FastPipeline。"""
    if use_agent:
        from lqdd.pipeline.agent_pipeline import AgentPipeline

        return AgentPipeline(config)
    config.report.system_version = "0.1.0"
    from lqdd.pipeline.fast_pipeline import FastPipeline

    return FastPipeline(config)


def _bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _degradations_to_rows(report: Any) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for d in report.degradations:
        x, y, bw, bh = d.bbox
        rows.append(
            [
                d.degradation_type,
                d.severity,
                round(float(d.confidence), 3),
                f"{x},{y},{bw},{bh}",
                d.detector,
                d.evidence.metric,
                d.evidence.value,
                d.evidence.threshold,
                d.evidence.detail,
            ]
        )
    return rows


def _agent_steps_to_rows(agent_meta: dict[str, Any] | None) -> list[list[Any]]:
    if not agent_meta:
        return []
    rows: list[list[Any]] = []
    for s in agent_meta.get("agent_steps", []) or []:
        rows.append(
            [
                s.get("step"),
                s.get("action"),
                (s.get("thought") or "")[:120],
                (s.get("observation") or "")[:120],
                s.get("latency_ms"),
            ]
        )
    return rows


def _vlm_discover_md(agent_meta: dict[str, Any] | None) -> str:
    if not agent_meta:
        return ""
    findings = agent_meta.get("vlm_discover_findings") or []
    if not findings:
        return ""
    lines = ["### vlm_discover 主动发现"]
    for f in findings:
        lines.append(
            f"- **{f.get('degradation_type')}** ({f.get('severity')}, conf={f.get('confidence')}): "
            f"{f.get('region_description')} — {f.get('reasoning')}"
        )
    return "\n".join(lines)


def _summary_md(report: Any) -> str:
    lines = [
        f"### MOS: **{report.overall_mos:.3f}** &nbsp; 严重度: **{report.severity}**",
        f"system_version: `{report.system_version}` &nbsp; 劣化数: **{len(report.degradations)}**",
    ]
    perf = report.performance
    lines.append(
        f"耗时: total={perf.total_ms:.0f}ms (scan={perf.global_scan_ms:.0f}, "
        f"detect={perf.detection_ms:.0f}, vlm={perf.vlm_ms:.0f}, judge={perf.judge_ms:.0f})"
    )
    am = report.agent_meta
    if am:
        lines.append(
            f"Agent: rounds={am.get('rounds_executed')} vlm_calls={am.get('vlm_calls')} "
            f"agent_driven_vlm={am.get('agent_driven_vlm')}"
        )
    return "\n".join(lines)


def run_single(image_path: str, mode: str, config_path: str) -> tuple[Any, str, list[list[Any]], list[list[Any]], str, Any]:
    """单帧检测回调。返回 (overlay_rgb, summary_md, deg_rows, agent_rows, vlm_md, full_json)。"""
    from lqdd.models.inputs import SingleFrameInput
    from lqdd.models.report import report_to_dict
    from lqdd.report.viz_variants import render_mask_overlay

    if not image_path:
        raise ValueError("请先上传一张图像")

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise OSError(f"无法读取图像: {image_path}")

    config = _resolve_config(config_path)
    use_agent = mode.startswith("V1") and config.agent.enabled
    pipeline = _build_pipeline(config, use_agent)

    fi = SingleFrameInput(frame=frame, frame_id=Path(image_path).stem, mode="fast")
    report = pipeline.run(fi)

    overlay = render_mask_overlay(frame, report, style="contour_fill")
    overlay_rgb = _bgr_to_rgb(overlay)

    summary = _summary_md(report)
    deg_rows = _degradations_to_rows(report)
    agent_rows = _agent_steps_to_rows(report.agent_meta)
    vlm_md = _vlm_discover_md(report.agent_meta)
    full_json = report_to_dict(report)
    return overlay_rgb, summary, deg_rows, agent_rows, vlm_md, full_json


def run_video(video_path: str, mode: str, config_path: str, max_frames: int) -> tuple[str, list[list[Any]], Any]:
    """视频多帧回调。返回 (summary_md, summary_rows, full_json)。"""
    from lqdd.models.report import report_to_dict
    from lqdd.pipeline.video_clip_runner import VideoClipRunner, sample_frames_from_video

    if not video_path:
        raise ValueError("请先上传一段视频")

    config = _resolve_config(config_path)
    use_agent = mode.startswith("V1") and config.agent.enabled
    pipeline = _build_pipeline(config, use_agent)
    runner = VideoClipRunner(pipeline)

    frames = sample_frames_from_video(video_path, max_frames=max_frames)
    result = runner.run(frames, clip_id=Path(video_path).stem)

    fr = result.flicker_result
    summary = (
        f"### 视频聚合结果\n"
        f"- 帧数: **{result.frame_count}**\n"
        f"- aggregate_mos: **{result.aggregate_mos:.3f}**\n"
        f"- 最差帧: MOS={result.worst_frame_mos:.3f} @ idx={result.worst_frame_index}\n"
        f"- 闪烁: is_flickering=**{fr.is_flickering}** ratio={fr.flicker_ratio:.3f} "
        f"segments={len(fr.flicker_segments)} max_luma_delta={fr.max_luma_delta:.1f}\n"
    )
    summary_rows = [[k, v] for k, v in result.degradation_summary.items()]
    full_json = {
        "clip_id": result.clip_id,
        "frame_count": result.frame_count,
        "aggregate_mos": result.aggregate_mos,
        "worst_frame_mos": result.worst_frame_mos,
        "worst_frame_index": result.worst_frame_index,
        "flicker": {
            "is_flickering": fr.is_flickering,
            "flicker_ratio": fr.flicker_ratio,
            "flicker_segments": [
                {
                    "start_frame": s.start_frame,
                    "end_frame": s.end_frame,
                    "max_delta": s.max_delta,
                    "metric": s.metric,
                    "severity": s.severity,
                }
                for s in fr.flicker_segments
            ],
        },
        "degradation_summary": result.degradation_summary,
        "frame_reports": [report_to_dict(r) for r in result.frame_reports],
    }
    return summary, summary_rows, full_json


def build_ui():
    """构造 Gradio Blocks 界面。延迟 import gradio 以保持核心包零侵入。"""
    import gradio as gr

    deg_headers = ["类型", "严重度", "置信度", "bbox", "检测器", "metric", "value", "threshold", "detail"]
    agent_headers = ["step", "action", "thought", "observation", "latency_ms"]
    summary_headers = ["degradation_type", "count"]

    with gr.Blocks(title="LQDD 画质劣化检测") as demo:
        gr.Markdown("# 局部画质劣化检测 (LQDD)\n无参考、可解释的局部画质劣化检测 — ReAct Agent + VLM")

        with gr.Row():
            # ── 左侧输入区 ──
            with gr.Column(scale=1):
                with gr.Tab("单帧图像"):
                    single_img = gr.Image(type="filepath", label="输入图像")
                    single_btn = gr.Button("运行单帧检测", variant="primary")
                with gr.Tab("视频 clip"):
                    video_in = gr.Video(label="输入视频")
                    max_frames = gr.Slider(2, 16, value=8, step=1, label="抽帧数")
                    video_btn = gr.Button("运行视频检测", variant="primary")
                mode = gr.Radio(
                    ["V1 ReAct Agent", "v0.1 基线"],
                    value="V1 ReAct Agent",
                    label="流水线模式",
                )
                config_path = gr.Textbox(value="config.yaml", label="配置文件路径（可选）")
                gr.Markdown(
                    "> V1 模式需 Ollama 已运行并拉取 `qwen2.5vl:7b` / `qwen2.5:1.5b`；"
                    "不可用时 Agent 自动规则降级。"
                )

            # ── 右侧结果区 ──
            with gr.Column(scale=2):
                overlay_out = gr.Image(label="mask 叠加预览")
                summary_out = gr.Markdown(label="总览")
                with gr.Accordion("劣化列表", open=True):
                    deg_out = gr.Dataframe(headers=deg_headers, datatype=["str"] * len(deg_headers), wrap=True)
                with gr.Accordion("Agent 决策轨迹", open=False):
                    agent_out = gr.Dataframe(headers=agent_headers, datatype=["str", "str", "str", "str", "number"], wrap=True)
                vlm_out = gr.Markdown(label="vlm_discover")
                with gr.Accordion("视频聚合", open=False, visible=False) as video_acc:
                    video_summary = gr.Markdown()
                    video_deg = gr.Dataframe(headers=summary_headers, datatype=["str", "number"], wrap=True)
                with gr.Accordion("完整 JSON", open=False):
                    json_out = gr.JSON()

        single_btn.click(
            run_single,
            inputs=[single_img, mode, config_path],
            outputs=[overlay_out, summary_out, deg_out, agent_out, vlm_out, json_out],
        )
        video_btn.click(
            run_video,
            inputs=[video_in, mode, config_path, max_frames],
            outputs=[video_summary, video_deg, json_out],
        ).then(lambda: gr.Accordion(visible=True), outputs=video_acc)
    return demo


def _wait_for_server(port: int, timeout_s: float = 30.0) -> bool:
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="lqdd-gui", description="LQDD 图形界面")
    parser.add_argument("--browser", action="store_true", help="用浏览器而非 pywebview 原生窗口")
    parser.add_argument("--port", type=int, default=7860, help="本地 server 端口")
    parser.add_argument("--share", action="store_true", help="Gradio 公网分享链接")
    args = parser.parse_args()

    demo = build_ui()

    import gradio as gr

    theme = gr.themes.Soft()

    use_webview = not args.browser
    if use_webview:
        try:
            import webview  # noqa: F401
        except ImportError:
            use_webview = False

    if use_webview:
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            prevent_thread_lock=True,
            show_api=False,
            share=False,
            theme=theme,
        )
        if not _wait_for_server(args.port):
            print(f"[lqdd-gui] server 未在端口 {args.port} 启动，回退浏览器", file=sys.stderr)
            demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share, show_api=False, theme=theme)
            return 0

        import webview

        webview.create_window(
            "LQDD 画质劣化检测",
            f"http://127.0.0.1:{args.port}",
            width=1280,
            height=860,
        )
        webview.start()
        try:
            demo.close()
        except Exception:
            pass
    else:
        demo.launch(
            server_name="127.0.0.1",
            server_port=args.port,
            share=args.share,
            show_api=False,
            theme=theme,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
