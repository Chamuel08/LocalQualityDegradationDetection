from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from lqdd.models.report import QualityReport, report_to_dict
from lqdd.report.viz_variants import (
    MaskVizStyle,
    build_legend_html,
    render_mask_overlay,
)

DEFAULT_VIZ_STYLE: MaskVizStyle = "contour_fill"


def _mos_html(report: QualityReport) -> str:
    """MOS 展示：有分时显示数值，不可用时显示原因（不回退默认分）。"""
    if report.overall_mos is not None:
        return f"{report.overall_mos:.3f}"
    reason = report.mos_unavailable_reason or "MOS 不可用"
    return f"<em>不可用</em>（{html.escape(reason)}）"


def _encode_image_b64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_frame_preview(
    frame_bgr: np.ndarray,
    report: QualityReport,
    style: MaskVizStyle = DEFAULT_VIZ_STYLE,
) -> np.ndarray:
    if not report.degradations:
        return frame_bgr.copy()
    return render_mask_overlay(frame_bgr, report, style=style)


def render_html_report(
    report: QualityReport,
    output_path: Path,
    frame_bgr: np.ndarray | None = None,
    viz_style: MaskVizStyle = DEFAULT_VIZ_STYLE,
) -> Path:
    data = report_to_dict(report)
    payload = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    empty_msg = "未检出显著劣化"

    if report.degradations:
        body_rows = []
        for d in report.degradations:
            ev = d.evidence
            x, y, bw, bh = d.bbox
            mask_note = "有" if d.region_mask_rle else "—"
            body_rows.append(
                f"<tr><td>{html.escape(d.degradation_type)}</td>"
                f"<td>{html.escape(d.severity)}</td>"
                f"<td>{d.confidence:.2f}</td>"
                f"<td><code>{x},{y},{bw},{bh}</code></td>"
                f"<td>{mask_note}</td>"
                f"<td>{html.escape(ev.method)}</td>"
                f"<td>{html.escape(ev.metric)}</td>"
                f"<td>{ev.value}</td>"
                f"<td>{ev.threshold}</td>"
                f"<td>{html.escape(ev.detail)}</td></tr>"
            )
        rows = "\n".join(body_rows)
        deg_section = f"""
  <h2>劣化列表</h2>
  <table>
    <thead><tr>
      <th>类型</th><th>严重度</th><th>置信度</th><th>bbox</th><th>mask</th>
      <th>method</th><th>metric</th><th>value</th><th>threshold</th><th>detail</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>"""
    else:
        deg_section = f"<p><em>{empty_msg}</em></p>"

    img_section = ""
    legend_html = ""
    if frame_bgr is not None:
        overlay = _draw_frame_preview(frame_bgr, report, style=viz_style)
        b64 = _encode_image_b64(overlay)
        legend_html = build_legend_html(report.degradations)
        if b64:
            img_section = f"""
  <h2>帧预览（局部 mask）</h2>
  <p class="hint">不规则区域轮廓来自检测器输出的 <code>region_mask_rle</code>，非矩形外框。</p>
  <img alt="frame preview" src="data:image/png;base64,{b64}" style="max-width:100%;"/>
  {legend_html}"""

    content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <title>画质报告 {html.escape(report.report_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; font-size: 0.9rem; }}
    pre {{ background: #f6f8fa; padding: 1rem; overflow: auto; }}
    .hint {{ color: #666; font-size: 0.9rem; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 0.75rem; margin: 0.75rem 0 1rem; font-size: 0.9rem; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 0.35rem; }}
    .swatch {{ width: 14px; height: 14px; border-radius: 3px; border: 1px solid rgba(0,0,0,.15); }}
    .legend-empty {{ color: #888; font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>局部画质劣化检测报告</h1>
  <p><strong>MOS</strong>: {_mos_html(report)} &nbsp;
     <strong>严重度</strong>: {html.escape(report.severity)}</p>
  {img_section}
  {deg_section}
  <h2>完整 JSON</h2>
  <pre>{payload}</pre>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path
