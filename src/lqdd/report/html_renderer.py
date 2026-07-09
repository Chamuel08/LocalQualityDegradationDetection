from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import cv2
import numpy as np

from lqdd.models.report import QualityReport, report_to_dict


def _encode_image_b64(frame_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", frame_bgr)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _draw_bboxes(frame_bgr: np.ndarray, report: QualityReport) -> np.ndarray:
    vis = frame_bgr.copy()
    for d in report.degradations:
        x, y, w, h = d.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(
            vis,
            d.degradation_type,
            (x, max(0, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return vis


def render_html_report(report: QualityReport, output_path: Path, frame_bgr: np.ndarray | None = None) -> Path:
    data = report_to_dict(report)
    payload = html.escape(json.dumps(data, ensure_ascii=False, indent=2))
    empty_msg = "未检出显著劣化"

    if report.degradations:
        body_rows = []
        for d in report.degradations:
            ev = d.evidence
            body_rows.append(
                f"<tr><td>{html.escape(d.degradation_type)}</td>"
                f"<td>{html.escape(d.severity)}</td>"
                f"<td>{d.confidence:.2f}</td>"
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
      <th>类型</th><th>严重度</th><th>置信度</th>
      <th>method</th><th>metric</th><th>value</th><th>threshold</th><th>detail</th>
    </tr></thead>
    <tbody>{rows}</tbody>
  </table>"""
    else:
        deg_section = f"<p><em>{empty_msg}</em></p>"

    img_section = ""
    if frame_bgr is not None:
        overlay = _draw_bboxes(frame_bgr, report)
        b64 = _encode_image_b64(overlay)
        if b64:
            img_section = f"""
  <h2>帧预览（bbox 叠加）</h2>
  <img alt="frame preview" src="data:image/png;base64,{b64}" style="max-width:100%;"/>"""

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
  </style>
</head>
<body>
  <h1>局部画质劣化检测报告</h1>
  <p><strong>MOS</strong>: {report.overall_mos:.3f} &nbsp;
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
