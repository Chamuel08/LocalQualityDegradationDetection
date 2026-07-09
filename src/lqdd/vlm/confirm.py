from __future__ import annotations

import base64
import time

import cv2
import numpy as np

from lqdd.agent.prompts import VLM_CONFIRM_PROMPT
from lqdd.config.loader import AgentConfig, VLMConfig
from lqdd.models.agent import AgentContext, VlmPendingItem, VLMResult
from lqdd.models.report import DegradationItem, TraceEntry
from lqdd.vlm.client import MockVLMClient, OllamaVLMClient, VLMClient, parse_vlm_response


def _crop_roi(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    x, y, w, h = bbox
    h_img, w_img = frame.shape[:2]
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)
    return frame[max(0, y) : y2, max(0, x) : x2]


def _to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_vlm_client(vlm_cfg: VLMConfig, mock: VLMClient | None = None) -> VLMClient:
    if mock is not None:
        return mock
    if vlm_cfg.provider == "ollama":
        return OllamaVLMClient(vlm_cfg)
    return MockVLMClient()


def collect_pending_vlm(
    degradations: list[DegradationItem],
    agent_cfg: AgentConfig,
) -> list[VlmPendingItem]:
    pending: list[VlmPendingItem] = []
    for d in degradations:
        if agent_cfg.grey_zone_lower <= d.confidence < agent_cfg.grey_zone_upper:
            pending.append(
                VlmPendingItem(
                    degradation_id=d.degradation_id,
                    detector=d.detector,
                    bbox=tuple(d.bbox),
                    preliminary_confidence=d.confidence,
                    region_type=d.region_type,
                )
            )
    return pending


def run_vlm_confirm(
    ctx: AgentContext,
    client: VLMClient,
    agent_cfg: AgentConfig,
    vlm_cfg: VLMConfig,
) -> list[VLMResult]:
    results: list[VLMResult] = []
    frame = ctx.frame_input.frame if ctx.frame_input else None
    if frame is None:
        return results

    ctx.pending_vlm = collect_pending_vlm(ctx.preliminary_degradations, agent_cfg)
    t_total = time.perf_counter()

    for item in ctx.pending_vlm:
        if ctx.vlm_calls_count >= vlm_cfg.max_calls_per_frame:
            ctx.traces.append(
                TraceEntry(
                    stage="vlm_confirm",
                    module="VLMConfirm",
                    timestamp_ms=0.0,
                    duration_ms=0.0,
                    input_summary={"degradation_id": item.degradation_id},
                    output_summary={},
                    decision="vlm_skipped: quota_exceeded",
                    mode="fast",
                )
            )
            continue

        deg = next((d for d in ctx.preliminary_degradations if d.degradation_id == item.degradation_id), None)
        if not deg:
            continue

        roi = _crop_roi(frame, list(item.bbox))
        b64 = _to_b64(roi)
        prompt = VLM_CONFIRM_PROMPT.format(
            region_type=item.region_type,
            preliminary_result=deg.degradation_type,
            confidence=f"{item.preliminary_confidence:.2f}",
            detector_judgment=deg.description,
            degradation_types=deg.degradation_type,
        )

        t0 = time.perf_counter()
        raw = client.confirm(prompt, b64)
        latency = (time.perf_counter() - t0) * 1000.0
        ctx.vlm_calls_count += 1

        if raw is None:
            if item.preliminary_confidence >= agent_cfg.hard_decision_threshold:
                ctx.traces.append(
                    TraceEntry(
                        stage="vlm_confirm",
                        module="VLMConfirm",
                        timestamp_ms=0.0,
                        duration_ms=latency,
                        input_summary={"degradation_id": item.degradation_id},
                        output_summary={},
                        decision="vlm_skipped: service_unavailable",
                        mode="fast",
                    )
                )
            continue

        result = parse_vlm_response(raw, item.region_type, deg.degradation_type, latency)
        results.append(result)
        ctx.traces.append(
            TraceEntry(
                stage="vlm_confirm",
                module="VLMConfirm",
                timestamp_ms=0.0,
                duration_ms=latency,
                input_summary={"degradation_id": item.degradation_id, "detector": item.detector},
                output_summary={"is_degraded": result.is_degraded, "vlm_confidence": result.vlm_confidence},
                decision="vlm_confirmed",
                mode="fast",
            )
        )

    ctx.vlm_ms = (time.perf_counter() - t_total) * 1000.0
    ctx.vlm_results = results
    return results
