from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx

from lqdd.config.loader import VLMConfig
from lqdd.models.agent import VLMResult


class VLMClient(ABC):
    @abstractmethod
    def confirm(self, prompt: str, image_b64: str) -> dict[str, Any] | None:
        ...


class OllamaVLMClient(VLMClient):
    def __init__(self, config: VLMConfig) -> None:
        self.config = config

    def confirm(self, prompt: str, image_b64: str) -> dict[str, Any] | None:
        url = f"{self.config.host.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
            "format": "json",
            "stream": False,
        }
        try:
            with httpx.Client(timeout=self.config.timeout_ms / 1000.0, trust_env=False) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                return json.loads(content)
        except Exception:
            return None


class MockVLMClient(VLMClient):
    def __init__(self, fixture_path: Path | None = None) -> None:
        self.fixture_path = fixture_path
        self._default = {
            "is_degraded": True,
            "confidence": 0.82,
            "severity": "moderate",
            "reasoning": "ROI 可见绿色溢色，与抠像绿边一致",
            "mos_impact_estimate": -0.3,
            "root_cause": "matting_error",
            "ux_impact": "边缘不自然，影响主体融合观感",
        }

    def confirm(self, prompt: str, image_b64: str) -> dict[str, Any] | None:
        if self.fixture_path and self.fixture_path.is_file():
            return json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return dict(self._default)


def parse_vlm_response(raw: dict[str, Any], region_type: str, degradation_type: str, latency_ms: float) -> VLMResult:
    return VLMResult(
        region_type=region_type,
        degradation_type=degradation_type,
        is_degraded=bool(raw.get("is_degraded", False)),
        vlm_confidence=float(raw.get("confidence", 0.0)),
        vlm_severity=str(raw.get("severity", "minor")),
        reasoning=str(raw.get("reasoning", "")),
        mos_impact_estimate=float(raw.get("mos_impact_estimate", -0.2)),
        root_cause=str(raw.get("root_cause", "other")),
        ux_impact=str(raw.get("ux_impact", "")),
        vlm_latency_ms=latency_ms,
    )
