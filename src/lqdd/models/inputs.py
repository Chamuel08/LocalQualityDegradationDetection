from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BBox = tuple[int, int, int, int]


@dataclass
class SourceInfo:
    video_path: str | None = None
    timestamp_ms: float | None = None
    resolution: tuple[int, int] = (0, 0)
    codec: str | None = None
    generator: str | None = None


@dataclass
class BadcaseMetadata:
    frame_id: str = ""
    resolution: tuple[int, int] | None = None
    bitrate_kbps: int | None = None
    codec: str | None = None
    has_overlay: bool = False
    overlay_types: list[str] | None = None


@dataclass
class SingleFrameInput:
    frame: np.ndarray
    frame_id: str
    mode: str = "fast"
    source_info: SourceInfo | None = None
    metadata: BadcaseMetadata | None = None
    ignore_regions: list[BBox] | None = None


@dataclass
class RegionNomination:
    region_type: int
    bbox: BBox
    mask: np.ndarray
    anomaly_score: float
    confidence: float
    suggested_detectors: list[str]
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class GlobalScanOutput:
    frame_index: int
    segmentation_map: np.ndarray
    global_quality_score: float
    is_fast_pass: bool
    is_fast_reject: bool
    nominations: list[RegionNomination]
    scan_duration_ms: float
    overlay_mask: np.ndarray | None = None
    foreground_mask: np.ndarray | None = None
    edge_mask: np.ndarray | None = None
