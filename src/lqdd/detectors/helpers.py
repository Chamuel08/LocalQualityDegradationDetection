from __future__ import annotations

import uuid

import numpy as np

from lqdd.models.enums import RegionType, RootCauseCategory
from lqdd.models.report import DegradationItem, Evidence, RootCauseHypothesis
from lqdd.report.mask_codec import encode_mask_rle


def make_degradation_item(
    *,
    detector: str,
    degradation_type: str,
    region_type: RegionType,
    severity: str,
    confidence: float,
    mos_impact: float,
    bbox: tuple[int, int, int, int],
    region_mask: np.ndarray | None,
    method: str,
    metric: str,
    value: float,
    threshold: float,
    detail: str,
    description: str,
    cause: RootCauseCategory,
    frame_index: int = 0,
) -> DegradationItem:
    mask_rle = encode_mask_rle(region_mask) if region_mask is not None and region_mask.any() else None
    return DegradationItem(
        degradation_id=f"deg_{uuid.uuid4().hex[:8]}",
        region_type=region_type.name.lower(),
        degradation_type=degradation_type,
        severity=severity,
        confidence=confidence,
        mos_impact=mos_impact,
        bbox=list(bbox),
        region_mask_rle=mask_rle,
        frame_indices=[frame_index],
        description=description,
        detector=detector,
        evidence=Evidence(
            method=method,
            metric=metric,
            value=round(value, 4),
            threshold=threshold,
            detail=detail,
        ),
        root_cause_hypothesis=RootCauseHypothesis(
            cause=cause.value,
            confidence=min(0.9, 0.45 + max(0.0, value - threshold) * 0.35),
        ),
    )
