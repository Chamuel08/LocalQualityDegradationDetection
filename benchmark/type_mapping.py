"""GT 劣化类型 ↔ lqdd 检测器 映射。"""

from __future__ import annotations

# manifest 中 artifacts[].type → lqdd detector 名（评测脚本用）
GT_TYPE_TO_DETECTOR: dict[str, str | None] = {
    "block": "compression_artifact",
    "edge_compression": "compression_artifact",
    "blur": "blur_artifact",
    "mosaic": "mosaic_artifact",
    "overexposure": "face_artifact",
    "underexposure": "face_artifact",
    "banding": "banding_artifact",
    "noise": None,
    "background": "background_artifact",
    "hair": "hair_texture",
    "hand": "hand_anomaly",
    "face": "face_artifact",
}

# lqdd degradation_type 也可能出现在 report 里
DETECTOR_ALIASES: dict[str, set[str]] = {
    "compression_artifact": {"compression_artifact", "blockiness", "block"},
    "edge_bleed": {"edge_bleed", "green_spill", "edge"},
    "blur_artifact": {"blur_artifact", "blur", "texture_loss"},
    "mosaic_artifact": {"mosaic_artifact", "mosaic"},
    "banding_artifact": {"banding_artifact", "banding"},
    "background_artifact": {"background_artifact", "background"},
    "hair_texture": {"hair_texture", "hair_texture_loss", "hair"},
    "face_artifact": {"face_artifact", "overexposure", "face_blur", "face"},
    "hand_anomaly": {"hand_anomaly", "hand"},
}


def is_supported_gt_type(gt_type: str) -> bool:
    return GT_TYPE_TO_DETECTOR.get(gt_type) is not None


def pred_matches_gt_type(detector: str, degradation_type: str, gt_type: str) -> bool:
    expected = GT_TYPE_TO_DETECTOR.get(gt_type)
    if expected is None:
        return False
    aliases = DETECTOR_ALIASES.get(expected, {expected})
    return detector in aliases or degradation_type in aliases
