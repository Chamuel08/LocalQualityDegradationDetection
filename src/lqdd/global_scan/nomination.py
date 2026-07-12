from __future__ import annotations

import cv2
import numpy as np

from lqdd.config.loader import GlobalScanConfig
from lqdd.detectors.base import (
    bbox_from_mask,
    compute_blockiness_score,
    compute_banding_score,
    compute_mosaic_score,
    compute_texture_loss_score,
    face_region_mask,
    fft_highfreq_ratio,
    hair_region_mask,
    hand_region_mask,
)
from lqdd.models.enums import RegionType
from lqdd.models.inputs import RegionNomination


def build_nominations(
    frame_bgr: np.ndarray,
    foreground: np.ndarray,
    edge_mask: np.ndarray,
    overlay_mask: np.ndarray,
    cfg: GlobalScanConfig,
) -> list[RegionNomination]:
    nominations: list[RegionNomination] = []
    h, w = frame_bgr.shape[:2]
    fg_area = foreground.sum()
    if fg_area < h * w * 0.01:
        return nominations

    gray = np.dot(frame_bgr[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)

    edge_active = edge_mask & ~overlay_mask
    if edge_active.any():
        bbox = bbox_from_mask(edge_active)
        b, g, r = frame_bgr[:, :, 0], frame_bgr[:, :, 1], frame_bgr[:, :, 2]
        green_excess = g.astype(np.float32) - 0.5 * (r.astype(np.float32) + b.astype(np.float32))
        edge_vals = green_excess[edge_active]
        green_bias = float(np.clip(edge_vals.mean() / 128.0, 0, 1)) if edge_vals.size else 0.0
        nominations.append(
            RegionNomination(
                region_type=int(RegionType.EDGE),
                bbox=bbox,
                mask=edge_active,
                anomaly_score=min(1.0, 0.4 + green_bias),
                confidence=min(1.0, 0.5 + green_bias * 0.4),
                suggested_detectors=["edge_bleed"],
                features={"green_channel_bias": green_bias},
            )
        )

    blockiness = compute_blockiness_score(gray)
    block_hint = float(np.clip((blockiness - 1.0) / 2.0, 0, 1))
    if blockiness >= cfg.nomination_threshold or block_hint > 0.3:
        bg_mask = (~foreground) & (~overlay_mask)
        if not bg_mask.any():
            bg_mask = np.ones((h, w), dtype=bool)
        nominations.append(
            RegionNomination(
                region_type=int(RegionType.BACKGROUND),
                bbox=bbox_from_mask(bg_mask),
                mask=bg_mask,
                anomaly_score=block_hint,
                confidence=min(1.0, 0.45 + block_hint * 0.4),
                suggested_detectors=["compression_artifact", "background_artifact", "banding_artifact"],
                features={"blockiness_hint": block_hint, "blockiness_score": blockiness},
            )
        )

    mosaic_hint = compute_mosaic_score(gray)
    if mosaic_hint >= 0.35:
        body_mask = foreground & ~overlay_mask
        nominations.append(
            RegionNomination(
                region_type=int(RegionType.BODY),
                bbox=bbox_from_mask(body_mask) if body_mask.any() else (0, 0, w, h),
                mask=body_mask if body_mask.any() else foreground,
                anomaly_score=mosaic_hint,
                confidence=min(1.0, 0.42 + mosaic_hint * 0.45),
                suggested_detectors=["mosaic_artifact", "compression_artifact"],
                features={"mosaic_hint": mosaic_hint},
            )
        )

    x, y, bw, bh = bbox_from_mask(foreground)
    if bw > 0 and bh > 0:
        body_crop = gray[y : y + bh, x : x + bw]
        texture_loss = compute_texture_loss_score(body_crop, reference_var=2400.0)
        if texture_loss >= 0.22:
            body_mask = foreground & ~overlay_mask
            nominations.append(
                RegionNomination(
                    region_type=int(RegionType.BODY),
                    bbox=(x, y, bw, bh),
                    mask=body_mask,
                    anomaly_score=texture_loss,
                    confidence=min(1.0, 0.4 + texture_loss * 0.5),
                    suggested_detectors=["blur_artifact", "compression_artifact"],
                    features={"texture_loss_hint": texture_loss},
                )
            )

    hair = hair_region_mask(foreground, h, w)
    if hair.sum() > 400:
        ys, xs = np.where(hair)
        hair_crop = gray[int(ys.min()) : int(ys.max()) + 1, int(xs.min()) : int(xs.max()) + 1]
        hf_ratio = fft_highfreq_ratio(hair_crop)
        hair_loss = float(np.clip(1.0 - hf_ratio / 0.22, 0, 1))
        if hair_loss >= 0.35:
            nominations.append(
                RegionNomination(
                    region_type=int(RegionType.HAIR),
                    bbox=bbox_from_mask(hair),
                    mask=hair,
                    anomaly_score=hair_loss,
                    confidence=min(1.0, 0.45 + hair_loss * 0.4),
                    suggested_detectors=["hair_texture"],
                    features={"hair_highfreq_ratio": hf_ratio},
                )
            )

    face = face_region_mask(foreground, h, w)
    if face.sum() > 300:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        over_ratio = float((lab[:, :, 0][face] > 220).mean())
        lap_var = float(cv2.Laplacian(gray[face].reshape(-1, 1), cv2.CV_64F).var())
        face_score = max(over_ratio, float(np.clip(1.0 - lap_var / 120.0, 0, 1)))
        if face_score >= 0.3:
            nominations.append(
                RegionNomination(
                    region_type=int(RegionType.FACE),
                    bbox=bbox_from_mask(face),
                    mask=face,
                    anomaly_score=face_score,
                    confidence=min(1.0, 0.48 + face_score * 0.35),
                    suggested_detectors=["face_artifact"],
                    features={"face_overexposure": over_ratio, "face_laplacian": lap_var},
                )
            )

    hand = hand_region_mask(foreground, h, w)
    if hand.sum() > 250:
        edges = cv2.Canny(gray, 60, 140)
        edge_density = float(edges[hand].mean() / 255.0)
        if edge_density >= 0.1:
            nominations.append(
                RegionNomination(
                    region_type=int(RegionType.HAND),
                    bbox=bbox_from_mask(hand),
                    mask=hand,
                    anomaly_score=min(1.0, edge_density * 3.0),
                    confidence=min(1.0, 0.42 + edge_density),
                    suggested_detectors=["hand_anomaly"],
                    features={"hand_edge_density": edge_density},
                )
            )

    bg = (~foreground) & (~overlay_mask)
    if bg.sum() > h * w * 0.1:
        xb, yb, bwb, bhb = bbox_from_mask(bg)
        band_score = compute_banding_score(gray[yb : yb + bhb, xb : xb + bwb])
        if band_score >= 0.32:
            nominations.append(
                RegionNomination(
                    region_type=int(RegionType.BACKGROUND),
                    bbox=(xb, yb, bwb, bhb),
                    mask=bg,
                    anomaly_score=band_score,
                    confidence=min(1.0, 0.44 + band_score * 0.4),
                    suggested_detectors=["banding_artifact", "background_artifact"],
                    features={"banding_hint": band_score},
                )
            )

    return nominations
