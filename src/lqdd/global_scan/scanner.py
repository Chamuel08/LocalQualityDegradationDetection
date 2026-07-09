from __future__ import annotations

import time

import cv2
import numpy as np

from lqdd.config.loader import GlobalScanConfig
from lqdd.global_scan.nomination import build_nominations
from lqdd.global_scan.segmentation import build_edge_band, build_segmentation_map, segment_foreground
from lqdd.global_scan.text_ui import detect_text_ui_bands, merge_ignore_regions
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


class GlobalScanner:
    def __init__(self, config: GlobalScanConfig) -> None:
        self.config = config

    def scan(self, frame_input: SingleFrameInput) -> GlobalScanOutput:
        t0 = time.perf_counter()
        frame = frame_input.frame
        h, w = frame.shape[:2]

        foreground = segment_foreground(frame)
        edge_mask = build_edge_band(foreground, self.config.edge_expand_px)
        text_ui = detect_text_ui_bands(frame, self.config)
        overlay_mask = merge_ignore_regions(text_ui, frame_input.ignore_regions, h, w)
        seg_map = build_segmentation_map(foreground, edge_mask, overlay_mask, self.config)
        nominations = build_nominations(frame, foreground, edge_mask, overlay_mask, self.config)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        global_score = float(np.clip(lap_var / 500.0, 0, 1))

        duration_ms = (time.perf_counter() - t0) * 1000.0
        return GlobalScanOutput(
            frame_index=0,
            segmentation_map=seg_map,
            global_quality_score=global_score,
            is_fast_pass=False,
            is_fast_reject=False,
            nominations=nominations,
            scan_duration_ms=duration_ms,
            overlay_mask=overlay_mask,
            foreground_mask=foreground,
            edge_mask=edge_mask & ~overlay_mask,
        )
