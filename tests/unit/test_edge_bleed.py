import numpy as np

from lqdd.config.loader import EdgeBleedConfig
from lqdd.detectors.edge_bleed.detector import EdgeBleedDetector
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


def _green_edge_frame() -> np.ndarray:
    h, w = 120, 160
    img = np.full((h, w, 3), (40, 80, 120), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.circle(img, (w // 2, h // 2), 30, (200, 180, 160), -1)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    edge = (dist >= 24) & (dist <= 36)
    img[edge] = (10, 240, 10)
    return img


def _scan_with_edge(frame: np.ndarray) -> GlobalScanOutput:
    h, w = frame.shape[:2]
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    edge_mask = (dist >= 24) & (dist <= 36)
    fg = dist < 30
    return GlobalScanOutput(
        frame_index=0,
        segmentation_map=np.zeros((h, w), dtype=np.uint8),
        global_quality_score=0.5,
        is_fast_pass=False,
        is_fast_reject=False,
        nominations=[],
        scan_duration_ms=1.0,
        foreground_mask=fg,
        edge_mask=edge_mask,
    )


def test_edge_bleed_detects_green_spill() -> None:
    frame = _green_edge_frame()
    detector = EdgeBleedDetector(EdgeBleedConfig())
    items = detector.detect(
        SingleFrameInput(frame=frame, frame_id="test", mode="fast"),
        _scan_with_edge(frame),
    )
    assert items
    item = items[0]
    assert item.detector == "edge_bleed"
    assert "绿" in item.evidence.detail or "溢出" in item.evidence.detail
    assert item.mos_impact < 0
    assert item.root_cause_hypothesis.cause == "matting_error"


def test_edge_bleed_skips_high_delta_e_without_green_spill() -> None:
    """Portrait-like edge: background ΔE high but no green channel excess."""
    h, w = 128, 72
    frame = np.full((h, w, 3), (30, 40, 180), dtype=np.uint8)
    cv2 = __import__("cv2")
    cv2.circle(frame, (w // 2, h // 2), 22, (210, 180, 160), -1)
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    edge_mask = (dist >= 20) & (dist <= 28)
    fg = dist < 24
    scan = GlobalScanOutput(
        frame_index=0,
        segmentation_map=np.zeros((h, w), dtype=np.uint8),
        global_quality_score=0.5,
        is_fast_pass=False,
        is_fast_reject=False,
        nominations=[],
        scan_duration_ms=1.0,
        foreground_mask=fg,
        edge_mask=edge_mask,
    )
    detector = EdgeBleedDetector(EdgeBleedConfig())
    items = detector.detect(
        SingleFrameInput(frame=frame, frame_id="portrait", mode="fast"),
        scan,
    )
    assert items == []
