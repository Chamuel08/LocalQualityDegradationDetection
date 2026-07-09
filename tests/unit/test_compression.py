import numpy as np

from lqdd.config.loader import CompressionConfig
from lqdd.detectors.compression.detector import CompressionArtifactDetector
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


def _blocky_frame() -> np.ndarray:
    import cv2

    small = np.random.default_rng(0).integers(0, 255, (30, 40, 3), dtype=np.uint8)
    return cv2.resize(small, (160, 120), interpolation=cv2.INTER_NEAREST)


def test_compression_detects_blockiness() -> None:
    frame = _blocky_frame()
    h, w = frame.shape[:2]
    detector = CompressionArtifactDetector(CompressionConfig(blockiness_coarse_threshold=1.0))
    scan = GlobalScanOutput(
        frame_index=0,
        segmentation_map=np.zeros((h, w), dtype=np.uint8),
        global_quality_score=0.4,
        is_fast_pass=False,
        is_fast_reject=False,
        nominations=[],
        scan_duration_ms=1.0,
        overlay_mask=np.zeros((h, w), dtype=bool),
    )
    items = detector.detect(SingleFrameInput(frame=frame, frame_id="block", mode="fast"), scan)
    assert items
    assert items[0].detector == "compression_artifact"
    assert items[0].evidence.metric == "blockiness_score"
    assert items[0].mos_impact < 0
