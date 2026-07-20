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
    detector = CompressionArtifactDetector(
        CompressionConfig(blockiness_coarse_threshold=1.0, mild_blockiness_threshold=0.9)
    )
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


def test_compression_detects_texture_loss() -> None:
    import cv2

    # 带结构的纹理图 + 细高频噪声 + 轻度高斯模糊：模拟真实压缩导致的高频损失，
    # 但 lap_var 仍 >= 80（不被 is_ai_generated_style 误判为 AI 合成柔和风格而跳过）。
    rng = np.random.default_rng(1)
    small = rng.integers(0, 255, (60, 40, 3), dtype=np.uint8)
    base = cv2.resize(small, (135, 240), interpolation=cv2.INTER_LINEAR)
    base = cv2.add(base, rng.integers(0, 120, (240, 135, 3), dtype=np.uint8))
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=0.8)
    h, w = blurred.shape[:2]
    detector = CompressionArtifactDetector(
        CompressionConfig(texture_loss_threshold=0.35, blockiness_coarse_threshold=1.4)
    )
    scan = GlobalScanOutput(
        frame_index=0,
        segmentation_map=np.zeros((h, w), dtype=np.uint8),
        global_quality_score=0.4,
        is_fast_pass=False,
        is_fast_reject=False,
        nominations=[],
        scan_duration_ms=1.0,
    )
    items = detector.detect(SingleFrameInput(frame=blurred, frame_id="blur", mode="fast"), scan)
    assert items
    assert items[0].evidence.metric == "texture_loss_score"
