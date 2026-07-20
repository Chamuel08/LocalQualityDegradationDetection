import cv2
import numpy as np

from lqdd.config.loader import BlurConfig, MosaicConfig, BandingConfig
from lqdd.detectors.banding.detector import BandingArtifactDetector
from lqdd.detectors.blur.detector import BlurArtifactDetector
from lqdd.detectors.mosaic.detector import MosaicArtifactDetector
from lqdd.models.inputs import GlobalScanOutput, SingleFrameInput


def _scan(h: int, w: int, **kwargs) -> GlobalScanOutput:
    defaults = dict(
        frame_index=0,
        segmentation_map=np.zeros((h, w), dtype=np.uint8),
        global_quality_score=0.4,
        is_fast_pass=False,
        is_fast_reject=False,
        nominations=[],
        scan_duration_ms=1.0,
    )
    defaults.update(kwargs)
    return GlobalScanOutput(**defaults)


def test_blur_detector_on_gaussian_blur() -> None:
    # 带结构的纹理图 + 细高频噪声 + 轻度高斯模糊：模拟真实模糊导致的高频损失，
    # 但 lap_var 仍 >= 80（不被 is_ai_generated_style 误判为 AI 合成柔和风格而跳过）。
    rng = np.random.default_rng(2)
    small = rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)
    base = cv2.resize(small, (320, 240), interpolation=cv2.INTER_LINEAR)
    base = cv2.add(base, rng.integers(0, 120, (240, 320, 3), dtype=np.uint8))
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=0.8)
    h, w = blurred.shape[:2]
    fg = np.ones((h, w), dtype=bool)
    detector = BlurArtifactDetector(BlurConfig(texture_loss_threshold=0.25))
    items = detector.detect(
        SingleFrameInput(frame=blurred, frame_id="blur", mode="fast"),
        _scan(h, w, foreground_mask=fg),
    )
    assert items
    assert items[0].detector == "blur_artifact"


def test_mosaic_detector_on_nearest_upscale() -> None:
    import cv2

    base = np.random.default_rng(3).integers(0, 255, (12, 16, 3), dtype=np.uint8)
    frame = cv2.resize(base, (320, 240), interpolation=cv2.INTER_NEAREST)
    h, w = frame.shape[:2]
    detector = MosaicArtifactDetector(MosaicConfig(score_threshold=0.35))
    items = detector.detect(
        SingleFrameInput(frame=frame, frame_id="mosaic", mode="fast"),
        _scan(h, w),
    )
    assert items
    assert items[0].detector == "mosaic_artifact"


def test_banding_detector_on_quantized_gradient() -> None:
    h, w = 180, 260
    gradient = np.linspace(20, 230, h, dtype=np.uint8)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        val = int(gradient[y] // 10) * 10
        img[y, :] = (val, val // 2, 255 - val // 2)
    bg = np.ones((h, w), dtype=bool)
    detector = BandingArtifactDetector(BandingConfig(score_threshold=0.32))
    items = detector.detect(
        SingleFrameInput(frame=img, frame_id="band", mode="fast"),
        _scan(h, w, foreground_mask=~bg),
    )
    assert items
    assert items[0].detector == "banding_artifact"
