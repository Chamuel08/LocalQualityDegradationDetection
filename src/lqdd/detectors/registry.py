from __future__ import annotations

from lqdd.config.loader import AppConfig
from lqdd.detectors.background.detector import BackgroundArtifactDetector
from lqdd.detectors.banding.detector import BandingArtifactDetector
from lqdd.detectors.base import Detector
from lqdd.detectors.blur.detector import BlurArtifactDetector
from lqdd.detectors.compression.detector import CompressionArtifactDetector
from lqdd.detectors.edge_bleed.detector import EdgeBleedDetector
from lqdd.detectors.face_artifact.detector import FaceArtifactDetector
from lqdd.detectors.hair_texture.detector import HairTextureDetector
from lqdd.detectors.hand_anomaly.detector import HandAnomalyDetector
from lqdd.detectors.mosaic.detector import MosaicArtifactDetector

ALL_DETECTOR_NAMES: list[str] = [
    "edge_bleed",
    "compression_artifact",
    "blur_artifact",
    "mosaic_artifact",
    "banding_artifact",
    "background_artifact",
    "hair_texture",
    "face_artifact",
    "hand_anomaly",
]

CORE_DETECTOR_NAMES: list[str] = ["edge_bleed", "compression_artifact"]


def build_detector_registry(config: AppConfig) -> dict[str, Detector]:
    return {
        "edge_bleed": EdgeBleedDetector(config.edge_bleed),
        "compression_artifact": CompressionArtifactDetector(config.compression),
        "blur_artifact": BlurArtifactDetector(config.blur),
        "mosaic_artifact": MosaicArtifactDetector(config.mosaic),
        "banding_artifact": BandingArtifactDetector(config.banding),
        "background_artifact": BackgroundArtifactDetector(config.background_artifact),
        "hair_texture": HairTextureDetector(config.hair_texture),
        "face_artifact": FaceArtifactDetector(config.face_artifact),
        "hand_anomaly": HandAnomalyDetector(config.hand_anomaly),
    }


def run_detectors(
    registry: dict[str, Detector],
    names: list[str],
    frame_input,
    scan_output,
) -> list:
    degradations = []
    for name in names:
        detector = registry.get(name)
        if detector is None:
            continue
        degradations.extend(detector.detect(frame_input, scan_output))
    return degradations
