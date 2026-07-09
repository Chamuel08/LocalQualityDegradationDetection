from enum import IntEnum, StrEnum


class RegionType(IntEnum):
    BACKGROUND = 0
    FACE = 1
    HAIR = 2
    HAND = 3
    EDGE = 4
    BODY = 6
    TEXT_UI = 7


class Severity(StrEnum):
    GOOD = "good"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class RootCauseCategory(StrEnum):
    GENERATION_ARTIFACT = "generation_artifact"
    ENCODING_LOSS = "encoding_loss"
    ENHANCEMENT_ARTIFACT = "enhancement_artifact"
    MATTING_ERROR = "matting_error"
    LIGHTING_INCONSISTENCY = "lighting_inconsistency"
    OTHER = "other"
