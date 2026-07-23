from __future__ import annotations

"""VideoClipRunner — 视频 clip 输入的外层包装器。

**设计说明**

V1 的所有核心代码（SingleFrameInput、检测器、AgentPipeline）均是单帧接口，
本模块通过「外层包装」方式支持多帧输入，**不修改任何单帧内部接口**：

- 对每帧独立调用 pipeline.run(SingleFrameInput)，得到逐帧报告
- 在帧间聚合层额外运行 TemporalFlicker 检测（依赖多帧，单帧无法做）
- 返回 VideoClipReport（逐帧报告 + 时域闪烁摘要 + 聚合 MOS）

CLI 用法（示例）：
    from lqdd.pipeline.video_clip_runner import VideoClipRunner, sample_frames_from_video
    from lqdd.config.loader import load_config
    from lqdd.pipeline.agent_pipeline import AgentPipeline

    config = load_config()
    pipeline = AgentPipeline(config)
    runner = VideoClipRunner(pipeline)

    frames = sample_frames_from_video("input.mp4", max_frames=8)
    result = runner.run(frames, clip_id="clip_001")
    print(result.aggregate_mos, result.flicker_result.is_flickering)
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lqdd.temporal_flicker.detector import TemporalFlickerResult, detect_temporal_flicker
from lqdd.models.inputs import SingleFrameInput
from lqdd.models.report import QualityReport


@dataclass
class VideoClipReport:
    """VideoClipRunner 的聚合输出。"""

    clip_id: str
    frame_count: int
    frame_reports: list[QualityReport]
    flicker_result: TemporalFlickerResult
    aggregate_mos: float | None          # 逐帧 MOS 的均值；全部不可用时为 None
    worst_frame_mos: float | None
    worst_frame_index: int               # 最差帧在 frame_reports 中的索引；无可用 MOS 时为 -1
    degradation_summary: dict           # 各 degradation_type 跨帧出现次数


class VideoClipRunner:
    """对视频 clip（多帧列表）运行单帧 pipeline + TemporalFlicker 聚合。

    Args:
        pipeline: 支持 .run(SingleFrameInput) -> QualityReport 的 pipeline 对象
            （AgentPipeline 或 FastPipeline 均可）。
        flicker_luma_threshold: 亮度跳变阈值（灰度值，0-255）。
        flicker_hue_threshold: 色相跳变阈值（H 通道，0-180）。
        flicker_min_ratio: 闪烁帧间比例阈值，超过即报告整体闪烁。
    """

    def __init__(
        self,
        pipeline,
        flicker_luma_threshold: float = 8.0,
        flicker_hue_threshold: float = 6.0,
        flicker_min_ratio: float = 0.15,
    ) -> None:
        self.pipeline = pipeline
        self.flicker_luma_threshold = flicker_luma_threshold
        self.flicker_hue_threshold = flicker_hue_threshold
        self.flicker_min_ratio = flicker_min_ratio

    def run(
        self,
        frames: list[np.ndarray],
        clip_id: str = "clip",
        frame_ids: list[str] | None = None,
    ) -> VideoClipReport:
        """对帧列表运行检测并聚合结果。

        Args:
            frames: BGR uint8 帧列表，至少 1 帧。
            clip_id: clip 标识符（用于报告 video_id）。
            frame_ids: 可选的逐帧 ID 列表，不足时自动补全为 f"{clip_id}_f{i:04d}"。

        Returns:
            VideoClipReport
        """
        if not frames:
            raise ValueError("frames 列表不能为空")

        n = len(frames)
        if frame_ids is None:
            frame_ids = [f"{clip_id}_f{i:04d}" for i in range(n)]
        else:
            # 补全到 n 个
            frame_ids = list(frame_ids) + [f"{clip_id}_f{i:04d}" for i in range(len(frame_ids), n)]

        # --- 逐帧运行 pipeline ---
        frame_reports: list[QualityReport] = []
        for i, (frame, fid) in enumerate(zip(frames, frame_ids)):
            fi = SingleFrameInput(
                frame=frame,
                frame_id=fid,
                mode="fast",
            )
            report = self.pipeline.run(fi)
            frame_reports.append(report)

        # --- TemporalFlicker（帧间层，单帧无法做）---
        flicker_result = detect_temporal_flicker(
            frames,
            luma_delta_threshold=self.flicker_luma_threshold,
            hue_delta_threshold=self.flicker_hue_threshold,
            min_flicker_ratio=self.flicker_min_ratio,
        )

        # --- 聚合 MOS（跳过 overall_mos 为 None 的帧，即 CLIP-IQA 不可用的帧）---
        mos_list = [r.overall_mos for r in frame_reports if r.overall_mos is not None]
        if mos_list:
            aggregate_mos = round(float(np.mean(mos_list)), 3)
            worst_mos = round(float(min(mos_list)), 3)
            # worst_frame_index 指向第一个 overall_mos == min 的帧
            worst_idx = next(
                i for i, r in enumerate(frame_reports) if r.overall_mos == worst_mos
            )
        else:
            aggregate_mos = None
            worst_mos = None
            worst_idx = -1

        # --- 跨帧 degradation 汇总 ---
        degradation_summary: dict[str, int] = {}
        for report in frame_reports:
            for deg in report.degradations:
                degradation_summary[deg.degradation_type] = (
                    degradation_summary.get(deg.degradation_type, 0) + 1
                )

        return VideoClipReport(
            clip_id=clip_id,
            frame_count=n,
            frame_reports=frame_reports,
            flicker_result=flicker_result,
            aggregate_mos=aggregate_mos,
            worst_frame_mos=worst_mos,
            worst_frame_index=worst_idx,
            degradation_summary=degradation_summary,
        )


def sample_frames_from_video(
    video_path: str | Path,
    max_frames: int = 8,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[np.ndarray]:
    """从视频文件中均匀抽取帧。

    Args:
        video_path: 视频文件路径（mp4 / mov / avi 等 OpenCV 支持的格式）。
        max_frames: 最多抽取帧数（均匀间隔）。
        start_sec: 起始时间（秒）。
        end_sec: 结束时间（秒），None 表示到视频末尾。

    Returns:
        BGR uint8 numpy 帧列表。

    Raises:
        FileNotFoundError: 视频文件不存在。
        RuntimeError: 视频无法打开或无可用帧。
    """
    import cv2

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在：{path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec is not None else total_frames
    end_frame = min(end_frame, total_frames)
    clip_len = max(1, end_frame - start_frame)

    # 均匀采样索引
    if clip_len <= max_frames:
        indices = list(range(start_frame, end_frame))
    else:
        step = clip_len / max_frames
        indices = [int(start_frame + i * step) for i in range(max_frames)]

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"从视频 {path} 中未能读取到任何帧")

    return frames