from __future__ import annotations

import time

from lqdd.agent.router import default_fast_detectors
from lqdd.config.loader import AppConfig
from lqdd.detectors.registry import build_detector_registry, run_detectors
from lqdd.global_scan.scanner import GlobalScanner
from lqdd.models.inputs import SingleFrameInput
from lqdd.models.report import PerformanceMetrics, QualityReport, TraceEntry
from lqdd.report.generator import ReportGenerator


def run_agent(*_args, **_kwargs) -> None:
    """V1 placeholder — see specs/002-v1-agent-layer."""
    raise NotImplementedError("Agent layer is deferred to V1 (specs/002-v1-agent-layer)")


class FastPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.scanner = GlobalScanner(config.global_scan)
        self.registry = build_detector_registry(config)
        self.report_generator = ReportGenerator(config.report)

    def run(self, frame_input: SingleFrameInput) -> QualityReport:
        t0 = time.perf_counter()
        traces: list[TraceEntry] = []

        traces.append(
            TraceEntry(
                stage="mode_select",
                module="FastPipeline",
                timestamp_ms=0.0,
                duration_ms=0.0,
                input_summary={"requested_mode": frame_input.mode},
                output_summary={"selected_mode": "fast"},
                decision="mode_fast",
                mode="fast",
            )
        )

        scan = self.scanner.scan(frame_input)
        traces.append(
            TraceEntry(
                stage="global_scan",
                module="GlobalScanner",
                timestamp_ms=0.0,
                duration_ms=scan.scan_duration_ms,
                input_summary={"frame_id": frame_input.frame_id, "shape": list(frame_input.frame.shape)},
                output_summary={
                    "nominations": len(scan.nominations),
                    "global_quality_score": scan.global_quality_score,
                },
                decision="scan_complete",
                mode="fast",
            )
        )

        routed = default_fast_detectors()
        traces.append(
            TraceEntry(
                stage="routing",
                module="FastRouter",
                timestamp_ms=scan.scan_duration_ms,
                duration_ms=0.0,
                input_summary={"nominations": [n.suggested_detectors for n in scan.nominations]},
                output_summary={"detectors": routed},
                decision="full_detector_suite",
                mode="fast",
            )
        )

        t_det = time.perf_counter()
        degradations = run_detectors(self.registry, routed, frame_input, scan)
        det_ms = (time.perf_counter() - t_det) * 1000.0

        traces.append(
            TraceEntry(
                stage="detection",
                module="FastDetectors",
                timestamp_ms=scan.scan_duration_ms,
                duration_ms=det_ms,
                input_summary={"detectors": routed},
                output_summary={"degradation_count": len(degradations)},
                decision="detectors_complete",
                mode="fast",
            )
        )

        t_agg = time.perf_counter()
        total_ms = (time.perf_counter() - t0) * 1000.0
        agg_ms = (time.perf_counter() - t_agg) * 1000.0
        perf = PerformanceMetrics(
            total_ms=total_ms,
            global_scan_ms=scan.scan_duration_ms,
            detection_ms=det_ms,
            aggregation_ms=agg_ms,
        )

        traces.append(
            TraceEntry(
                stage="aggregation",
                module="ReportGenerator",
                timestamp_ms=scan.scan_duration_ms + det_ms,
                duration_ms=agg_ms,
                input_summary={"degradation_count": len(degradations)},
                output_summary={"severity_pending": True},
                decision="aggregate_mos",
                mode="fast",
            )
        )

        return self.report_generator.generate(
            frame_input,
            scan,
            degradations,
            traces,
            perf,
        )
