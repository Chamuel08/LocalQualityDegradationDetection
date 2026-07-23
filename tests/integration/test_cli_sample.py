import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests"))
from benchmark_util import benchmark_image_for_type, benchmark_manifest_path


def _run_detect(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(REPO / "detect.py"), *args]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


LEGACY = ["--legacy-fixed"]


@pytest.fixture(scope="module", autouse=True)
def ensure_benchmark() -> None:
    manifest = benchmark_manifest_path()
    if manifest.is_file():
        return
    script = REPO / "scripts" / "generate_benchmark_dataset.py"
    if not script.is_file():
        pytest.skip("benchmark generator script missing")
    subprocess.run([sys.executable, str(script), "--samples-per-type", "2", "--clean-count", "2"], cwd=REPO, check=False)


def _require_image(primary_type: str) -> Path:
    img = benchmark_image_for_type(primary_type)
    if img is None or not img.is_file():
        pytest.skip(f"no benchmark image for type={primary_type}")
    return img


def test_cli_single_edge_json() -> None:
    image = _require_image("green_spill") or _require_image("edge_compression")
    proc = _run_detect(["--image", str(image), "--mode", "fast", *LEGACY])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert all("detail" in d["evidence"] for d in data["degradations"])


def test_cli_block_detection() -> None:
    image = _require_image("block")
    proc = _run_detect(["--image", str(image), "--mode", "fast", *LEGACY])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert isinstance(data["degradations"], list)


def test_cli_batch_output_dir(tmp_path: Path) -> None:
    manifest = benchmark_manifest_path()
    if not manifest.is_file():
        pytest.skip("benchmark manifest missing")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    root = manifest.parent
    clean = [s for s in data.get("samples", data) if s.get("is_clean")][:3]
    if len(clean) < 2:
        pytest.skip("not enough clean benchmark samples")
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    for sample in clean:
        src = (root / sample["image"]).resolve()
        dest = clean_dir / f"{sample['id']}.png"
        shutil.copy2(src, dest)
    out_dir = tmp_path / "reports"
    proc = _run_detect(
        ["--image-dir", str(clean_dir), "--mode", "fast", "--output-dir", str(out_dir), *LEGACY]
    )
    assert proc.returncode == 0, proc.stderr
    assert len(list(out_dir.glob("*.json"))) >= 2


def test_cli_html_output(tmp_path: Path) -> None:
    image = _require_image("block")
    html_path = tmp_path / "report.html"
    proc = _run_detect(["--image", str(image), "--mode", "fast", "--output", str(html_path)])
    assert proc.returncode == 0, proc.stderr
    text = html_path.read_text(encoding="utf-8")
    assert "method" in text
    assert "metric" in text


def test_cli_invalid_mode() -> None:
    image = _require_image("block")
    proc = _run_detect(["--image", str(image), "--mode", "deep"])
    assert proc.returncode == 2
    assert "deep mode" in proc.stderr


def test_cli_missing_image() -> None:
    proc = _run_detect(["--image", "/nonexistent/frame.png", "--mode", "fast"])
    assert proc.returncode == 1
    assert "image not found" in proc.stderr


def test_cli_video_output(tmp_path: Path) -> None:
    """--video should emit a VideoClipReport JSON (V2 clip pipeline)."""
    import cv2
    import numpy as np

    video = tmp_path / "tiny.mp4"
    h, w = 128, 128
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video), fourcc, 8, (w, h))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter mp4v not available")
    try:
        for i in range(6):
            frame = np.full((h, w, 3), 40 + i * 10, np.uint8)
            cv2.rectangle(frame, (20, 20), (100, 100), (0, 0, 200), -1)
            writer.write(frame)
    finally:
        writer.release()
    if not video.is_file() or video.stat().st_size == 0:
        pytest.skip("could not write synthetic video")

    out = tmp_path / "video_report.json"
    proc = _run_detect(
        ["--video", str(video), "--mode", "fast", "--legacy-fixed",
         "--max-frames", "4", "--output", str(out)]
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text(encoding="utf-8"))
    # VideoClipReport top-level shape
    assert data["clip_id"] == "tiny"
    assert data["frame_count"] == 4
    assert "aggregate_mos" in data  # number or null
    assert "worst_frame_mos" in data
    assert "worst_frame_index" in data
    assert "flicker" in data and "is_flickering" in data["flicker"]
    assert isinstance(data["degradation_summary"], dict)
    assert isinstance(data["frame_reports"], list) and len(data["frame_reports"]) == 4


def test_cli_video_missing_file(tmp_path: Path) -> None:
    proc = _run_detect(["--video", "/nonexistent/clip.mp4", "--mode", "fast", "--legacy-fixed"])
    assert proc.returncode == 1
    assert "video not found" in proc.stderr
