import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _run_detect(args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(REPO / "detect.py"), *args]
    return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)


LEGACY = ["--legacy-fixed"]


@pytest.fixture(scope="module", autouse=True)
def ensure_samples() -> None:
    script = REPO / "scripts" / "generate_synthetic_samples.py"
    if not (REPO / "data" / "sample" / "frames" / "edge" / "edge_01.png").is_file():
        subprocess.run([sys.executable, str(script)], cwd=REPO, check=True)


def test_cli_single_edge_json() -> None:
    image = REPO / "data" / "sample" / "frames" / "edge" / "edge_01.png"
    proc = _run_detect(["--image", str(image), "--mode", "fast", *LEGACY])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    detectors = {d["detector"] for d in data["degradations"]}
    assert "edge_bleed" in detectors
    assert all("detail" in d["evidence"] for d in data["degradations"])


def test_cli_block_detection() -> None:
    image = REPO / "data" / "sample" / "frames" / "block" / "block_01.png"
    proc = _run_detect(["--image", str(image), "--mode", "fast", *LEGACY])
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert any(d["detector"] == "compression_artifact" for d in data["degradations"])


def test_cli_batch_output_dir(tmp_path: Path) -> None:
    image_dir = REPO / "data" / "sample" / "frames" / "normal"
    out_dir = tmp_path / "reports"
    proc = _run_detect(
        ["--image-dir", str(image_dir), "--mode", "fast", "--output-dir", str(out_dir), *LEGACY]
    )
    assert proc.returncode == 0, proc.stderr
    json_files = list(out_dir.glob("*.json"))
    assert len(json_files) >= 3


def test_cli_html_output(tmp_path: Path) -> None:
    image = REPO / "data" / "sample" / "frames" / "edge" / "edge_01.png"
    html_path = tmp_path / "report.html"
    proc = _run_detect(["--image", str(image), "--mode", "fast", "--output", str(html_path)])
    assert proc.returncode == 0, proc.stderr
    text = html_path.read_text(encoding="utf-8")
    assert "method" in text
    assert "metric" in text


def test_cli_invalid_mode() -> None:
    image = REPO / "data" / "sample" / "frames" / "edge" / "edge_01.png"
    proc = _run_detect(["--image", str(image), "--mode", "deep"])
    assert proc.returncode == 2
    assert "deep mode" in proc.stderr


def test_cli_missing_image() -> None:
    proc = _run_detect(["--image", "/nonexistent/frame.png", "--mode", "fast"])
    assert proc.returncode == 1
    assert "image not found" in proc.stderr
