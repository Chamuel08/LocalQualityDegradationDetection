from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "specs" / "001-v0-fast-mvp" / "contracts" / "quality-report.schema.json"
SCHEMA_V1_PATH = REPO_ROOT / "specs" / "002-v1-agent-layer" / "contracts" / "quality-report.v1.schema.json"
VLM_SCHEMA_PATH = REPO_ROOT / "specs" / "002-v1-agent-layer" / "contracts" / "vlm-confirm.schema.json"
JUDGE_SCHEMA_PATH = REPO_ROOT / "specs" / "002-v1-agent-layer" / "contracts" / "llm-judge.schema.json"
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SAMPLE_ROOT = REPO_ROOT / "data" / "sample"
EDGE_DIR = SAMPLE_ROOT / "frames" / "edge"
BLOCK_DIR = SAMPLE_ROOT / "frames" / "block"
NORMAL_DIR = SAMPLE_ROOT / "frames" / "normal"
EXPECTED_DIR = SAMPLE_ROOT / "expected"


@pytest.fixture
def schema_path() -> Path:
    return SCHEMA_PATH


@pytest.fixture
def schema_v1_path() -> Path:
    return SCHEMA_V1_PATH


@pytest.fixture
def vlm_schema_path() -> Path:
    return VLM_SCHEMA_PATH


@pytest.fixture
def judge_schema_path() -> Path:
    return JUDGE_SCHEMA_PATH


@pytest.fixture
def mock_vlm_fixture() -> Path:
    return FIXTURES / "mock_vlm_responses" / "grey_confirm.json"


@pytest.fixture
def mock_judge_dispatch_fixture() -> Path:
    return FIXTURES / "mock_judge_responses" / "dispatch_compression.json"


@pytest.fixture
def sample_edge_dir() -> Path:
    return EDGE_DIR


@pytest.fixture
def sample_block_dir() -> Path:
    return BLOCK_DIR


@pytest.fixture
def sample_normal_dir() -> Path:
    return NORMAL_DIR


@pytest.fixture
def expected_dir() -> Path:
    return EXPECTED_DIR
