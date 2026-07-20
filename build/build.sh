#!/bin/bash
set -euo pipefail

# ────────────────────────────────────────────────────────────
# LQDD GUI — 一键构建脚本
# 产出: dist/lqdd-gui/lqdd-gui (可执行文件)
# 运行: 启动本地 Gradio server + pywebview 原生窗口
# 前置: Xcode Command Line Tools (macOS) / Python 3.11
# 注意: 可执行文件不含 Ollama/模型权重，需另装 Ollama 并 pull 模型
# ────────────────────────────────────────────────────────────

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV="build/venv-pack"
PYTHON="$VENV/bin/python"

echo "============================================"
echo "  LQDD GUI — PyInstaller Build"
echo "============================================"

# ── Step 0: venv ──
if [ ! -f "$PYTHON" ]; then
    echo "[0/3] Creating Python 3.11 venv..."
    python3.11 -m venv "$VENV"
    "$VENV/bin/pip" install --upgrade pip wheel setuptools
    "$VENV/bin/pip" install -r build/requirements_pack.txt
fi
echo "[OK] Python venv ready ($($PYTHON --version 2>&1))"

# ── Step 1: 安装 lqdd（可编辑，供 PyInstaller 收集）──
echo ""
echo "[1/3] Installing lqdd (editable)..."
"$VENV/bin/pip" install -e . --no-deps
echo "[OK] lqdd installed"

# ── Step 2: PyInstaller ──
echo ""
echo "[2/3] PyInstaller packaging..."
"$VENV/bin/pyinstaller" build/app.spec \
    --distpath dist \
    --workpath build/pyi_build \
    --noconfirm
echo "[OK] Packaged"

# ── Step 3: 输出 ──
echo ""
echo "============================================"
echo "  BUILD COMPLETE"
echo "============================================"
if [ -d "dist/lqdd-gui" ]; then
    echo "  产物: dist/lqdd-gui/lqdd-gui"
    if [ "$(uname)" = "Darwin" ]; then
        du -sh dist/lqdd-gui 2>/dev/null | awk '{print "  体积: "$2}'
    fi
fi
echo ""
echo "  运行: ./dist/lqdd-gui/lqdd-gui"
echo "  前置: ollama serve + ollama pull qwen2.5vl:7b + ollama pull qwen2.5:1.5b"
echo "  降级: MOS=rule, hand_anomaly=边缘密度 fallback"
echo "============================================"
