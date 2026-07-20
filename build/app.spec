# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec — LQDD GUI 可执行文件打包

产物: dist/lqdd-gui/lqdd-gui (目录模式)
运行: 启动本地 Gradio server + pywebview 原生窗口

使用:
  pyinstaller build/app.spec --distpath dist --workpath build/pyi_build --noconfirm
"""
import os
import sys
from PyInstaller.utils.hooks import collect_all

_cwd = os.getcwd()
PROJECT_ROOT = _cwd
for _ in range(5):
    if os.path.isdir(os.path.join(PROJECT_ROOT, "src", "lqdd")):
        break
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

datas = []
binaries = []
hiddenimports = []

# ── 收集 GUI 依赖完整资源 ──
for pkg in ["gradio", "gradio_client", "pywebview", "cv2", "numpy", "yaml", "jsonschema", "httpx"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# ── 数据文件: 配置模板 ──
_cfg = os.path.join(PROJECT_ROOT, "config.example.yaml")
if os.path.exists(_cfg):
    datas.append((_cfg, "."))

# ── lqdd 包源码（src layout，需显式加入 pathex）──
a = Analysis(
    [os.path.join(PROJECT_ROOT, "src", "lqdd", "ui", "app.py")],
    pathex=[os.path.join(PROJECT_ROOT, "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "torch",
        "torchvision",
        "pyiqa",
        "mediapipe",
        "matplotlib",
        "PIL",
        "pytest",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lqdd-gui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # cv2 .so 被 UPX 压缩后容易损坏
    console=False,  # 原生窗口模式无需控制台；调试可改 True
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="lqdd-gui",
)
