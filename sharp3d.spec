# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for sharp3d --onedir build."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

block_cipher = None

# Project paths
ROOT = Path(SPECPATH)
SRC = ROOT / "src"
ML_SHARP_SRC = ROOT.parent / "ml-sharp" / "src"
SITE_PACKAGES = Path(sys.executable).parent.parent / "Lib" / "site-packages"

# ---- Collect binaries (DLLs / .pyd) ----
binaries = []

# PyTorch CUDA
binaries += collect_dynamic_libs("torch")

# TensorRT
binaries += collect_dynamic_libs("tensorrt_libs")
binaries += collect_dynamic_libs("tensorrt_bindings")

# ONNX Runtime
binaries += collect_dynamic_libs("onnxruntime")

# gsplat CUDA extensions
binaries += collect_dynamic_libs("gsplat")

# triton (if present)
try:
    binaries += collect_dynamic_libs("triton")
except Exception:
    pass

# ---- Collect data files ----
datas = []

# sharp3d source
datas += [(str(SRC / "sharp3d"), "sharp3d")]

# ml-sharp (sharp) source
datas += [(str(ML_SHARP_SRC / "sharp"), "sharp")]

# Assets (icon)
datas += [(str(ROOT / "assets" / "sharp3d_icon.ico"), "assets")]

# Pre-exported ONNX models + FP16 weights (skip download/export at runtime)
_cache = ROOT / ".cache"
if (_cache / "onnx" / "patch_encoder.onnx").exists():
    datas += [(str(_cache / "onnx" / "patch_encoder.onnx"), "models")]
    datas += [(str(_cache / "onnx" / "patch_encoder.onnx.data"), "models")]
    datas += [(str(_cache / "onnx" / "image_encoder.onnx"), "models")]
    datas += [(str(_cache / "onnx" / "image_encoder.onnx.data"), "models")]
if (_cache / "sharp_fp16.pt").exists():
    datas += [(str(_cache / "sharp_fp16.pt"), "models")]

# PySide6 Qt plugins
datas += collect_data_files("PySide6", subdir="Qt")

# ---- Package metadata (importlib.metadata / pkg_resources) ----
# These packages query their own version or discover plugins at runtime.
for pkg in [
    "imageio",
    "imageio-ffmpeg",
    "numpy",
    "Pillow",
    "torch",
    "torchvision",
    "onnxruntime-gpu",
    "gsplat",
    "huggingface_hub",
    "timm",
    "safetensors",
    "scipy",
    "plyfile",
    "pynvml",
    "nvidia-ml-py",
    "PySide6",
    "onnx",
    "onnxscript",
    "jaxtyping",
]:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# ---- Hidden imports ----
hiddenimports = [
    # sharp3d
    *collect_submodules("sharp3d"),
    # sharp (ml-sharp)
    *collect_submodules("sharp"),
    # scipy (including _external.array_api_compat.*)
    *collect_submodules("scipy"),
    # PyTorch
    "torch._C",
    "torch.cuda",
    "torch.nn.functional",
    "torch.autograd",
    # triton (torch.compile backend)
    *collect_submodules("triton"),
    # gsplat
    "gsplat",
    "gsplat.cuda",
    "gsplat.rendering",
    # PySide6
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # Others
    "numpy",
    "PIL",
    "imageio",
    "imageio_ffmpeg",
    "plyfile",
    "pynvml",
    "onnxruntime",
    "onnx",
    "onnxscript",
    "timm",
    "safetensors",
    "huggingface_hub",
    "jaxtyping",
]

# ---- Excludes ----
excludes = []

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[str(SRC), str(ML_SHARP_SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    exclude_binaries=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sharp3d",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    icon=str(ROOT / "assets" / "sharp3d_icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sharp3d-v2.0.0-beta-win64",
)
