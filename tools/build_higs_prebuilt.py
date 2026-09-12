"""Build prebuilt HiGS extensions so machines WITHOUT MSVC can use them.

gsplat loads two native extensions for the HiGS render path, each trying a
*prebuilt import* before falling back to a JIT build (which needs MSVC+nvcc):

  1. scene packing  -> top-level module   ``gsplat_scene_cuda``
  2. HiGS rasterizer -> package submodule ``gsplat.experimental.render.kernels.csrc``

This script JIT-builds both ONCE on a machine that has MSVC + CUDA toolkit,
then copies the resulting .pyd files to the locations the prebuilt imports
expect. After that, any machine with the same Python/torch/CUDA-runtime stack
(e.g. a copied venv or the PyInstaller bundle) imports them directly — no
compiler needed.

Note: the rasterizer must be *rebuilt under the name* ``csrc`` (not renamed
afterwards) because a .pyd's ``PyInit_<name>`` entry point must match its
module name.

Usage (plain PowerShell is fine — the VS developer env is injected
automatically via vswhere/VsDevCmd)::

    ..\\sharp3d-env\\Scripts\\python.exe tools\\build_higs_prebuilt.py
    # for distribution to machines with different NVIDIA GPUs:
    ..\\sharp3d-env\\Scripts\\python.exe tools\\build_higs_prebuilt.py --archs "7.5;8.0;8.6;8.9;9.0;12.0+PTX"
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def ensure_msvc_env() -> None:
    """Put cl.exe on PATH by importing the VS developer environment."""
    if shutil.which("cl"):
        print("MSVC: cl already on PATH")
        return
    vswhere = (Path(os.environ["ProgramFiles(x86)"])
               / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if not vswhere.exists():
        raise SystemExit("vswhere.exe not found — is Visual Studio installed?")
    vs_path = subprocess.check_output(
        [str(vswhere), "-latest", "-products", "*",
         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
         "-property", "installationPath"], text=True).strip()
    if not vs_path:
        raise SystemExit("No VS installation with C++ build tools found")
    vsdevcmd = Path(vs_path) / "Common7" / "Tools" / "VsDevCmd.bat"
    print(f"MSVC: importing environment from {vsdevcmd}")
    out = subprocess.check_output(
        f'cmd /s /c ""{vsdevcmd}" -arch=amd64 -host_arch=amd64 >nul 2>&1 && set"',
        shell=True, text=True, encoding="oem", errors="replace")
    for line in out.splitlines():
        key, sep, val = line.partition("=")
        if sep and key and not key.startswith("="):
            os.environ[key] = val
    if not shutil.which("cl"):
        raise SystemExit("cl.exe still not on PATH after VsDevCmd")
    print("MSVC: developer environment ready")


def build_scene(site_pkgs: Path) -> Path:
    """Build gsplat_scene_cuda (pure C++) and install the prebuilt .pyd."""
    import torch.utils.cpp_extension as jit
    from gsplat.scene.kernels.cuda.build import build_and_load_scene_cuda

    print("\n=== building gsplat_scene_cuda (scene packing, C++ only) ===")
    build_and_load_scene_cuda()
    build_dir = Path(jit._get_build_directory("gsplat_scene_cuda", verbose=False))
    pyd = build_dir / "gsplat_scene_cuda.pyd"
    if not pyd.exists():
        raise SystemExit(f"build finished but {pyd} not found")
    dst = site_pkgs / pyd.name
    shutil.copy2(pyd, dst)
    print(f"installed prebuilt: {dst}")
    return dst


def build_higs_csrc(site_pkgs: Path) -> Path:
    """Build the HiGS rasterizer under the module name ``csrc`` and install it."""
    import torch.utils.cpp_extension as jit
    from gsplat.experimental.render.kernels.cuda.build import (
        get_build_parameters,
    )

    print("\n=== building HiGS rasterizer as 'csrc' (CUDA, takes minutes) ===")
    bp = get_build_parameters()
    build_dir = Path(jit._get_build_directory("gsplat_higs_csrc", verbose=False))
    build_dir.mkdir(parents=True, exist_ok=True)
    jit.load(
        name="csrc",  # PyInit_csrc must match the prebuilt import name
        sources=bp.sources,
        extra_cflags=bp.extra_cflags,
        extra_cuda_cflags=bp.extra_cuda_cflags,
        extra_include_paths=bp.extra_include_paths,
        extra_ldflags=bp.extra_ldflags,
        build_directory=str(build_dir),
        verbose=False,
        is_python_module=False,  # just build; loading 'csrc' here would clash
    )
    pyd = build_dir / "csrc.pyd"
    if not pyd.exists():
        raise SystemExit(f"build finished but {pyd} not found")
    dst = (site_pkgs / "gsplat" / "experimental" / "render" / "kernels"
           / pyd.name)
    shutil.copy2(pyd, dst)
    print(f"installed prebuilt: {dst}")
    return dst


def verify(python: str) -> None:
    """Import both prebuilt extensions in a FRESH process (no JIT possible)."""
    code = (
        "import os;"
        # simulate a machine without MSVC: hide cl from the child
        "os.environ['PATH']=';'.join(p for p in os.environ['PATH'].split(';')"
        " if 'Visual Studio' not in p and 'MSVC' not in p);"
        "import gsplat_scene_cuda;"
        "from gsplat.experimental.render.kernels import _backend;"
        "assert _backend._get_backend() is not None, 'HiGS backend is None';"
        "import torch;"
        "assert hasattr(torch.ops.experimental,"
        " 'gaussian_render_inference_only'), 'op not registered';"
        "print('PREBUILT OK: scene + HiGS rasterizer import without a compiler')"
    )
    subprocess.run([python, "-c", code], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archs", default=None,
        help="TORCH_CUDA_ARCH_LIST for distribution builds, e.g. "
             '"7.5;8.0;8.6;8.9;9.0;12.0+PTX". Default: this machine\'s GPU only.')
    args = parser.parse_args()

    if args.archs:
        os.environ["TORCH_CUDA_ARCH_LIST"] = args.archs
        print(f"TORCH_CUDA_ARCH_LIST={args.archs}")

    ensure_msvc_env()

    import gsplat
    site_pkgs = Path(gsplat.__file__).resolve().parent.parent
    print(f"site-packages: {site_pkgs}")

    build_scene(site_pkgs)
    build_higs_csrc(site_pkgs)
    verify(sys.executable)

    print("\nDone. Ship these two files with the app to skip MSVC entirely:")
    print(f"  {site_pkgs / 'gsplat_scene_cuda.pyd'}")
    print(f"  {site_pkgs / 'gsplat/experimental/render/kernels/csrc.pyd'}")


if __name__ == "__main__":
    main()
