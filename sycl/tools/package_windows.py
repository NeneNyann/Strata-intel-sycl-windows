"""Package the native Windows SYCL engine using Strata's Windows release layout.

Run from a oneAPI/Visual Studio developer prompt:
    python sycl/tools/package_windows.py --build build-sycl-release \
        --vision-build build-vision-sycl-release --out dist

Produces strata-windows-x64-sycl.zip: strata.exe, strata-device.exe,
strata-vision.exe, BUILD.json, oneapi/bin DLLs and oneapi/licenses.
The Intel display driver is supplied by the user; no compiler is required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
ASSET = "strata-windows-x64-sycl.zip"
DYNAMIC = ("ur_loader.dll", "ur_adapter_level_zero.dll", "ur_adapter_level_zero_v2.dll", "umf.dll", "sycl-jit.dll")
STARTUP = ("ur_win_proxy_loader.dll", "libmmd.dll", "svml_dispmd.dll", "msvcp140.dll",
           "vcruntime140.dll", "vcruntime140_1.dll")


def imports(dumpbin: str, path: Path) -> list[str]:
    output = subprocess.run([dumpbin, "/nologo", "/dependents", str(path)],
                            capture_output=True, text=True, check=True).stdout
    return re.findall(r"^\s+([\w.+-]+\.dll)\s*$", output, re.M | re.I)


def runtime_roots() -> dict[str, Path]:
    base = Path(os.environ.get("ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI"))
    return {"compiler": Path(os.environ.get("CMPLR_ROOT") or base / "compiler" / "latest"),
            "mkl": Path(os.environ.get("MKLROOT") or base / "mkl" / "latest"),
            "tbb": Path(os.environ.get("TBBROOT") or base / "tbb" / "latest"),
            "tcm": Path(os.environ.get("TCM_ROOT") or base / "tcm" / "latest"),
            "dnnl": Path(os.environ.get("DNNLROOT") or base / "dnnl" / "latest")}


def package(build: Path, vision_build: Path, out: Path) -> Path:
    dumpbin = shutil.which("dumpbin")
    if not dumpbin:
        raise RuntimeError("dumpbin is missing: initialize the oneAPI/Visual Studio x64 environment")
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    if "STRATA_PORTABLE:BOOL=ON" not in cache:
        raise RuntimeError("release engine must be built with STRATA_PORTABLE=ON")
    vcache = (vision_build / "CMakeCache.txt").read_text(encoding="utf-8")
    if "STRATA_PORTABLE:BOOL=ON" not in vcache or "GGML_SYCL:BOOL=ON" not in vcache:
        raise RuntimeError("release vision must use STRATA_PORTABLE=ON and GGML_SYCL=ON")
    roots = runtime_roots()
    inventory = {p.name.lower(): p for root in roots.values() for p in (root / "bin").glob("*.dll")}
    redist = Path(os.environ.get("VCToolsRedistDir", ""))
    crt_dirs = sorted((redist / "x64").glob("Microsoft.VC*.CRT")) if redist.is_dir() else []
    if not crt_dirs:
        raise RuntimeError("VCToolsRedistDir is missing: initialize the Visual Studio x64 environment")
    inventory.update({p.name.lower(): p for p in crt_dirs[-1].glob("*.dll")})

    # MKL loads CPU-dispatched kernels by name, outside the PE import tables. Exclude MPI and debug libraries.
    dynamic = list(DYNAMIC)
    for p in (roots["mkl"] / "bin").glob("*.dll"):
        if re.match(r"mkl_(avx|core|def|mc|rt|sequential|intel_thread|tbb_thread|vml|sycl_blas)", p.name) and \
                not re.search(r"(?:blas|thread|vml)d\.\d", p.name):
            dynamic.append(p.name.lower())
    if "libimalloc.dll" in inventory:
        dynamic.append("libimalloc.dll")
    if "libiomp5md.dll" in inventory:
        dynamic.append("libiomp5md.dll")

    compiler_redist = roots["compiler"] / "share/doc/compiler/credist.txt"
    allowed = set(re.findall(r"<installdir>/bin/([^\s]+\.dll)",
                             compiler_redist.read_text(encoding="utf-8-sig"), re.I))
    allowed = {x.lower() for x in allowed}
    programs = [build / "strata.exe", build / "strata-device.exe", vision_build / "bin/strata-vision.exe"]
    for p in programs:
        if not p.is_file():
            raise RuntimeError(f"missing program: {p}")
    need, todo = {}, list(programs)
    for name in dynamic:
        if name not in inventory:
            raise RuntimeError(f"missing runtime DLL: {name}")
        need[name] = inventory[name]
        todo.append(inventory[name])
    system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    while todo:
        for name in imports(dumpbin, todo.pop()):
            low = name.lower()
            if low in need:
                continue
            if low in inventory:
                need[low] = inventory[low]
                todo.append(inventory[low])
            elif low.startswith(("api-ms-", "ext-ms-")) or (system / name).is_file():
                continue
            else:
                raise RuntimeError(f"unresolved DLL dependency: {name}")
    compiler_bin = (roots["compiler"] / "bin").resolve()
    for name, p in need.items():
        if p.parent.resolve() == compiler_bin and name not in allowed:
            raise RuntimeError(f"compiler DLL is not in credist.txt: {name}")

    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "strata-windows-x64-sycl"
    if stage.resolve().parent != out:
        raise RuntimeError("package staging path is outside the output directory")
    if stage.exists():
        shutil.rmtree(stage)
    rbin = stage / "oneapi/bin"
    licenses = stage / "oneapi/licenses"
    rbin.mkdir(parents=True)
    licenses.mkdir()
    for p in programs:
        shutil.copy2(p, stage / p.name)
    for name, p in sorted(need.items()):
        shutil.copy2(p, rbin / p.name)
        # Resolve the core SYCL runtime and adapters from the executable's own folder first.
        if name.startswith("sycl") or name in DYNAMIC or name in STARTUP:
            shutil.copy2(p, stage / p.name)
    license_files = [("compiler-redist.txt", compiler_redist),
                     ("compiler-LICENSE.rtf", roots["compiler"] / "share/doc/compiler/licensing/c/LICENSE.rtf"),
                     ("mkl-LICENSE.rtf", roots["mkl"] / "share/doc/mkl/licensing/license.rtf")]
    license_files.append(("Strata-LICENSE", ROOT / "LICENSE"))
    llama_dir = re.search(r"^LLAMA_DIR:PATH=(.+)$", vcache, re.M)
    if not llama_dir:
        raise RuntimeError("vision cache does not identify its pinned llama.cpp checkout")
    license_files.append(("llama.cpp-LICENSE", Path(llama_dir.group(1).strip()) / "LICENSE"))
    tbb_licenses = list(roots["tbb"].rglob("LICENSE.txt"))
    if not tbb_licenses:
        raise RuntimeError("TBB license file is missing")
    license_files.append(("tbb-LICENSE.txt", tbb_licenses[0]))
    if any(p.parent.resolve() == (roots["tcm"] / "bin").resolve() for p in need.values()):
        license_files.append(("tcm-LICENSE.txt", roots["tcm"] / "share/doc/tcm/licensing/license.txt"))
    if "dnnl.dll" in need:
        license_files.append(("oneDNN-LICENSE", roots["dnnl"] / "share/doc/dnnl/LICENSE"))
    for name, p in license_files:
        shutil.copy2(p, licenses / name)
    for component, root in roots.items():
        for p in (root / "share/doc").rglob("*third*party*"):
            if p.is_file():
                shutil.copy2(p, licenses / f"{component}-{p.name}")
    (licenses / "NOTICE.txt").write_text(
        "Intel oneAPI runtime components are shipped unmodified under their accompanying licenses.\n"
        "Microsoft Visual C++ runtime DLLs are redistributable under the Visual Studio license.\n"
        "The Intel graphics driver is not bundled.\n\n" + "\n".join(sorted(need)), encoding="utf-8")
    version = re.search(r"project\(strata VERSION ([\d.]+)",
                        (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")).group(1)
    meta = {"source": "prebuilt", "backend": "sycl", "platform": "windows-x64", "version": version,
            "archs": ["spir64"], "cpu_isa": "avx2", "vision": "gpu", "lib_dirs": ["oneapi/bin"],
            "compiler": roots["compiler"].resolve().name, "mkl": roots["mkl"].resolve().name}
    (stage / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    archive = out / ASSET
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(stage).as_posix())
    print(f"{archive}: {archive.stat().st_size / 2**20:.0f} MiB, engine {version}, {len(need)} runtime DLLs")
    return archive


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--vision-build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    try:
        package(a.build, a.vision_build, a.out)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        ap.exit(1, f"package_windows: {e}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
