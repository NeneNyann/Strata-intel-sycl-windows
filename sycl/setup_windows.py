"""Native Windows Intel installer: reuse setup.py's model/download/pack/launcher flow.

Called by START-HERE.bat through setup.py --backend sycl. Published engines use
the same engine/ directory, BUILD.json and rollback layout as CUDA/HIP.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup as S
from sycl.setup_intel import INTEL_ARC

ASSET = "strata-windows-x64-sycl.zip"
SYCL_WINDOWS_PREBUILT_URL = "https://github.com/NeneNyann/Strata-intel-sycl-windows/releases/latest/download/"


def prebuilt_bases(url_base: str) -> list[str]:
    base = url_base if url_base.endswith(("/", "\\")) else url_base + "/"
    default = SYCL_WINDOWS_PREBUILT_URL.rstrip("/\\") + "/"
    suffix = "/releases/latest/download/"
    if base == default and base.startswith("https://github.com/") and base.endswith(suffix):
        repo = base[:-len(suffix)]
        return [f"{repo}/releases/download/v{S.source_version()}/", base]
    return [base]


def intel_gpus() -> list[dict]:
    """Known discrete Arc adapters; WMI's 32-bit AdapterRAM is not their VRAM size."""
    script = "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,DriverVersion | ConvertTo-Json -Compress"
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30, check=True)
        adapters = json.loads(result.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    if isinstance(adapters, dict):
        adapters = [adapters]
    found = []
    for adapter in adapters or []:
        match = re.search(r"VEN_8086&DEV_([0-9A-F]{4})", adapter.get("PNPDeviceID", ""), re.I)
        if not match or match.group(1).lower() not in INTEL_ARC:
            continue
        name, vram = INTEL_ARC[match.group(1).lower()]
        found.append({"index": len(found), "name": f"Intel {name}", "vram_gb": vram, "arch": "xe",
                      "driver": adapter.get("DriverVersion", ""), "vendor": "intel"})
    return found


def engine_env(eng: Path) -> dict:
    env = dict(os.environ)
    dirs = S.hip_lib_dirs(eng)  # common BUILD.json lib_dirs format
    env["PATH"] = os.pathsep.join([str(eng), *(str(d) for d in dirs), env.get("PATH", "")])
    env.pop("ONEAPI_DEVICE_SELECTOR", None)
    return env


def devices(eng: Path) -> list[dict]:
    try:
        result = subprocess.run([str(eng / "strata-device.exe")], env=engine_env(eng), capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60, check=True)
        return json.loads(result.stdout)["devices"]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as e:
        S.fail(f"the Intel engine cannot enumerate Level Zero devices: {e}",
               "install a current Intel graphics driver, then run START-HERE.bat again")


def normalized_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("(r)", "").replace("(tm)", "").replace("graphics", ""))


def select_device(eng: Path, card: dict) -> dict:
    name = normalized_name(card["name"])
    matches = [d for d in devices(eng) if not d.get("integrated") and name in normalized_name(d["name"])]
    if not matches:
        S.fail(f"{card['name']} is not exposed through Level Zero", "check the Intel graphics driver")
    # Same-name cards are offered in WMI order; retain their occurrence when possible.
    peers = [g for g in intel_gpus() if g["name"] == card["name"]]
    rank = next((i for i, g in enumerate(peers) if g["index"] == card["index"]), 0)
    if rank >= len(matches):
        S.fail("the selected Intel adapter is missing from Level Zero")
    d = matches[rank]
    return {**card, "sycl_index": d["index"], "vram_gb": d["memory_bytes"] / 2**30}


def metadata(eng: Path) -> dict:
    try:
        return json.loads((eng / "BUILD.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def valid_package(eng: Path) -> bool:
    meta = metadata(eng)
    version = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:3] if x.isdigit())
    return meta.get("backend") == "sycl" and meta.get("platform") == "windows-x64" and \
        version >= S.MIN_ENGINE and all((eng / name).is_file() for name in
                                       ("strata.exe", "strata-device.exe", "strata-vision.exe"))


def get_prebuilt(url_base, gpu, updating=False) -> Path | None:
    eng = ROOT / "engine"
    if valid_package(eng) and not updating:
        S.ok("ready-made Intel engine already installed")
        return eng
    if not url_base:
        return None
    eng.mkdir(exist_ok=True)
    bases = prebuilt_bases(url_base)
    base = None
    for candidate in bases:
        if candidate.startswith(("http://", "https://")):
            try:
                req = urllib.request.Request(candidate + ASSET, method="HEAD", headers={"User-Agent": "strata-setup"})
                urllib.request.urlopen(req, timeout=60).close()
            except OSError:
                continue
        elif not Path(candidate + ASSET).is_file():
            continue
        base = candidate
        break
    if base is None:
        S.warn("no Windows Intel release zip at the selected release/mirror")
        return None
    archive = eng / ASSET
    S.drop_archive(archive)
    S.download(base + ASSET, archive, "Strata Intel engine")
    # TemporaryDirectory avoids reusing a partial unpack and stays inside the standard engine folder.
    with tempfile.TemporaryDirectory(prefix="_sycl-unpack-", dir=eng) as temp:
        unpacked = Path(temp)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                target = (unpacked / name).resolve()
                if not target.is_relative_to(unpacked.resolve()):
                    S.fail("the Intel engine zip contains a path outside its package")
            z.extractall(unpacked)
        if not valid_package(unpacked):
            S.fail("the zip is not a compatible Windows SYCL engine", "use the zip from this release or a newer one")
        S.install_unpacked(unpacked, eng)
    S.drop_archive(archive)
    S.ok(f"ready-made Intel engine {metadata(eng).get('version')}")
    return eng


def vision_choice(asked):
    if asked in ("no", "none"):
        return "none"
    if asked in ("yes", "gpu", "cpu"):
        return "gpu" if asked == "yes" else asked
    return "gpu" if S.ask("Do you want images?", ["y", "n"], "n", YES) == "y" else "none"


def to_native(cfg: dict, card: dict) -> dict:
    out = dict(cfg)
    out["backend"] = "sycl"
    out.pop("cuda", None)
    out.pop("gpus_asked", None)
    out.pop("layer_split", None)
    env = dict(out.get("env") or {})
    env.pop("STRATA_HIPBLASLT_TUNING", None)
    env.update({"ONEAPI_DEVICE_SELECTOR": f"level_zero:{card['sycl_index']}",
                "STRATA_RESIDENT_PIN": "0", "STRATA_VERIFY_DEVICE_PLAN": "1"})
    out["env"] = env
    out["gpu"] = card["index"]
    return out


def update_selected_gpu(cfg_path: Path, cfg: dict, gpu=None) -> dict:
    if isinstance(gpu, list):
        if len(gpu) != 1:
            S.fail("native Windows SYCL currently supports one selected GPU")
        gpu = gpu[0]
    selected = gpu if gpu is not None else cfg.get("gpu")
    if isinstance(selected, list):
        if len(selected) != 1:
            S.fail("native Windows SYCL currently supports one selected GPU")
        selected = selected[0]
    cards = intel_gpus()
    card = next((g for g in cards if g["index"] == selected), None) if selected is not None else \
        max(cards, key=lambda g: g["vram_gb"], default=None)
    if card is None:
        S.fail("the configured Intel Arc was not found", "use START-HERE.bat --setup --backend sycl to select it again")
    cfg = to_native(cfg, select_device(Path(cfg["exe"]).parent, card))
    S.write_config(cfg_path, cfg)
    return cfg


YES = False


def install(argv) -> int:
    """Replace only the GPU/engine/config steps; keep setup's real RAM and all model steps."""
    global YES
    YES = "--yes" in argv or "--check" in argv
    cards = intel_gpus()
    if not cards:
        S.fail("no supported discrete Intel Arc found in Windows", "check the Intel graphics driver")
    rest = []
    iterator = iter(argv)
    for arg in iterator:
        if arg == "--gpus" or arg.startswith("--gpus="):
            value = next(iterator, "") if arg == "--gpus" else arg.split("=", 1)[1]
            if value == "all" and len(cards) == 1:
                value = str(cards[0]["index"])
            if not value.isdigit():
                S.fail("native Windows SYCL currently supports one selected GPU; use --gpu N")
            rest += ["--gpu", value]
        else:
            rest.append(arg)
    def option(name):
        for i, arg in enumerate(rest):
            if arg == name:
                return rest[i + 1] if i + 1 < len(rest) else None
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
        return None
    if option("--kv") == "k8v4":
        if option("--kv-streaming") == "on":
            S.fail("the SYCL engine does not support k8v4 KV streaming", "use --kv-streaming off or --kv int8")
        if option("--kv-streaming") is None:
            S.warn("k8v4 KV streaming is unsupported on SYCL; keeping KV in VRAM (--kv-streaming off)")
            rest += ["--kv-streaming", "off"]
    if "--build" in rest and "--check" not in rest:
        rc = subprocess.call(["cmd", "/c", str(ROOT / "sycl/build-release-windows.bat")], cwd=ROOT)
        if rc:
            return rc
        rest.remove("--build")
        if not any(arg == "--prebuilt" or arg.startswith("--prebuilt=") for arg in rest):
            rest += ["--prebuilt", str(ROOT / "dist")]

    if not any(arg == "--prebuilt" or arg.startswith("--prebuilt=") for arg in rest) and \
            "STRATA_PREBUILT_URL" not in os.environ:
        rest += ["--prebuilt", SYCL_WINDOWS_PREBUILT_URL]

    real_say, real_fail = S.say, S.fail
    def say_intel(message=""):
        message = str(message).replace("Your AMD GPUs:", "Your Intel GPUs:")
        message = message.replace("(AMD, experimental: docs/AMD_HIP.md)", "(Intel Arc: docs/SYCL_WINDOWS.md)")
        message = message.replace("(AMD: docs/AMD_HIP.md)", "(Intel Arc: docs/SYCL_WINDOWS.md)")
        message = message.replace("untested on AMD", "untested on Intel Arc")
        real_say(message)
    def fail_intel(message, *details):
        if "no ready-made AMD engine" in str(message):
            real_fail("no compatible Windows Intel engine zip was found",
                      "use --prebuilt <release zip directory>, or --build with oneAPI installed; docs/SYCL_WINDOWS.md")
        if "older-CPU engine" in str(message):
            real_fail("the Windows Intel release engine requires an AVX2 CPU")
        real_fail(message, *details)
    S.say, S.fail = say_intel, fail_intel
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: cards
    S.amd_problem = lambda g: None
    S.hip_vision = vision_choice
    S.get_prebuilt_hip = get_prebuilt
    S.hipblaslt_table = lambda *a, **k: None
    chosen = {}
    def hip_card(eng, gpu, listed):
        card = select_device(eng, gpu)
        chosen.update(card)
        return {**card, "count": len(cards)}
    S.hip_card = hip_card
    write = S.write_run_script
    def write_run_script(model, cfg_path, port, open_browser=True):
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        cfg = to_native(cfg, chosen)
        S.write_config(Path(cfg_path), cfg)
        return write(model, cfg_path, port, open_browser)
    S.write_run_script = write_run_script
    def update_installed_engine(url_base, toolkit=None):
        eng = ROOT / "engine"
        installed = metadata(eng)
        def version(text):
            return tuple(int(x) for x in str(text).split(".")[:3] if x.isdigit())
        if valid_package(eng) and version(installed.get("version", "0")) >= version(S.source_version()):
            return
        try:
            if get_prebuilt(url_base, cards[0], updating=True) is None and installed:
                S.warn("the Intel engine update is unavailable; keeping the installed one")
        except (Exception, SystemExit) as e:
            if not installed:
                raise
            S.warn(f"could not update the Intel engine ({e}); keeping the installed one")
    S.update_installed_engine = update_installed_engine
    # Use the normal native GGUF resident path, with the real RAM rules. Explicit user choices still win.
    if not any(x == "--low-ram" or x.startswith("--low-ram=") for x in rest):
        rest += ["--low-ram", "resident"]
    sys.argv = [sys.argv[0], *rest, "--backend", "hip"]
    return S.main()


if __name__ == "__main__":
    try:
        sys.exit(install(sys.argv[1:]))
    except KeyboardInterrupt:
        S.say("\nstopped.")
        sys.exit(1)
