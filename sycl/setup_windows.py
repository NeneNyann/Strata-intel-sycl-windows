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
from strata_control.targets import (INTEL_ARC, intel_device_id, intel_name,
                                    intel_pci_address, intel_uuid, match_intel_devices)

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
    script = r"""[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
Get-CimInstance Win32_VideoController | ForEach-Object {
    $adapter = $_
    $bus = $address = $null
    try {
        $p = Get-PnpDeviceProperty -InstanceId $adapter.PNPDeviceID -KeyName DEVPKEY_Device_BusNumber,DEVPKEY_Device_Address -ErrorAction Stop
        $bus = ($p | Where-Object KeyName -eq DEVPKEY_Device_BusNumber).Data
        $address = ($p | Where-Object KeyName -eq DEVPKEY_Device_Address).Data
    } catch {}
    [PSCustomObject]@{Name=$adapter.Name; PNPDeviceID=$adapter.PNPDeviceID; DriverVersion=$adapter.DriverVersion; PciBus=$bus; PciAddress=$address}
} | ConvertTo-Json -Compress"""
    try:
        result = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30, check=True)
        adapters = json.loads(result.stdout or "[]")
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    if isinstance(adapters, dict):
        adapters = [adapters]
    if not isinstance(adapters, list):
        return []
    from strata_control.engine_component import windows_display_adapters, windows_intel_cards
    registry = windows_display_adapters()
    found = []
    for adapter in adapters or []:
        if not isinstance(adapter, dict):
            continue
        match = re.search(r"VEN_8086&DEV_([0-9A-F]{4})", str(adapter.get("PNPDeviceID") or ""), re.I)
        if not match or not windows_intel_cards([{"vendor": "8086", "device": int(match.group(1), 16),
                                                 "name": adapter.get("Name", "")} ]):
            continue
        devid = match.group(1).lower()
        name, vram = INTEL_ARC.get(devid, (None, 0.0))
        actual = [a for a in registry if a.get("vendor") == "8086" and intel_device_id(a.get("device")) == int(devid, 16)]
        if len(actual) == 1 and actual[0].get("vram_gb", 0) > 0:
            vram = actual[0]["vram_gb"]
        card = {"index": len(found), "name": f"Intel {name}" if name else adapter["Name"], "vram_gb": vram,
                "arch": "xe", "driver": adapter.get("DriverVersion", ""), "vendor": "intel", "devid": devid,
                "instance_id": adapter["PNPDeviceID"]}
        bus, address = adapter.get("PciBus"), adapter.get("PciAddress")
        if isinstance(bus, int) and isinstance(address, int) and 0 <= bus <= 255 and 0 <= address >> 16 <= 31 and 0 <= address & 0xffff <= 7:
            card["pci_address"] = f"{bus:02x}:{address >> 16:02x}.{address & 0xffff}"
        found.append(card)
    return found


def engine_env(eng: Path) -> dict:
    env = dict(os.environ)
    dirs = S.hip_lib_dirs(eng)  # common BUILD.json lib_dirs format
    env["PATH"] = os.pathsep.join([str(eng), *(str(d) for d in dirs), env.get("PATH", "")])
    env.pop("ONEAPI_DEVICE_SELECTOR", None)
    env.pop("SYCL_DEVICE_FILTER", None)
    return env


def devices(eng: Path) -> list[dict]:
    try:
        result = subprocess.run([str(eng / "strata-device.exe")], env=engine_env(eng), capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=60, check=True)
        rows = json.loads(result.stdout)["devices"]
        if not isinstance(rows, list) or any(not isinstance(d, dict) or type(d.get("index")) is not int or d["index"] < 0 for d in rows):
            raise ValueError("invalid device list")
        return rows
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as e:
        S.fail(f"the Intel engine cannot enumerate Level Zero devices: {e}",
               "install a current Intel graphics driver, then run START-HERE.bat again")


def normalized_name(name: str) -> str:
    return intel_name(name)


def select_device(eng: Path, card: dict) -> dict:
    matches = match_intel_devices(devices(eng), device_id=card.get("devid"), name=card["name"],
                                  pci_address=card.get("pci_address"))
    if not matches:
        S.fail(f"{card['name']} is not exposed through Level Zero", "check the Intel graphics driver")
    peers = [g for g in intel_gpus() if (g.get("devid") == card.get("devid") if card.get("devid")
                                        else intel_name(g["name"]) == intel_name(card["name"]))]
    if len(matches) != 1 or (len(peers) > 1 and not (intel_pci_address(card.get("pci_address"))
                                                   and intel_pci_address(matches[0].get("pci_address")))):
        S.fail("the selected Intel adapter cannot be identified uniquely",
               "use the current Windows SYCL engine package so setup can match each card's PCI address")
    d = matches[0]
    if type(d.get("memory_bytes")) is not int or d["memory_bytes"] <= 0:
        S.fail("the Intel engine did not report this card's VRAM size", "update the Intel graphics driver")
    if d.get("usm_host") is False or d.get("usm_device") is False:
        S.fail("this Intel adapter does not expose the memory capabilities the SYCL engine needs",
               "update the Intel graphics driver")
    selected = {**card, "sycl_index": d["index"], "vram_gb": d["memory_bytes"] / 2**30,
                "uuid": intel_uuid(d.get("uuid")), "pci_address": d.get("pci_address") or card.get("pci_address")}
    if isinstance(d.get("alchemist"), bool):
        selected["alchemist"] = d["alchemist"]
    return selected


def metadata(eng: Path) -> dict:
    try:
        return json.loads((eng / "BUILD.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}


def valid_package(eng: Path) -> bool:
    meta = metadata(eng)
    version = tuple(int(x) for x in str(meta.get("version", "0")).split(".")[:4] if x.isdigit())
    return meta.get("backend") == "sycl" and meta.get("platform") == "windows-x64" and \
        version + (0,) * (4 - len(version)) >= S.MIN_ENGINE and all((eng / name).is_file() for name in
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


def vision_choice(asked, vram_gb: float = 0.0):
    if asked in ("no", "none", "off"):
        return "none"
    if asked in ("yes", "gpu", "cpu"):
        return "gpu" if asked == "yes" else asked
    return "gpu" if S.ask("Do you want images?", ["y", "n"], "n", YES) == "y" else "none"


def vision_reader(eng: Path, vision: str, gpu: dict):
    """The Windows SYCL reader runs on either device: --gpu selects Intel, otherwise CPU.

    Its release name stays strata-vision.exe. The shared AMD selector would call it
    CPU-only because it looks for strata-vision-hip.exe instead.
    """
    from strata_control import vision_reader as VR
    exe = eng / "strata-vision.exe"
    if vision in ("no", "none", "off", ""):
        reader = VR.Reader("none")
    elif not exe.is_file():
        reader = VR.Reader("none", note="this engine has no picture reader: pictures off")
    elif vision == "cpu":
        reader = VR.Reader("cpu", exe)
    elif metadata(eng).get("vision") == "gpu":
        reader = VR.Reader("gpu", exe, exe)
    else:
        reader = VR.Reader("cpu", exe, note="this engine has only the CPU picture reader")
    if reader.note:
        S.warn(reader.note)
    return reader.mode, reader


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
    env.pop("SYCL_DEVICE_FILTER", None)
    if card.get("alchemist", bool(re.fullmatch(r"arc(?:pro)?a\d{2,3}m?", intel_name(card["name"])))):
        for key in ("IGC_EnableDPEmulation", "OverrideDefaultFP64Settings", "NEOReadDebugKeys"):
            env.setdefault(key, "1")
    out["env"] = env
    out["gpu"] = card["index"]
    out["gpu_name"] = card["name"]
    if card.get("devid"):
        out["gpu_device_id"] = card["devid"]
    for key, value in (("gpu_uuid", card.get("uuid")), ("gpu_instance_id", card.get("instance_id")),
                       ("gpu_pci_address", card.get("pci_address"))):
        if value:
            out[key] = value
        else:
            out.pop(key, None)
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
    if gpu is not None:
        card = next((g for g in cards if str(g["index"]) == str(selected)), None)
    elif cfg.get("gpu_instance_id"):
        card = next((g for g in cards if g.get("instance_id", "").lower() == cfg["gpu_instance_id"].lower()), None)
    else:
        matches = [g for g in cards if not cfg.get("gpu_device_id") or
                   intel_device_id(g.get("devid")) == intel_device_id(cfg["gpu_device_id"])]
        if cfg.get("gpu_uuid"):
            target = match_intel_devices(devices(Path(cfg["exe"]).parent), uuid=cfg["gpu_uuid"])
            matches = [g for g in matches if len(target) == 1 and len(match_intel_devices(
                target, device_id=g.get("devid"), name=g["name"], pci_address=g.get("pci_address"))) == 1]
        if cfg.get("gpu_name") and not cfg.get("gpu_device_id"):
            matches = [g for g in matches if intel_name(g["name"]) == intel_name(cfg["gpu_name"])]
        if not any(cfg.get(k) for k in ("gpu_uuid", "gpu_device_id", "gpu_name")) and selected is not None:
            matches = [g for g in matches if str(g["index"]) == str(selected)] if len(cards) == 1 else []
        card = matches[0] if len(matches) == 1 else None
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
    if option("--family") in ("qwen36", "nvfp4", "gyro"):
        S.fail("Qwen3.6, NVFP4 and Gyro are not supported by the Windows SYCL engine",
               "choose a Flash-Next model; docs/SYCL_WINDOWS.md")
    if option("--kv") == "k8v4":
        if option("--kv-streaming") == "on":
            S.fail("the SYCL engine does not support k8v4 KV streaming", "use --kv-streaming off or --kv int8")
        if option("--kv-streaming") is None:
            S.warn("k8v4 KV streaming is unsupported on SYCL; keeping KV in VRAM (--kv-streaming off)")
            rest += ["--kv-streaming", "off"]
    if option("--parallel") not in (None, "1", "auto"):
        S.fail("the Windows Intel engine currently serves one request at a time")
    if option("--parallel") is None:
        rest += ["--parallel", "1"]
    elif option("--parallel") == "auto":
        for i, arg in enumerate(rest):
            if arg == "--parallel":
                rest[i + 1] = "1"
            elif arg == "--parallel=auto":
                rest[i] = "--parallel=1"
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
    model_families = S.menu_families
    S.menu_families = lambda hip: [f for f in model_families(hip) if f not in S.OWN_PATH_FAMILIES and f != "gyro"]
    S.gpus = lambda *a, **k: []
    S.amd_gpus = lambda *a, **k: cards
    S.amd_problem = lambda g: None
    S.hip_vision = vision_choice
    S.hip_reader = vision_reader
    S.get_prebuilt_hip = get_prebuilt
    S.hipblaslt_table = lambda *a, **k: None
    chosen = {}
    def hip_card(eng, gpu, listed):
        card = select_device(eng, gpu)
        chosen.update(card)
        return {**card, "count": len(cards)}
    S.hip_card = hip_card
    if S.CONTROL:
        control_engine_step = S.control_engine_step
        def control_step(gpu, vision):
            llama, eng, meta, lib_dirs, gpu, vision = control_engine_step(gpu, vision)
            if (eng / "strata-device.exe").is_file():
                gpu = hip_card(eng, gpu, cards)
                vision, _reader = vision_reader(eng, vision, gpu)
            else:
                chosen.update({**gpu, "sycl_index": gpu["index"]})
            return llama, eng, meta, lib_dirs, gpu, vision
        S.control_engine_step = control_step
    write_setup_config = S.write_setup_config
    def write_native_config(cfg_path, cfg, *args, **kwargs):
        return write_setup_config(cfg_path, to_native(cfg, chosen), *args, **kwargs)
    S.write_setup_config = write_native_config
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
            return tuple(int(x) for x in str(text).split(".")[:4] if x.isdigit())
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
