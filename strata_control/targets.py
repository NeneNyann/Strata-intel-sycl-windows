"""Which OS and backend combinations the project commits to, and what this machine is.

Pure and tiny on purpose: the engine resolver on the start path (engine_component.py) needs these answers, and the
start path must not import the component fetcher (components.py) to get them.
"""

from __future__ import annotations

import platform
import re


class ComponentError(RuntimeError):
    """A component could not be used. The message names the reason and nothing was changed."""

    def __init__(self, message: str, code: int = 6) -> None:
        super().__init__(message)
        self.code = code


#: The OS × backend families the handoff commits to. A component may only be activated where its
#: family is listed here, so an unqualified combination is refused rather than attempted: the packet
#: includes this fork's native Windows SYCL engine; Intel Macs remain unsupported.
BACKEND_MATRIX: dict[str, tuple[str, ...]] = {
    "metal": ("macos-arm64",),
    "cuda": ("windows-x86_64", "linux-x86_64"),
    "hip": ("windows-x86_64", "linux-x86_64"),
    "sycl": ("windows-x86_64", "linux-x86_64"),
    "cpu": ("macos-arm64", "macos-x86_64", "windows-x86_64", "linux-x86_64"),
}

# Names and initial VRAM estimates shared by Windows and Linux setup. The runtime probe supplies the actual VRAM.
INTEL_ARC = {"e223": ("Arc Pro B70", 32.0), "e221": ("Arc Pro B60", 24.0), "e211": ("Arc Pro B60", 24.0),
             "e20b": ("Arc B580", 12.0), "e20c": ("Arc B570", 10.0), "e212": ("Arc Pro B50", 16.0),
             "56a0": ("Arc A770", 16.0), "56a1": ("Arc A750", 8.0), "56a2": ("Arc A580", 8.0),
             "56a5": ("Arc A380", 6.0), "56a6": ("Arc A310", 4.0), "5690": ("Arc A770M", 16.0)}
WIN_INTEL_ARC_IDS = frozenset(int(pci, 16) for pci in INTEL_ARC)


def intel_device_id(value) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 0 <= value <= 0xffff else None
    value = str(value or "").strip().lower()
    return int(value, 16) if re.fullmatch(r"(?:0x)?[0-9a-f]{1,4}", value) else None


def intel_name(value) -> str:
    value = re.sub(r"\(r\)|\(tm\)|®|™|graphics|intel", "", str(value or ""), flags=re.I)
    return re.sub(r"[^a-z0-9]", "", value.lower())


def intel_uuid(value) -> str | None:
    text = str(value or "").replace("-", "").lower()
    return text if re.fullmatch(r"[0-9a-f]{32}", text) and int(text, 16) else None


def intel_pci_address(value) -> str | None:
    text = str(value or "").strip().lower()
    return text if re.fullmatch(r"(?:[0-9a-f]{4}:)?[0-9a-f]{2}:[0-1][0-9a-f]\.[0-7]", text) else None


def match_intel_devices(devices, *, device_id=None, name=None, uuid=None, pci_address=None) -> list[dict]:
    """Match hardware identities, not enumeration positions or name substrings. Callers require one result."""
    devices = [d for d in devices if isinstance(d, dict) and d.get("integrated") is False
               and d.get("vendor_id", 0x8086) == 0x8086]
    if uuid is not None:
        identity = intel_uuid(uuid)
        return [d for d in devices if identity and intel_uuid(d.get("uuid")) == identity]
    if pci_address is not None and any(intel_pci_address(d.get("pci_address")) for d in devices):
        address = intel_pci_address(pci_address)
        def same(d):
            other = intel_pci_address(d.get("pci_address"))
            return address and other and (address == other if address.count(":") == other.count(":")
                                         else address[-7:] == other[-7:])
        devices = [d for d in devices if same(d)]
    if device_id is not None:
        wanted = intel_device_id(device_id)
        aliases = {intel_name(n) for pci, (n, _) in INTEL_ARC.items() if int(pci, 16) == wanted}
        if name:
            aliases.add(intel_name(name))
        # Older probes only report the name: permit an exact unique alias, never a substring.
        devices = [d for d in devices if wanted is not None and
                   (intel_device_id(d["device_id"]) == wanted if intel_device_id(d.get("device_id")) is not None
                    else intel_name(d.get("name")) in aliases)]
    elif name:
        aliases = {int(pci, 16) for pci, (n, _) in INTEL_ARC.items() if intel_name(n) == intel_name(name)}
        devices = [d for d in devices if intel_name(d.get("name")) == intel_name(name)
                   or intel_device_id(d.get("device_id")) in aliases]
    return devices


def host_platform() -> str:
    """This machine as an `os-arch` key, matching `BACKEND_MATRIX`."""
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(platform.system())
    if system is None:
        raise ComponentError(f"unsupported operating system: {platform.system()}")
    machine = platform.machine().lower()
    arch = {"arm64": "arm64", "aarch64": "arm64",
            "x86_64": "x86_64", "amd64": "x86_64"}.get(machine, machine)
    return f"{system}-{arch}"


def backend_supported(backend: str, target: str | None = None) -> tuple[bool, str]:
    """Whether a backend family is committed to on a platform, with the reason if not."""
    target = target or host_platform()
    if backend not in BACKEND_MATRIX:
        return False, f"unknown backend family {backend!r}"
    platforms = BACKEND_MATRIX[backend]
    if target in platforms:
        return True, ""
    return False, (f"{backend} is not a supported combination on {target}; it is supported on "
                   f"{', '.join(platforms)}. Unsupported combinations are not attempted.")
