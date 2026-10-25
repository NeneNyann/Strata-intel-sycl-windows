"""Tests for setup.py --backend sycl: native Windows release setup and Linux source/container setup. The CUDA / HIP
paths never see it. No GPU, no network.

    python -m unittest tools.test_setup_sycl
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

def run(argv, win):
    out = io.StringIO()
    with mock.patch.object(setup, "WIN", win), mock.patch.object(setup.subprocess, "call", return_value=0) as call, \
            contextlib.redirect_stdout(out):
        try:
            rc = setup.sycl_setup(argv)
        except SystemExit as e:
            rc = ("exit", e.code)
    return rc, out.getvalue(), call


class SyclBackend(unittest.TestCase):
    def test_windows_hands_over_to_native_setup(self):
        rc, out, call = run(["--backend", "sycl"], win=True)
        self.assertEqual(rc, 0)
        self.assertIn("Windows uses strata-windows-x64-sycl.zip", out)
        self.assertIn("strata-windows-x64-sycl.zip", out)
        self.assertTrue(call.call_args[0][0][1].endswith(str(Path("sycl") / "setup_windows.py")))
        self.assertEqual(call.call_args[0][0][2:], [])

    def test_linux_hands_over_to_setup_intel_without_the_backend_flag(self):
        rc, out, call = run(["--model", "IQ2_XS", "--backend", "sycl", "--port", "8085"], win=False)
        self.assertEqual(rc, 0)
        self.assertIn("supported since 0.1.40.2", out)
        self.assertIn("built from source", out)
        cmd = call.call_args[0][0]
        self.assertTrue(cmd[1].endswith(str(Path("sycl") / "setup_intel.py")))
        self.assertEqual(cmd[2:], ["--model", "IQ2_XS", "--port", "8085"])

    def test_backend_equals_form_is_dropped_too(self):
        _, _, call = run(["--backend=sycl", "--check"], win=False)
        self.assertEqual(call.call_args[0][0][2:], ["--check"])

    def test_the_parser_accepts_sycl(self):
        src = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertRegex(src, r'"--backend", choices=\["cuda", "hip", "sycl"\]')

    def test_setup_intel_finds_the_setup_functions_it_replaces(self):
        """sycl/setup_intel.py swaps these steps; setup.py must still have them (it stops with a message otherwise)."""
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        names = re.search(r'for name in \(([^)]*)\):', src).group(1)
        for name in re.findall(r'"(\w+)"', names):
            self.assertTrue(callable(getattr(setup, name, None)), name)

    def test_setup_intel_replacements_take_every_argument_of_the_function_they_replace(self):
        """#870: setup.py grew write_run_script's fourth argument and the port's replacement still took three, so
        setup died with a TypeError after the download.  Every `S.<name> = <replacement>` in setup_intel.py must accept
        all the parameters of setup.<name> (or take *args / **kwargs)."""
        import ast
        import inspect
        tree = ast.parse((ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8"))
        defs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        checked = 0
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Attribute)
                    and isinstance(n.targets[0].value, ast.Name) and n.targets[0].value.id == "S"):
                continue
            name = n.targets[0].attr
            fn = n.value if isinstance(n.value, ast.Lambda) else defs.get(getattr(n.value, "id", ""))
            orig = getattr(setup, name, None)
            if fn is None or not callable(orig):
                continue
            a = fn.args
            if a.vararg is not None and a.kwarg is not None:
                checked += 1
                continue
            have = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            for param in inspect.signature(orig).parameters.values():
                if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                    continue
                self.assertIn(param.name, have, f"setup_intel's {name} lacks setup.{name}'s `{param.name}`")
            checked += 1
        self.assertGreaterEqual(checked, 8)

    def test_arc_pro_b60_is_known_under_both_pci_ids(self):
        from strata_control.targets import INTEL_ARC, WIN_INTEL_ARC_IDS
        from sycl import setup_intel
        self.assertIs(setup_intel.INTEL_ARC, INTEL_ARC)
        for pci in ("e211", "e221"):
            self.assertEqual(INTEL_ARC[pci], ("Arc Pro B60", 24.0))
            self.assertIn(int(pci, 16), WIN_INTEL_ARC_IDS)


class WindowsPackageVersion(unittest.TestCase):
    def test_hotfix_versions_are_compared_in_full(self):
        from sycl import setup_windows as windows
        with tempfile.TemporaryDirectory() as folder:
            eng = Path(folder)
            for name in ("strata.exe", "strata-device.exe", "strata-vision.exe"):
                (eng / name).touch()
            with mock.patch.object(setup, "MIN_ENGINE", (0, 1, 40, 4)):
                for version, accepted in (("0.1.40", False), ("0.1.40.3", False),
                                          ("0.1.40.4", True), ("0.1.40.5", True), ("0.1.41", True)):
                    with self.subTest(version=version):
                        (eng / "BUILD.json").write_text(json.dumps({"backend": "sycl", "platform": "windows-x64",
                                                                   "version": version}), encoding="utf-8")
                        self.assertEqual(windows.valid_package(eng), accepted)


class WindowsUnsupportedFamilies(unittest.TestCase):
    def test_cuda_only_families_are_refused_before_installing(self):
        from sycl import setup_windows as windows
        for family in ("qwen36", "nvfp4", "gyro"):
            with self.subTest(family=family), mock.patch.object(windows, "intel_gpus", return_value=[{"index": 0}]), \
                    mock.patch.object(setup, "fail", side_effect=SystemExit(1)) as fail:
                with self.assertRaises(SystemExit):
                    windows.install(["--family", family, "--yes"])
                self.assertIn("not supported", fail.call_args[0][0])


class CMakeOption(unittest.TestCase):
    def test_sycl_is_off_by_default_and_exclusive(self):
        src = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8-sig")
        self.assertIn('option(STRATA_ENABLE_SYCL "EXPERIMENTAL', src)
        self.assertRegex(src, r'option\(STRATA_ENABLE_SYCL "[^"]*" OFF\)')
        self.assertIn("add_subdirectory(sycl)", src)


class SyclVersion(unittest.TestCase):
    def setUp(self):
        from sycl import setup_intel
        self.mod = setup_intel

    def test_version_matches_the_source_and_satisfies_native_models(self):
        version = self.mod.sycl_version()
        self.assertNotEqual(version, "0")
        self.assertEqual(version, setup.source_version())
        numbers = tuple(int(x) for x in version.split("."))
        for model in ("UD-IQ4_XS", "UD-Q4_K_XL"):
            with self.subTest(model=model):
                self.assertGreaterEqual(numbers, setup.MODELS[model].get("engine", setup.UNSLOTH_ENGINE))

    def test_version_follows_a_source_hotfix_with_a_variable_sycl_project(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.41.2 LANGUAGES CXX)\n", encoding="utf-8")
            (root / "sycl").mkdir()
            (root / "sycl" / "CMakeLists.txt").write_text(
                "set(STRATA_SYCL_VERSION 0.1.40.2)\n"
                "project(strata_sycl VERSION ${STRATA_SYCL_VERSION} LANGUAGES C CXX)\n", encoding="utf-8")
            with mock.patch.object(self.mod, "ROOT", root), mock.patch.object(setup, "ROOT", root):
                self.assertEqual(self.mod.sycl_version(), "0.1.41.2")


class IntelIds(unittest.TestCase):
    """sycl/setup_intel.py names a card by its PCI device id. The Arc Pro B60s measured on 2x B60 (`lspci -nn` 8086:e211,
    bmg-g21) must not fall through to "Intel GPU e211 (xe)" with a guessed VRAM size."""

    @staticmethod
    def table():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.INTEL_ARC

    def test_b60_e211_is_named_with_its_vram(self):
        self.assertEqual(self.table()["e211"], ("Arc Pro B60", 24.0))

    def test_the_ids_that_were_listed_before_are_kept(self):
        t = self.table()
        self.assertEqual(t["e223"][0], "Arc Pro B70")
        self.assertEqual(t["e221"][0], "Arc Pro B60")


class ToSycl(unittest.TestCase):
    """The config sycl/setup_intel.py writes: an explicit --vram-reserve-mib is kept, the B-series streams its experts,
    the A-series (i915) cannot (a single pinned host allocation above a few GB fails there) and loads them into RAM."""

    @staticmethod
    def mod():
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_intel", ROOT / "sycl" / "setup_intel.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def cfg(self, m, extra=()):
        base = m.MOUNT
        return {"exe": "x", "cwd": "y", "args": ["--pack", str(base / "d" / "pack"), "--native", str(base / "d" / "a.gguf"),
                                                 "--expert-cache", "auto", "--prefill", "auto", "--max-context", "32768",
                                                 "--kv", "int8", *extra]}

    def test_b_series_streams_and_defaults_the_reserve(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "1024")
        self.assertNotIn("STRATA_VERIFY_NO_HOST", c.get("env", {}))

    def test_xe_ranks_the_experts_with_the_sycl_profile_and_i915_keeps_the_shipped_one(self):
        m = self.mod()
        shipped = str(m.MOUNT / "data" / "expert-profile.bin")
        other = str(m.MOUNT / "mine" / "learned.bin")
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile-sycl.bin"))
        c = m.to_sycl(self.cfg(m, ["--expert-profile", other]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("learned.bin"))   # the user's own file stays
        c = m.to_sycl(self.cfg(m, ["--expert-profile", shipped]), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertTrue(c["args"][c["args"].index("--expert-profile") + 1].endswith("expert-profile.bin"))   # the CPU computes the misses there

    def test_an_explicit_reserve_is_kept_on_both_drivers(self):
        m = self.mod()
        for vram, drv in ((32.0, "xe"), (8.0, "i915")):
            c = m.to_sycl(self.cfg(m, ["--vram-reserve-mib", "500"]), m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, vram, drv)
            self.assertEqual(c["args"].count("--vram-reserve-mib"), 1)
            self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "500")

    def test_a_series_loads_experts_into_ram_and_leaves_no_host_off(self):
        m = self.mod()
        c = m.to_sycl(self.cfg(m), m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--stream-experts", c["args"])
        self.assertEqual(c["args"][c["args"].index("--ple-io") + 1], "ram")
        self.assertEqual(c["args"][c["args"].index("--vram-reserve-mib") + 1], "300")
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")
        self.assertEqual(c["env"]["STRATA_SYCL_BIN"], "build-sycl/strata")

    def test_gyro_memory_warning_reads_the_real_ram(self):
        # setup's RAM rule is fed 1024 GB on the Intel path; Gyro counts VRAM + RAM, so it must see the real RAM
        m = self.mod()
        import importlib.util
        spec = importlib.util.spec_from_file_location("setup_for_test", ROOT / "setup.py")
        s = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s)
        have = m.real_mem_have(s.mem_have, 16.0, 1024.0)
        self.assertEqual(have("Gyro-S", 1024.0, 8.0), 24.0)              # 16 GB RAM + 8 GB VRAM: below Gyro-S's 27.6, setup warns
        self.assertLess(have("Gyro-S", 1024.0, 8.0), s.MODELS["Gyro-S"]["ram_gb"])
        self.assertEqual(have("Gyro-S", 40.0, 8.0), 48.0)                # a real RAM figure passes through
        self.assertEqual(have("Q2_0", 1024.0, 8.0), 16.0)                # the other sizes count RAM only, as before

    def test_a_series_keeps_gyros_ngram_table_on_the_ssd(self):
        # Gyro's table is half of the 58-92 GB file: --ple-io ram would not fit beside the experts' arena on a 64 GB PC
        m = self.mod()
        cfg = self.cfg(m)
        cfg["args"][cfg["args"].index("--native") + 1] = str(m.MOUNT / "d" / "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf")
        c = m.to_sycl(cfg, m.ROOT / "build-sycl" / "strata", 64.0, {}, 8.0, "i915")
        self.assertNotIn("--ple-io", c["args"])
        self.assertEqual(c["env"]["STRATA_VERIFY_NO_HOST"], "0")

    def test_split_short_rule(self):
        # #1980: 2x B60 24 GB - IQ3_XXS (42.9 GB) was refused under NO_HOST=1, IQ2_XS (35.5 GB) loads fully
        m = self.mod()
        self.assertTrue(m.split_likely_short(42.9, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(35.5, [24.0, 24.0]))
        self.assertFalse(m.split_likely_short(50.3, [32.0, 32.0]))   # IQ3_S on 2x B70: 79%
        self.assertTrue(m.split_likely_short(77.0, [32.0, 32.0]))
        self.assertFalse(m.split_likely_short(0, [24.0]))
        self.assertFalse(m.split_likely_short(40.0, []))

    def test_small_card_rule(self):
        m = self.mod()
        self.assertTrue(m.small_card(8.0, "i915"))
        self.assertTrue(m.small_card(24.0, "i915"))
        self.assertTrue(m.small_card(10.0, "xe"))
        self.assertFalse(m.small_card(32.0, "xe"))

    def test_pictures_follow_the_one_reader_rule_and_the_vision_section_survives(self):
        """Intel Arc: pictures on gives the CPU reader (strata-vision) on the host under 24 GB and, until a 24 GB+ Arc has run the SYCL reader,
        also from 24 GB; only --vision gpu asks for the SYCL reader (strata_control/vision_reader.py is the one place for the rule)."""
        m = self.mod()
        for asked in (None, "yes", "cpu", "on"):
            self.assertEqual(m.intel_vision(asked, 8.0), "cpu")
            self.assertEqual(m.intel_vision(asked, 32.0), "cpu")
        self.assertEqual(m.intel_vision("gpu", 8.0), "gpu")
        self.assertEqual(m.intel_vision("gpu", 32.0), "gpu")
        for asked in ("no", "none", "off"):
            self.assertEqual(m.intel_vision(asked, 32.0), "none")
        section = {"exe": "/h/engine-sycl/strata-vision", "mmproj": "/h/models/mmproj.gguf", "model": "/h/models/m.gguf", "gpu": False,
                   "max_tokens": 300}
        cfg = self.cfg(m, ["--vision"])
        cfg["vision"] = section
        c = m.to_sycl(cfg, m.ROOT / "build-sycl-aot" / "strata", 64.0, {}, 32.0, "xe")
        self.assertEqual(c["vision"], section)                              # host paths: the reader runs on the host, not in the container
        self.assertIn("--vision", c["args"])

    def test_pictures_are_on_by_default_on_every_arc_and_a_battlemage_is_told_it_is_new(self):
        """Owner decision 2026-10-22: an Arc B-series card runs like any Arc. The first picture on the Arc Pro B70 reset the GPU (xe, 2026-10-19) is a
        known risk, not a refusal: pictures are on (the CPU reader) and setup prints one plain line about it."""
        m = self.mod()
        for driver, devid in (("i915", "56a1"), ("i915", ""), ("xe", "56a0"), ("xe", "e223"), ("xe", "e20b"), ("xe", "e211"), ("xe", "")):
            self.assertTrue(m.pictures_on_by_default(driver, devid), (driver, devid))
        self.assertTrue(m.is_b_series("xe", "e223"))                        # B70
        self.assertTrue(m.is_b_series("xe", "e20b"))                        # B580
        self.assertFalse(m.is_b_series("i915", "56a1"))                     # A750
        self.assertFalse(m.is_b_series("xe", "56a0"))                       # A770 on xe
        self.assertIn("new to Strata", m.B_SERIES_LINE)
        self.assertIn("issues", m.B_SERIES_LINE)
        src = (ROOT / "sycl" / "setup_intel.py").read_text(encoding="utf-8")
        self.assertIn("B_SERIES_LINE", src)
        self.assertNotIn('sys.argv += ["--vision", "none"]', src)          # no B-series card has pictures switched off for it
        self.assertIn("S.hip_vision = intel_vision", src)
        self.assertIn("S.build_vision_cpu(", src)                           # the CPU reader is compiled by setup's own step
        self.assertIn("build_vision_sycl(", src)                            # --vision gpu: the SYCL reader, when the oneAPI compiler is there

    def test_the_sycl_reader_is_built_only_when_asked_and_the_cpu_reader_stays_without_the_compiler(self):
        from unittest import mock
        m = self.mod()
        with mock.patch.object(m, "oneapi_compiler", return_value=None), mock.patch.object(m.S, "warn") as warn:
            self.assertFalse(m.build_vision_sycl_reader(Path("/nonexistent"), "/llama"))
        self.assertIn("oneAPI compiler", warn.call_args[0][0])
        self.assertIn("pictures are read on the CPU", warn.call_args[0][0])

    def test_the_container_shares_dev_shm_with_the_server_for_the_embeddings(self):
        sh = (ROOT / "sycl" / "serve" / "strata-sycl.sh").read_text(encoding="utf-8")
        self.assertIn("-v /dev/shm:/dev/shm", sh)
        self.assertIn("STRATA_SYCL_SHM", sh)

    def test_a_picture_that_cannot_reach_the_container_is_said_once(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("server_intel_t", ROOT / "sycl" / "serve" / "server_intel.py")
        src = (ROOT / "sycl" / "serve" / "server_intel.py").read_text(encoding="utf-8")
        self.assertIn("install_shared_embeddings()", src.split('if __name__ == "__main__":')[1])
        sys.path.insert(0, str(ROOT / "sycl" / "serve"))
        try:
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
        except ImportError as e:                                           # no oneAPI / xe telemetry import here: the source check above stands
            self.skipTest(str(e))
        finally:
            sys.path.pop(0)
        out = io.StringIO()
        with mock.patch.object(m.S, "combined_embeddings_path", lambda d, n: Path("/tmp/x/req.sve")), contextlib.redirect_stdout(out):
            m.install_shared_embeddings()
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
            m.S.combined_embeddings_path(Path("/tmp/x"), 10)
        self.assertEqual(out.getvalue().count("/dev/shm has no room"), 1)


class WindowsIntelTelemetry(unittest.TestCase):
    def test_counter_deltas_and_driver_reset(self):
        from serve.telemetry import _Intel
        reader = object.__new__(_Intel)
        reader.prev = {}
        self.assertIsNone(reader._rate("power", _Intel.Counter(100, 10)))
        self.assertEqual(reader._rate("power", _Intel.Counter(600, 20)), 50)
        self.assertIsNone(reader._rate("power", _Intel.Counter(10, 30)))
        self.assertIsNone(reader._rate("power", None))
        self.assertIsNone(reader._rate("power", _Intel.Counter(900, 40)))

    def test_memory_and_engine_activity_use_driver_values(self):
        from serve.telemetry import _Intel
        reader = object.__new__(_Intel)
        reader.prev = {}
        reader.memory, reader.engines, reader.temps, reader.power = [1], [2], [], None
        state = _Intel.Memory()
        state.size, state.free = 16 * 2**30, 6 * 2**30
        with mock.patch.object(reader, "_get", side_effect=[state, _Intel.Counter(100, 1000),
                                                          state, _Intel.Counter(130, 1100)]):
            first, second = reader.read(), reader.read()
        self.assertNotIn("util", first)
        self.assertEqual(second["mem_total"], 16 * 2**30)
        self.assertEqual(second["mem_used"], 10 * 2**30)
        self.assertAlmostEqual(second["util"], 30)
        self.assertNotIn("temp", second)

    def test_sycl_monitor_selects_intel_instead_of_nvml(self):
        from serve import telemetry
        with mock.patch.object(telemetry, "_Intel") as intel, mock.patch.object(telemetry, "_Nvml") as nvml:
            self.assertIs(telemetry.gpu_reader(1, intel=True, name="Intel Arc Pro B50", device_id="e212"), intel.return_value)
            intel.assert_called_once_with(1, "Intel Arc Pro B50", "e212", None)
            nvml.assert_not_called()

    def test_unavailable_driver_memory_is_not_reported_as_zero_free_vram(self):
        from serve.telemetry import _Intel
        reader = object.__new__(_Intel)
        reader.prev = {}
        reader.memory, reader.engines, reader.temps, reader.power = [1], [], [], None
        for size, free in ((0, 0), (10, 11)):
            state = _Intel.Memory()
            state.size, state.free = size, free
            with mock.patch.object(reader, "_get", return_value=state):
                self.assertEqual(reader.read(), {})

    def test_pci_id_matches_a_generic_driver_name_without_selecting_the_other_card(self):
        from serve import telemetry
        props = {}
        for handle, pci in ((21, 0xe20b), (22, 0xe212)):
            p = telemetry._Intel.Properties()
            p.core.vendorId, p.core.type, p.core.deviceId = 0x8086, 1, pci
            p.core.name = f"Intel(R) Graphics [0x{pci:x}]".encode()
            props[handle] = p
        def enum(fn, *args):
            return [11] if fn == "zesDriverGet" else [21, 22] if fn == "zesDeviceGet" else []
        def fn(name, *args):
            return (lambda *a: 0) if name == "zesInit" else (lambda *a: 1)
        with mock.patch.object(telemetry.os, "name", "nt"), \
                mock.patch.object(telemetry.ctypes, "WinDLL", create=True), \
                mock.patch.object(telemetry._Intel, "_enum", side_effect=enum), \
                mock.patch.object(telemetry._Intel, "_fn", side_effect=fn), \
                mock.patch.object(telemetry._Intel, "_get", side_effect=lambda f, h, *a: props[h]):
            reader = telemetry._Intel(0, "Intel Arc Pro B50", "e212")
        self.assertTrue(reader.ok())
        self.assertEqual(reader.dev, 22)
        self.assertEqual(reader.name(), "Intel Arc Pro B50")
        self.assertIsNone(reader.note)


    def test_duplicate_models_require_uuid_and_do_not_use_enumeration_order(self):
        from serve import telemetry
        props = {}
        for handle, uuid in ((21, "1"*32), (22, "2"*32)):
            p = telemetry._Intel.Properties()
            p.core.vendorId, p.core.type, p.core.deviceId = 0x8086, 1, 0xe20b
            p.core.name = b"Intel(R) Graphics [0xe20b]"
            p.core.uuid[:] = bytes.fromhex(uuid)
            props[handle] = p
        def enum(fn, *args):
            return [11] if fn == "zesDriverGet" else [22, 21] if fn == "zesDeviceGet" else []
        def fn(name, *args):
            if name == "zesDeviceGetCardPowerDomain": raise AttributeError("older driver")
            return lambda *a: 0
        with mock.patch.object(telemetry.os, "name", "nt"), mock.patch.object(telemetry.ctypes, "WinDLL", create=True), \
                mock.patch.object(telemetry._Intel, "_enum", side_effect=enum), \
                mock.patch.object(telemetry._Intel, "_fn", side_effect=fn), \
                mock.patch.object(telemetry._Intel, "_get", side_effect=lambda f, h, *a: props[h]):
            ambiguous = telemetry._Intel(0, device_id=0xe20b)
            self.assertFalse(ambiguous.ok())
            selected = telemetry._Intel(0, device_id=0xe20b, uuid="1"*32)
            self.assertTrue(selected.ok())
            self.assertEqual(selected.dev, 21)
            self.assertIsNone(selected.power)
            missing = telemetry._Intel(0, device_id=0xe20b, uuid="3"*32)
            self.assertFalse(missing.ok())
            zero = telemetry._Intel(0, device_id=0xe20b, uuid="0"*32)
            self.assertFalse(zero.ok())


class WindowsInstallRegression(unittest.TestCase):
    def test_vision_hook_accepts_the_shared_installer_arguments(self):
        from sycl import setup_windows as windows
        self.assertEqual(windows.vision_choice("yes", 16.0), "gpu")
        self.assertEqual(windows.vision_choice("cpu", 12.0), "cpu")
        self.assertEqual(windows.vision_choice("off", 16.0), "none")

    def test_b50_matches_the_real_level_zero_name(self):
        from sycl import setup_windows as windows
        name, vram = windows.INTEL_ARC["e212"]
        card = {"index": 0, "name": "Intel " + name, "vram_gb": vram}
        devices = [{"index": 2, "name": "Intel(R) Arc(TM) Pro B50 Graphics",
                    "memory_bytes": 16 * 2**30, "integrated": False}]
        with mock.patch.object(windows, "devices", return_value=devices), \
                mock.patch.object(windows, "intel_gpus", return_value=[card]):
            self.assertEqual(windows.select_device(Path("unused"), card)["sycl_index"], 2)

    def test_sycl_reader_uses_gpu_and_can_fall_back_to_cpu(self):
        from sycl import setup_windows as windows
        from strata_control import vision_reader as VR
        with tempfile.TemporaryDirectory() as folder:
            eng = Path(folder)
            exe = eng / "strata-vision.exe"
            exe.touch()
            (eng / "BUILD.json").write_text(json.dumps({"backend": "sycl", "vision": "gpu"}))
            mode, reader = windows.vision_reader(eng, "gpu", {})
            self.assertEqual(mode, "gpu")
            cfg = VR.vision_section(reader, {"model": "model.gguf"})
            self.assertTrue(cfg["gpu"])
            self.assertEqual(cfg["exe"], str(exe))
            fallback = VR.cpu_fallback(cfg)
            self.assertFalse(fallback["gpu"])
            self.assertEqual(fallback["exe"], str(exe))
            mode, reader = windows.vision_reader(eng, "cpu", {})
            self.assertEqual(mode, "cpu")
            self.assertIsNone(reader.cpu_exe)



class WindowsIntelDeviceIdentity(unittest.TestCase):
    def setUp(self):
        from sycl import setup_windows
        self.w = setup_windows

    def choose(self, card, devices, cards=None):
        with mock.patch.object(self.w, "devices", return_value=devices), \
                mock.patch.object(self.w, "intel_gpus", return_value=cards or [card]):
            return self.w.select_device(Path("unused"), card)

    def test_every_catalog_card_matches_by_pci_id_despite_driver_branding(self):
        for pci, (name, memory) in self.w.INTEL_ARC.items():
            for driver_name in (f"Intel(R) {name.replace('Arc', 'Arc(TM)')} Graphics", f"Intel(R) Graphics [0x{pci}]"):
                with self.subTest(pci=pci, driver=driver_name):
                    card = {"index": 0, "name": "Intel " + name, "devid": pci, "vram_gb": memory}
                    device = {"index": 4, "name": driver_name, "device_id": int(pci, 16), "vendor_id": 0x8086,
                              "integrated": False, "memory_bytes": int(memory * 2**30)}
                    self.assertEqual(self.choose(card, [device])["sycl_index"], 4)

    def test_same_model_cards_follow_pci_address_when_the_orders_differ(self):
        cards = [{"index": i, "name": "Intel Arc B580", "devid": "e20b", "pci_address": f"0{i+3}:00.0"} for i in range(2)]
        devices = [{"index": 8-i, "name": "Intel(R) Graphics [0xe20b]", "device_id": 0xe20b,
                    "pci_address": "0000:" + c["pci_address"], "uuid": str(i+1)*32,
                    "integrated": False, "memory_bytes": 12*2**30} for i,c in enumerate(cards)]
        for card, device in zip(cards, devices):
            chosen = self.choose(card, list(reversed(devices)), cards)
            self.assertEqual(chosen["sycl_index"], device["index"])
            self.assertEqual(chosen["uuid"], device["uuid"])

    def test_legacy_name_matching_never_confuses_a770_and_a770m(self):
        card = {"index": 0, "name": "Intel Arc A770", "devid": "56a0"}
        mobile = {"index": 0, "name": "Intel(R) Arc(TM) A770M Graphics", "integrated": False, "memory_bytes": 16*2**30}
        desktop = {**mobile, "index": 3, "name": "Intel(R) Arc(TM) A770 Graphics"}
        self.assertEqual(self.choose(card, [mobile, desktop])["sycl_index"], 3)
        with mock.patch.object(self.w.S, "fail", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit): self.choose(card, [mobile])

    def test_ambiguous_legacy_devices_and_missing_capabilities_fail_before_configuration(self):
        card = {"index": 0, "name": "Intel Arc B580", "devid": "e20b"}
        good = {"index": 0, "name": card["name"], "device_id": 0xe20b, "integrated": False, "memory_bytes": 12*2**30}
        for devices in ([good, {**good, "index": 1}], [{**good, "integrated": True}],
                        [{**good, "vendor_id": 0x10de}], [{**good, "memory_bytes": 0}], [{**good, "usm_host": False}]):
            with self.subTest(devices=devices), mock.patch.object(self.w.S, "fail", side_effect=SystemExit(1)):
                with self.assertRaises(SystemExit): self.choose(card, devices)

    def test_actual_memory_overrides_the_a770_catalog_estimate(self):
        card = {"index": 0, "name": "Intel Arc A770", "devid": "56a0", "vram_gb": 16}
        device = {"index": 0, "name": card["name"], "device_id": 0x56a0, "integrated": False, "memory_bytes": 8*2**30}
        self.assertEqual(self.choose(card, [device])["vram_gb"], 8)

    def test_updates_keep_the_same_physical_card_when_wmi_reorders(self):
        cards = [{"index": 0, "instance_id": "PCI/other", "name": "Intel Arc B580", "devid": "e20b"},
                 {"index": 1, "instance_id": "PCI/original", "name": "Intel Arc Pro B50", "devid": "e212"}]
        cfg = {"gpu": 0, "gpu_instance_id": "pci/ORIGINAL", "exe": "unused/strata.exe",
               "args": ["--expert-cache", "3400", "--prefill", "8192"], "env": {"CUSTOM": "kept"}}
        def choose(eng, card): return {**card, "sycl_index": 7, "uuid": "a"*32}
        with mock.patch.object(self.w, "intel_gpus", return_value=cards), \
                mock.patch.object(self.w, "select_device", side_effect=choose), mock.patch.object(self.w.S, "write_config"):
            out = self.w.update_selected_gpu(Path("unused.json"), cfg)
            self.assertEqual(out["gpu"], 1)
            self.assertEqual(out["gpu_uuid"], "a"*32)
            self.assertEqual(out["args"], cfg["args"])
            self.assertEqual(out["env"]["ONEAPI_DEVICE_SELECTOR"], "level_zero:7")
        with mock.patch.object(self.w, "intel_gpus", return_value=cards[:1]), \
                mock.patch.object(self.w.S, "fail", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit): self.w.update_selected_gpu(Path("unused.json"), cfg)

    def test_alchemist_emulation_is_added_for_a_series_only_and_keeps_explicit_overrides(self):
        for pci, (name, _) in self.w.INTEL_ARC.items():
            with self.subTest(pci=pci):
                card = {"index": 0, "sycl_index": 0, "name": "Intel " + name, "devid": pci}
                cfg = self.w.to_native({"env": {}, "args": []}, card)
                self.assertEqual("IGC_EnableDPEmulation" in cfg["env"], name.startswith("Arc A"))
                if name.startswith("Arc A"):
                    out = self.w.to_native({"env": {"IGC_EnableDPEmulation": "0"}}, card)
                    self.assertEqual(out["env"]["IGC_EnableDPEmulation"], "0")

    def test_unknown_discrete_arc_model_is_probed_but_integrated_arc_is_excluded(self):
        adapters = [{"Name": "Intel(R) Arc(TM) Pro A99 Graphics", "PNPDeviceID": "PCI\\VEN_8086&DEV_1234", "PciBus": 4, "PciAddress": 0},
                    {"Name": "Intel(R) Arc(TM) Graphics", "PNPDeviceID": "PCI\\VEN_8086&DEV_5678"}, None]
        from strata_control import engine_component as ec
        with mock.patch.object(self.w.subprocess, "run", return_value=mock.Mock(stdout=json.dumps(adapters))), \
                mock.patch.object(ec, "windows_display_adapters", return_value=[]), \
                mock.patch("serve.telemetry._Intel", return_value=mock.Mock(discrete_device_ids={0x1234})):
            cards = self.w.intel_gpus()
        self.assertEqual(len(cards), 1)
        self.assertEqual(cards[0]["devid"], "1234")
        self.assertEqual(cards[0]["pci_address"], "04:00.0")
        self.assertEqual(cards[0]["vram_gb"], 0)



    def test_runtime_architecture_supplies_emulation_for_generically_named_pro_cards(self):
        card = {"index": 0, "sycl_index": 0, "name": "Intel Graphics", "alchemist": True}
        out = self.w.to_native({"env": {}}, card)
        self.assertEqual(out["env"]["OverrideDefaultFP64Settings"], "1")
        out = self.w.to_native({"env": {}}, {**card, "name": "Intel Arc A770", "alchemist": False})
        self.assertNotIn("OverrideDefaultFP64Settings", out["env"])

    def test_null_optional_pci_id_keeps_legacy_name_compatibility(self):
        card = {"index": 0, "name": "Intel Arc Pro B50", "devid": "e212"}
        d = {"index": 3, "device_id": None, "name": "Intel(R) Arc(TM) Pro B50 Graphics", "integrated": False,
             "memory_bytes": 16*2**30}
        self.assertEqual(self.choose(card, [d])["sycl_index"], 3)

    def test_two_windows_cards_and_only_one_visible_runtime_card_cannot_be_guessed(self):
        cards = [{"index": i, "name": "Intel Arc B580", "devid": "e20b"} for i in range(2)]
        d = {"index": 0, "device_id": 0xe20b, "name": "Intel Arc B580", "integrated": False,
             "memory_bytes": 12*2**30, "pci_address": "0000:03:00.0"}
        with mock.patch.object(self.w.S, "fail", side_effect=SystemExit(1)):
            with self.assertRaises(SystemExit): self.choose(cards[1], [d], cards)

    def test_probe_filters_are_cleared_before_global_device_enumeration(self):
        with mock.patch.dict(self.w.os.environ, {"ONEAPI_DEVICE_SELECTOR": "level_zero:9", "SYCL_DEVICE_FILTER": "level_zero:8"}), \
                mock.patch.object(self.w.S, "hip_lib_dirs", return_value=[]):
            env = self.w.engine_env(Path("unused"))
        self.assertNotIn("ONEAPI_DEVICE_SELECTOR", env)
        self.assertNotIn("SYCL_DEVICE_FILTER", env)

    def test_intel_free_vram_reads_the_saved_device_uuid(self):
        from serve import server, telemetry
        svc = object.__new__(server.Service)
        svc.backend, svc.gpu_index, svc.gpu_name = "sycl", 1, "Intel Arc B580"
        svc.gpu_device_id, svc.gpu_uuid = "e20b", "1"*32
        reader = mock.Mock(ok=lambda: True, read=lambda: {"mem_total": 12*2**30, "mem_used": 4*2**30})
        with mock.patch.object(telemetry, "gpu_reader", return_value=reader) as choose:
            self.assertEqual(svc.free_vram_mib(), 8192)
            self.assertEqual(choose.call_args.kwargs["uuid"], "1"*32)
            self.assertTrue(choose.call_args.kwargs["intel"])


if __name__ == "__main__":
    unittest.main()
