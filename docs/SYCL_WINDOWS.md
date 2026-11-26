# Intel Arc on Windows (experimental)

Requires Windows 10/11, a discrete Intel Arc GPU, a current Intel graphics driver and an AVX2 CPU.
Choose the model according to your RAM: [model sizes](MODELS.md#pick-by-memory).
The release package includes the engine, GPU image encoder and runtimes; no oneAPI or Visual Studio installation is needed.

## Install and start

Double-click `START-HERE.bat`. Setup selects Arc automatically when no supported NVIDIA or AMD card is present.
To select Intel explicitly:

```bat
START-HERE.bat --backend sycl
```

Follow the setup questions to download the model and start the WebUI at `http://127.0.0.1:8080`.
Later runs start the installed model. Use `--setup` to change settings or `--rollback-engine` to restore the previous engine.

## Vision

Select images during setup, or enable them on an existing installation:

```bat
START-HERE.bat --backend sycl --setup --vision gpu
```

Setup downloads the projector and configures the GPU encoder. Images can then be attached in the WebUI. Use `--vision cpu` for the CPU reader; a GPU reader that cannot start falls back to CPU.

## Build from source

Install Visual Studio C++ desktop tools, Intel oneAPI DPC++/C++ Compiler and oneMKL, CMake 3.24+, Ninja and Git.
From the repository root:

```bat
engine\sycl\build-release-windows.bat
```

Output: `dist/strata-windows-x64-sycl.zip`. To install the locally built package:

```bat
START-HERE.bat --backend sycl --prebuilt dist
```

## Benchmark

Measured on **2026-11-26** with engine **0.1.69** (commit `d8f11a799`) and the following local configuration. These results use full RAM expert
residency and fixed expert-cache budgets and prefill chunks. Only the current build was run in this test.

### Test environment

| Component | Configuration |
|---|---|
| GPU | Intel Arc B580, 12 GB; driver 32.0.101.9175 |
| CPU | Core i7-14700K, all cores |
| RAM | 64 GB DDR4-3600 CL16; 16-20-20-38, tRC 58, CR 1T |
| Model | Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS |
| Vision | `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, ggml SYCL GPU encoder, image-token cap 1024 |

### Test parameters

All four cases used these arguments, with the context capacity set to 8192 or 131072:

```text
--serve
--pack <windows-iq2-xs pack directory>
--native <Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf>
--ple-gguf <Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf>
--expert-profile <expert-profile.bin>
--resident-experts --expert-cache <budget from the table below>
--prefill <chunk from the table below> --max-context <capacity>
--kv int8 --kv-resident 20480
--spec 4 --spec-min-p 0.5 --mtp <mtp/rt directory>
--pcie-frac 0.37 --vram-reserve-mib <reserve below> --pool-workers 19
```

Environment: `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `STRATA_RESIDENT_PIN=0`,
`STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_STAGE_PIN=0`, `STRATA_SPIN_MAX=2000000`. The CPU pool used 19 workers plus the host thread. The MTP draft vocabulary was
the existing CJK set (106,299 tokens); the draft head occupied 137.9 MiB and the MTP layer about 802 MiB at 8K and 835 MiB at 128K.

**No `--resident-budget-gib` limit was set.** The engine copied the full non-VRAM expert complement
into pageable RAM, including RAM copies of slots lent to prefill where a loan was used. Expert files
were mapped from the GGUF shards; all measured requests reported **zero expert blob reads from files**.
Allocating a larger nominal budget would not add experts once this complement is complete. Expert RAM
alone is not total process memory: weights, KV, mappings and temporary allocations use additional RAM.

Vision cases added `--vision` and loaded a separate GPU encoder before the engine. A 448 × 448 white
image with a red square was encoded to warm it up; it remained loaded throughout the text benchmark.
The measured requests themselves contained no image. Text-only cases had no encoder process and no
`--vision`. Text-only 128K used a 700 MiB reserve; the other three cases used 256 MiB.
These reserves and cache budgets are measured settings for this PC.

### Results

Benchy v1 used the same **2,185 input tokens and 256 generated tokens** as the previous tests, with
greedy decoding (temperature 0; top-p 0.95, top-k 20, seed 42). Each row is one fresh engine session:
one first request followed by two identical requests, each reusing 2,182 prompt tokens and reading 3.
Decode is `generated tokens / native decode time`; first prefill is `2,185 / native prompt time`.
Startup/model loading is excluded. These are short-prompt measurements at two configured capacities,
**not a full 128K prompt or a validation of performance at 128K occupied context**.

| Case | Context capacity | First prefill (tok/s) | First decode (tok/s) | Repeat 1 (tok/s) | Repeat 2 (tok/s) |
|---|---:|---:|---:|---:|---:|
| Text only | 8,192 | 280.17 | **45.74** | **48.44** | **47.83** |
| GPU vision loaded | 8,192 | 250.50 | **40.30** | **43.98** | **42.59** |
| Text only | 131,072 | 409.71 | **46.02** | **51.80** | **50.59** |
| GPU vision loaded | 131,072 | 468.20 | **44.29** | **51.50** | **47.59** |

### Fixed allocation

The requested expert-cache budgets were 3043 (text 8K), 2262 (vision 8K), 3612 (text 128K) and
3049 (vision 128K). This option budgets largest-expert blobs; it is not the final number of slots.
The driver checked the fixed slot count and expert VRAM before generating and recorded the actual resident expert RAM and prefill loan. The 8K cases explicitly used `--no-prefill-borrow`; 128K used
`--prefill-borrow`. No cache or prefill size was selected by `auto` in the main four sessions.

| Case / capacity | VRAM expert slots | Expert VRAM (GiB) | Resident expert RAM (GiB) | Prefill chunk | Slots / GiB available to prefill loan | Loaded free VRAM (MiB) |
|---|---:|---:|---:|---:|---|---:|
| Text only, 8,192 | 3179 | 4.28 | 28.74 | 1024 | 0 / 0 | 1970 |
| GPU vision loaded, 8,192 | 2358 | 3.18 | 29.84 | 1024 | 0 / 0 | 1928 |
| Text only, 131,072 | 3780 | 5.08 | 31.21 | 8192 | 2448 / 3.27 | 1305 |
| GPU vision loaded, 131,072 | 3186 | 4.29 | 32.01 | 8192 | 2440 / 3.27 | 948 |

The prefill loan temporarily reuses expert-cache VRAM for prompt buffers; its GiB are part of the
expert-cache total, not an additional permanent allocation. The 8K cases used 1024-token chunks with
their own buffers; both 128K cases used fixed 8192-token chunks and borrowed cache slots. This
difference, along with different cache sizes, means a larger configured capacity can be faster for
this short prompt; it does not mean longer prompts are faster.

The text-only 128K cache budget leaves 1305 MiB free after loading. These VRAM figures report
allocations, not proof that WDDM keeps every allocation in dedicated physical VRAM.

At 128K, INT8 KV streaming kept 20,480 cells per QSA layer in VRAM and allocated **1.55 GiB of pinned
host KV memory**. At 8K all context cells fit in the resident setting and no host KV streaming arena
was reported. Expert RAM was pageable in all four cases, with **0 GiB device-pinned expert RAM**.

### System RAM and request behaviour

Memory was sampled every 0.5 seconds. Private memory is process commit; working set is physical RAM
resident in that process. Peak working set includes loading. These figures must not be added to
file sizes or treated as interchangeable measurements of RAM.

| Case / capacity | Engine max private (GiB) | Engine peak working set (GiB) | Vision max private (GiB) | Minimum free physical RAM during Benchy (GiB) |
|---|---:|---:|---:|---:|
| Text only, 8,192 | 39.41 | 55.16 | 0.00 | 25.44 |
| GPU vision loaded, 8,192 | 39.41 | 52.15 | 1.28 | 22.74 |
| Text only, 131,072 | 44.39 | 54.00 | 0.00 | 20.20 |
| GPU vision loaded, 131,072 | 44.39 | 55.16 | 1.27 | 19.69 |

Minimum available physical RAM across the four sessions was 0.00 GiB. Minimum available system commit was 29.46 GiB.
The remaining RAM during decode is not evidence of a partial expert cache: loading peaks and other
allocations need headroom, and the measured requests did not read experts from disk. These timings
do not establish that Windows performed no paging.

| Case / capacity | Decode expert-cache hit range | MTP accepted / offered range | Two-second live peak (tok/s) |
|---|---:|---:|---:|
| Text only, 8,192 | 55.7–62.1% | 75.8–79.4% | 56.5 |
| GPU vision loaded, 8,192 | 49.8–52.0% | 72.9–78.6% | 51.0 |
| Text only, 131,072 | 59.9–65.5% | 75.3–78.1% | 61.0 |
| GPU vision loaded, 131,072 | 55.2–60.1% | 74.7–78.1% | 58.5 |

All Benchy requests reached the 256-token cap (`finish=length`); none stopped early, and no generated
12-token sequence repeated within a response. Cache swaps, PCIe work and MTP acceptance can change
outputs and speed between requests. The live two-second rate can exceed the whole-decode average.

### Historical tool prompt (2026-10-12)

The same 1,233-token prompt containing IDA tool definitions was tested on 2026-10-12 in the 128K + vision session after Benchy, using temperature
0.6, top-p 0.95, top-k 20 and seed 42, with a 256-token output cap. The repeat reused 1,228 tokens.
Its user question asked the model to identify itself, with IDA tools available in the prompt.
This measures generation with tool definitions in the prompt, **not tool execution or an MCP round trip**.

| Request | Output tokens | Finish | Decode (tok/s) | Prompt read time (ms) |
|---|---:|---|---:|---:|
| tool-first | 149 | stop | **27.52** | 4561.7 |
| tool-repeat | 55 | stop | **41.51** | 66.7 |

### API benchmark

`terminal/tools/strata_bench.py` suite 1: three runs per test, temperature 0, thinking off, concurrency 1.
Same model and environment as above, with 65,536-token context capacity and GPU vision loaded.
Expert-cache budget 3049, prefill 8192, reserve 256 MiB, 19 CPU workers and 20,480 resident KV cells.
The actual cache held 3,186 slots / 4.29 GiB in VRAM and 31.94 GiB in RAM. The prefill loan was 2,392 slots / 3.21 GiB.
Speeds are engine-reported; TTFT is measured client-side. Values are medians with ranges.

| Workload | Prompt tokens | Runs | Prefill (tok/s) | Decode (tok/s) | TTFT (s) |
|---|---:|---:|---|---|---|
| Short chat | 43–51 | 9 | — | 39.3 (35.5–47.4) | 0.75 (0.66–0.80) |
| 4K fresh prompt | 4,019 | 3 | 1050.8 (1024.8–1055.7) | — | 3.86 (3.84–3.96) |
| 16K fresh prompt | 16,058 | 3 | 1412.4 (1395.9–1417.4) | — | 11.43 (11.37–11.55) |
| Agent session, six turns | 14,684 at the last turn | 3 | — | 46.6 (37.6–53.0) | 4.79 (3.48–5.54) |

Agent follow-up cache hit rate was 65.5%. Earlier-turn tokens re-read per session were 1 (0–73). The agent workload reads documents from the measured checkout.
Output caps were 100 tokens for chat, 24 for fresh prompts and 64 per agent turn.

### Historical WebUI writing (2026-10-30)

A user-run WebUI writing request on engine 0.1.54 on 2026-10-30, on the same PC and model, with MCP tools and vision
turned off. Context capacity was 131,072; the actual prompt was short and reused no tokens.

| Input tokens | Generated tokens | Prompt time | Prefill (tok/s) | Decode time | Decode (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 24 | 7,144 | 1.014 s | 23.7 | 187.009 s | **38.2** |

Settings: `--expert-cache 3400 --prefill 8192 --prefill-borrow --vram-reserve-mib 700`;
INT8 KV with 20,480 resident cells, MTP spec 4 / min-p 0.5, PCIe fraction 0.37 and full pageable RAM
expert residency. The environment settings were the same as above, with 13 CPU workers plus the host thread. The actual cache held
3,556 slots / 4.78 GiB in VRAM and 31.80 GiB in RAM; loaded free VRAM was 1501 MiB.

Expert-cache hits were 79.2%; MTP accepted 3,410 of 6,451 drafts (**52.9%**). Expert blob reads from
files were zero.

## Limits

- One discrete Arc GPU per model; integrated GPUs are unsupported. Setup matches PCI addresses and saves the selected device's UUID instead of relying on enumeration order. The Monitor reads Intel GPU load and VRAM through Level Zero Sysman; sensors the driver does not expose stay blank.
- K8V4 KV streaming is unsupported. Use `--kv-streaming off` with `--kv k8v4`, or use INT8 KV.
