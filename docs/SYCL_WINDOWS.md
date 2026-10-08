# Intel Arc on Windows (experimental)

Requires Windows 10/11, a discrete Intel Arc GPU, a current Intel graphics driver and an AVX2 CPU.
Choose the model according to your RAM: [model sizes](MODELS.md#pick-by-ram).
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

Setup downloads the projector and configures the GPU encoder. Images can then be attached in the WebUI.

## Build from source

Install Visual Studio C++ desktop tools, Intel oneAPI DPC++/C++ Compiler and oneMKL, CMake 3.24+, Ninja and Git.
From the repository root:

```bat
sycl\build-release-windows.bat
```

Output: `dist/strata-windows-x64-sycl.zip`. To install the locally built package:

```bat
START-HERE.bat --backend sycl --prebuilt dist
```

## Benchmark

Measured on **2026-10-09** with the following local configuration. These results use full RAM expert
residency and fixed expert-cache budgets and prefill chunks. Only the current build was run in this test.

### Test environment

| Component | Configuration |
|---|---|
| GPU | Intel Arc B580, 12 GB; driver 32.0.101.9034 |
| CPU | Core i7-14700K, all cores |
| RAM | 64 GB |
| Model | Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS |
| Vision | `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, ggml SYCL GPU encoder, image-token cap 1024 |

### Test parameters

All four cases used these arguments, with the context capacity changed to 8192 or 131072:

```text
--serve
--pack <windows-iq2-xs pack directory>
--native <Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf>
--ple-gguf <Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf>
--expert-profile <expert-profile.bin>
--resident-experts --expert-cache <budget from the table below>
--prefill <chunk from the table below> --max-context 131072
--kv int8 --kv-resident 20480
--spec 4 --spec-min-p 0.5 --mtp <mtp/rt directory>
--pcie-frac 0.37 --vram-reserve-mib <reserve below>
```

Environment: `ONEAPI_DEVICE_SELECTOR=level_zero:0`, `STRATA_RESIDENT_PIN=0`,
`STRATA_VERIFY_DEVICE_PLAN=1`, `STRATA_STAGE_PIN=0`. The all-core worker defaults were retained. The MTP draft vocabulary was
the existing CJK set (106,299 tokens); the draft head occupied 137.9 MiB and the MTP layer about 835 MiB.

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
| Text only | 8,192 | 219.03 | **42.57** | **43.61** | **45.16** |
| GPU vision loaded | 8,192 | 210.56 | **39.46** | **42.06** | **42.92** |
| Text only | 131,072 | 371.15 | **46.65** | **46.92** | **48.22** |
| GPU vision loaded | 131,072 | 356.81 | **44.31** | **47.00** | **46.53** |

### Fixed allocation

The requested expert-cache budgets were 3043 (text 8K), 2262 (vision 8K), 3612 (text 128K) and
3049 (vision 128K). This option budgets largest-expert blobs; it is not the final number of slots.
The driver checked the final engine-reported slot count, expert VRAM and resident expert RAM
against the table before generating. The 8K cases explicitly used `--no-prefill-borrow`; 128K used
`--prefill-borrow`. No cache or prefill size was selected by `auto` in the main four sessions.

| Case / capacity | VRAM expert slots | Expert VRAM (GiB) | Resident expert RAM (GiB) | Prefill chunk | Slots / GiB available to prefill loan | Loaded free VRAM (MiB) |
|---|---:|---:|---:|---:|---|---:|
| Text only, 8,192 | 3179 | 4.28 | 28.74 | 1024 | 0 / 0 | 1496 |
| GPU vision loaded, 8,192 | 2358 | 3.18 | 29.84 | 1024 | 0 / 0 | 1422 |
| Text only, 131,072 | 3780 | 5.08 | 31.50 | 8192 | 2662 / 3.56 | 776 |
| GPU vision loaded, 131,072 | 3186 | 4.29 | 32.30 | 8192 | 2655 / 3.56 | 268 |

The prefill loan temporarily reuses expert-cache VRAM for prompt buffers; its GiB are part of the
expert-cache total, not an additional permanent allocation. The 8K cases used 1024-token chunks with
their own buffers; both 128K cases used fixed 8192-token chunks and borrowed cache slots. This
difference, along with different cache sizes, means a larger configured capacity can be faster for
this short prompt; it does not mean longer prompts are faster.

The text-only 128K cache budget leaves 776 MiB free after loading. These VRAM figures report
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
| Text only, 8,192 | 40.19 | 54.27 | 0.00 | 24.07 |
| GPU vision loaded, 8,192 | 40.19 | 53.97 | 1.52 | 22.14 |
| Text only, 131,072 | 45.47 | 52.73 | 0.00 | 17.40 |
| GPU vision loaded, 131,072 | 45.47 | 51.78 | 1.52 | 14.76 |

Loading briefly reduced available physical RAM to 0.00 GiB; the minimum available system commit across these sessions was 7.58 GiB.
The remaining RAM during decode is not evidence of a partial expert cache: loading peaks and other
allocations need headroom, and the measured requests did not read experts from disk. These timings
do not establish that Windows performed no paging.

| Case / capacity | Decode expert-cache hit range | MTP accepted / offered range | Two-second live peak (tok/s) |
|---|---:|---:|---:|
| Text only, 8,192 | 57.7–60.7% | 73.5–76.4% | 53.0 |
| GPU vision loaded, 8,192 | 49.8–53.1% | 74.3–79.1% | 48.0 |
| Text only, 131,072 | 62.0–66.7% | 74.8–79.3% | 55.5 |
| GPU vision loaded, 131,072 | 57.9–61.2% | 74.8–78.8% | 54.5 |

All Benchy requests reached the 256-token cap (`finish=length`); none stopped early, and no generated
12-token sequence repeated within a response. Cache swaps, PCIe work and MTP acceptance can change
outputs and speed between requests. The live two-second rate can exceed the whole-decode average.

### Tool prompt

The same 1,233-token prompt containing IDA tool definitions was tested in the 128K + vision session after Benchy, using temperature
0.6, top-p 0.95, top-k 20 and seed 42, with a 256-token output cap. The repeat reused 1,228 tokens.
Its user question asked the model to identify itself, with IDA tools available in the prompt.
This measures generation with tool definitions in the prompt, **not tool execution or an MCP round trip**.

| Request | Output tokens | Finish | Decode (tok/s) | Prompt read time (ms) |
|---|---:|---|---:|---:|
| tool-first | 89 | stop | **31.70** | 4612.4 |
| tool-repeat | 120 | stop | **35.51** | 66.3 |

### WebUI writing

The latest completed user-run WebUI writing request on 2026-10-09, on the same PC and model, with MCP tools and vision
turned off. Context capacity was 131,072; the actual prompt was short and reused no tokens.

| Input tokens | Generated tokens | Prompt time | Prefill (tok/s) | Decode time | Decode (tok/s) |
|---:|---:|---:|---:|---:|---:|
| 28 | 5,871 | 0.491 s | 57.0 | 162.197 s | **36.2** |

Settings: `--expert-cache 3400 --prefill 8192 --prefill-borrow --vram-reserve-mib 700`;
INT8 KV with 20,480 resident cells, MTP spec 4 / min-p 0.5, PCIe fraction 0.37 and full pageable RAM
expert residency. The environment settings were the same as above. The actual cache held
3,514 slots / 4.73 GiB in VRAM and 31.86 GiB in RAM; loaded free VRAM was 477 MiB.

Expert-cache hits were 79.8%; MTP accepted 2,870 of 5,087 drafts (**56.4%**). Expert blob reads from
files were zero.

## Limits

- One discrete Arc GPU per model; integrated GPUs are unsupported.
- K8V4 KV streaming is unsupported. Use `--kv-streaming off` with `--kv k8v4`, or use INT8 KV.
