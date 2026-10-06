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

Measured on 2026-10-07 with an Intel Arc B580 12 GB, Core i7-14700K, 64 GB RAM and Intel graphics
driver 32.0.101.9034. Model: Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, INT8 KV, resident experts and MTP
(spec 4, min-p 0.5). Tests used the release package after updating to upstream `82f46a8`.

| Test | Context capacity | Input tokens | Output tokens | Average decode |
|---|---:|---:|---:|---:|
| Benchy v1, greedy | 8,192 | 2,185 | 256 | **31.46 tok/s** |
| Benchy v1, GPU vision loaded | 131,072 | 2,185 | 256 | **31.19 tok/s** |
| Same prompt repeated, 2,182 tokens reused | 131,072 | 2,185 | 256 | **33.67 tok/s** |
| Tool prompt, temperature 0.6 | 131,072 | 1,233 | 92 | **24.30 tok/s** |
| Same tool prompt repeated, 1,228 tokens reused | 131,072 | 1,233 | 78 | **25.34 tok/s** |

The 8K test read the prompt at **85.15 tok/s**; the first 128K-capacity test read it at **116.31 tok/s**.
The 128K tests kept the GPU image encoder loaded with a 700 MiB VRAM reserve. Tool prompts used top-p 0.95,
top-k 20 and seed 42; these tests generated responses without executing MCP tools.

WebUI live speed uses a two-second window, so it can exceed the whole-request average. The repeated
benchy request measured 35.41 tok/s window median and 39.40 tok/s peak. Speed varies with the prompt,
MTP acceptance and available VRAM. Context capacity here does not mean a full 128K-length prompt was tested.

## Limits

- One discrete Arc GPU per model; integrated GPUs are unsupported.
- K8V4 KV streaming is unsupported. Use `--kv-streaming off` with `--kv k8v4`, or use INT8 KV.
