# Strata — Windows Intel Arc SYCL

English | [简体中文](README.zh-CN.md)  

A personal fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) for running Qwen3.8-Flash-Next on
**Intel Arc GPUs under native Windows**. It retains Strata's WebUI, OpenAI / Anthropic APIs and MCP tools,
and adds Windows SYCL builds, GPU vision and release packages.

Primarily tested on an **Arc B580 12 GB**. Intel support remains experimental.

## Install

Requires Windows 10/11, a discrete Intel Arc GPU, a current Intel graphics driver and an AVX2 CPU.
Choose a [model size](docs/MODELS.md#pick-by-memory) for your RAM and disk space; the test PC has 64 GB RAM.

1. [Download this repository](https://github.com/NeneNyann/Strata-intel-sycl-windows/archive/refs/heads/main.zip)
   and extract it, or clone it.
2. Run from the project directory:

   ```bat
   START-HERE.bat --backend sycl
   ```

3. Choose the model, context and whether to enable images. Setup downloads the engine, model and MTP weights,
   then starts the WebUI at `http://127.0.0.1:8080`.

The release asset is `strata-windows-x64-sycl.zip`, containing the engine, GPU image encoder and runtimes.
Using the release package does not require oneAPI or Visual Studio.

For later starts, run `START-HERE.bat`. Add `--setup` to change settings. To enable images:

```bat
START-HERE.bat --backend sycl --setup --vision gpu
```

See [MCP tools](docs/DETAILS.md#tools-from-mcp-servers) for MCP configuration and
[installation options](docs/INSTALL.md) for other settings.

## Performance

Measured on 2026-10-30 with engine 0.1.54: Arc B580 12 GB, Core i7-14700K, 64 GB DDR4-3600 CL16
(16-20-20-38, tRC 58, CR 1T), Qwen3.8-Flash-Next GSQ-RCO IQ2_XS,
INT8 KV and MTP spec 4. Full RAM expert residency, fixed expert-cache budgets and prefill chunks, all CPU cores;
no RAM budget limit. Benchy v1: 2,185 input tokens, 256 output tokens, greedy.

| Case | Context capacity | First decode | Repeated decode |
|---|---:|---:|---:|
| Text only | 8K | **42.16 tok/s** | **42.86–46.26 tok/s** |
| GPU vision loaded | 8K | **39.18 tok/s** | **42.70–42.97 tok/s** |
| Text only | 128K | **46.30 tok/s** | **47.56–48.89 tok/s** |
| GPU vision loaded | 128K | **42.23 tok/s** | **45.57–48.47 tok/s** |

128K is the configured capacity; these tests used short prompts. Repeats reused 2,182 input tokens.
In a WebUI writing session with 128K context capacity, MCP tools and vision off, a 24-token prompt
generated 7,144 tokens at **38.2 tok/s**.

`tools/strata_bench.py` (64K, GPU vision loaded, three runs, medians): short-chat decode **37.4 tok/s**,
4K / 16K prefill **725.6 / 898.7 tok/s**, agent TTFT **6.15 s**.

See [Windows SYCL benchmarks](docs/SYCL_WINDOWS.md#benchmark) for full parameters, actual expert RAM/VRAM,
prefill allocation and tool-prompt results.

## Build a release

Install Visual Studio C++ desktop tools, Intel oneAPI DPC++/C++ Compiler, oneMKL, CMake 3.24+, Ninja and Git,
then run:

```bat
sycl\build-release-windows.bat
```

Output: `dist/strata-windows-x64-sycl.zip`. To install the locally built package:

```bat
START-HERE.bat --backend sycl --prebuilt dist
```

## Updates and limits

`UPDATE.bat` updates the code, engine and installed model configurations.

- The Windows installer currently supports one discrete Arc GPU per model.
- K8V4 KV streaming is unsupported. Use `--kv int8`, or `--kv-streaming off` with K8V4.
- GPU vision has been tested; full 128K-length prompts and multi-image conversations have not been fully validated.

[Windows SYCL guide](docs/SYCL_WINDOWS.md) · [Troubleshooting](docs/TROUBLESHOOTING.md) ·
[Upstream project](https://github.com/Niko1221/Strata)

## Original project documentation

The previous READMEs and translations are retained for reference:

[English](README.upstream.md) · [简体中文](README.upstream.zh-CN.md) · [日本語](README.upstream.ja.md) ·
[Deutsch](README.upstream.de.md) · [Français](README.upstream.fr.md) · [Español](README.upstream.es.md) ·
[Português](README.upstream.pt-BR.md) · [Polski](README.upstream.pl.md)

Uses the original project's [MIT License](LICENSE).
