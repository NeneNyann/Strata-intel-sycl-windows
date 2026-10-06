# Strata — Windows Intel Arc SYCL

English | [简体中文](README.zh-CN.md)  

A personal fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) for running Qwen3.8-Flash-Next on
**Intel Arc GPUs under native Windows**. It retains Strata's WebUI, OpenAI / Anthropic APIs and MCP tools,
and adds Windows SYCL builds, GPU vision and release packages.

Primarily tested on an **Arc B580 12 GB**. Intel support remains experimental.

## Install

Requires Windows 10/11, a discrete Intel Arc GPU, a current Intel graphics driver and an AVX2 CPU.
Choose a [model size](docs/MODELS.md#pick-by-ram) for your RAM and disk space; the test PC has 64 GB RAM.

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

Measured on 2026-10-07: Arc B580 12 GB, Core i7-14700K, 64 GB RAM, Qwen3.8-Flash-Next GSQ-RCO IQ2_XS,
INT8 KV, resident experts and MTP spec 4.

| Test | Context capacity | Average decode |
|---|---:|---:|
| Benchy v1: 2,185 input tokens, 256 output tokens, greedy | 8K | **31.46 tok/s** |
| Same prompt, GPU image encoder loaded | 128K | **31.19 tok/s** |
| Repeated request, 2,182 prompt tokens reused | 128K | **33.67 tok/s** |
| Tool prompt: 1,233 input tokens, temperature 0.6 | 128K | **24.30–25.34 tok/s** |

Speed varies with the prompt, MTP acceptance and available VRAM. 128K is the configured capacity;
these tests did not use full 128K-length prompts. See [Windows SYCL benchmarks](docs/SYCL_WINDOWS.md#benchmark)
for conditions and results.

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
[Português](README.upstream.pt-BR.md)

Uses the original project's [MIT License](LICENSE).
