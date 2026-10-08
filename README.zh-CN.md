# Strata — Windows Intel Arc SYCL

[English](README.md) | 简体中文  

这是 [Niko1221/Strata](https://github.com/Niko1221/Strata) 的自用 fork，主要用于在 **Windows 原生环境下使用 Intel Arc 显卡**运行 Qwen3.8-Flash-Next。保留原项目的 WebUI、OpenAI / Anthropic API 和 MCP 工具支持，加入 Windows SYCL 构建、GPU 图像编码及 release 安装包。

本仓库主要在 **Arc B580 12 GB** 上验证，Intel 支持仍属实验性。

## 安装

需要 Windows 10/11、独立 Intel Arc 显卡、最新 Intel 显卡驱动和支持 AVX2 的 CPU。按[模型大小](docs/MODELS.md#pick-by-ram)准备内存与磁盘空间；测试机器使用 64 GB 内存。

1. [下载本仓库](https://github.com/NeneNyann/Strata-intel-sycl-windows/archive/refs/heads/main.zip)并解压，或 clone 本仓库。
2. 在项目目录运行：

   ```bat
   START-HERE.bat --backend sycl
   ```

3. 按提示选择模型、上下文和是否启用图片。安装程序下载 engine、模型和 MTP 权重，然后启动 WebUI：`http://127.0.0.1:8080`。

Release 使用 `strata-windows-x64-sycl.zip`，包含 engine、GPU 图像编码器和运行库。使用安装包无需安装 oneAPI 或 Visual Studio。

后续启动运行 `START-HERE.bat`；更改设置添加 `--setup`。需要图片时运行：

```bat
START-HERE.bat --backend sycl --setup --vision gpu
```

MCP 连接配置见 [MCP 工具说明](docs/DETAILS.md#tools-from-mcp-servers)。其他安装与配置选项见 [安装文档](docs/INSTALL.md)。

## 性能

2026-10-09 实测：Arc B580 12 GB、Core i7-14700K、64 GB RAM，Qwen3.8-Flash-Next GSQ-RCO IQ2_XS，
INT8 KV、MTP spec 4。完整 RAM 专家驻留、固定专家缓存预算与 prefill chunk、全核，无 RAM 预算上限。
Benchy v1：2,185 输入、256 输出，greedy。

| 场景 | 上下文容量 | 首次 Decode | 重复 Decode |
|---|---:|---:|---:|
| 纯文本 | 8K | **42.57 tok/s** | **43.61–45.16 tok/s** |
| GPU 图像编码器驻留 | 8K | **39.46 tok/s** | **42.06–42.92 tok/s** |
| 纯文本 | 128K | **46.65 tok/s** | **46.92–48.22 tok/s** |
| GPU 图像编码器驻留 | 128K | **44.31 tok/s** | **46.53–47.00 tok/s** |

128K 表示配置容量，上述测试使用短提示词；重复请求复用 2,182 个输入 token。
WebUI 写作实测：128K 上下文容量，关闭 MCP tools 和 vision，28-token 提示词生成
5,871 token，平均 decode **36.2 tok/s**。

完整参数、实际专家 RAM/VRAM、prefill 分配及工具提示词结果见 [Windows SYCL benchmark](docs/SYCL_WINDOWS.md#benchmark)。

## 构建 release

安装 Visual Studio C++ 桌面工具、Intel oneAPI DPC++/C++ Compiler、oneMKL、CMake 3.24+、Ninja 和 Git，然后运行：

```bat
sycl\build-release-windows.bat
```

产物：`dist/strata-windows-x64-sycl.zip`。安装本地构建的包：

```bat
START-HERE.bat --backend sycl --prebuilt dist
```

## 更新与限制

`UPDATE.bat` 更新代码、引擎和已安装模型的配置。

- Windows 安装器目前支持单张独立 Arc 显卡。
- K8V4 KV streaming 不受支持；使用 `--kv int8`，或为 K8V4 设置 `--kv-streaming off`。
- GPU vision 已验证；完整 128K 长提示词和多图对话未做完整验证。

[Windows SYCL 使用说明](docs/SYCL_WINDOWS.md) · [故障排查](docs/TROUBLESHOOTING.md) · [上游项目](https://github.com/Niko1221/Strata)

## 原项目说明

保留原有说明及其翻译，供查阅通用功能：

[English](README.upstream.md) · [简体中文](README.upstream.zh-CN.md) · [日本語](README.upstream.ja.md) · [Deutsch](README.upstream.de.md) · [Français](README.upstream.fr.md) · [Español](README.upstream.es.md) · [Português](README.upstream.pt-BR.md)

沿用原项目的 [MIT License](LICENSE)。
