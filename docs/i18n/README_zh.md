<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../brand/flyx-readme-banner-dark.svg">
    <img alt="FLYX: fast, lightweight local inference." src="../brand/flyx-readme-banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="../../LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2d6a4f" alt="Apache-2.0 License"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555" alt="Windows and Linux">
  <a href="https://discord.gg/ZtUPXBxYhN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="加入 FLYX Discord"></a>
  <a href="https://enwove.ai/flyx/wechatqr"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="微信交流群二维码"></a>
</p>

<p align="center">
  <a href="../../README.md">English</a> ·
  <b>中文</b> ·
  <a href="README_ja.md">日本語</a> ·
  <a href="README_ko.md">한국어</a> ·
  <a href="README_es.md">Español</a> ·
  <a href="README_fr.md">Français</a> ·
  <a href="README_de.md">Deutsch</a>
</p>

---

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#实测性能">实测性能</a> ·
  <a href="../models.md">模型支持</a> ·
  <a href="../server.md">CLI 与 API</a> ·
  <a href="../../llms.txt">llms.txt</a>
</p>

**FLYX** 是一款零依赖的本地大模型引擎，Windows 单文件 `.exe` 仅约 57 MB：极速投机解码、原生长上下文、逐卡自动调优，免安装、免配置。直接加载受支持的 GGUF 模型，无需转换格式，通过 OpenAI 与 Anthropic 双协议 API 开箱接入浏览器、脚本与主流 Coding Agent。

<p align="center">
  <img src="../media/dashboard-zh-light.png" width="900" alt="FLYX 内置 Web 控制台：实时吞吐曲线、长上下文会话、显存/内存占比与近期 Agent 请求记录">
  <br><sub>内置图形控制台：模型管理、对话、实时吞吐与显存/内存监控、Coding Agent 一键接入。</sub>
</p>

## 核心特性

- **投机解码提速，输出一字不差** — 内置 MTP、DFlash2、DSpark 草稿器与提示查找：Qwen3.8-27B 在 RTX 5090 上写代码达 **565.5 tok/s**，是普通解码的 **6.03×**，且每个 token 都与普通解码的结果完全一致（[实测性能](#实测性能)）。
- **原生长上下文，不降智** — Qwen3.8-27B 的完整 262K 上下文，单张消费级显卡即可跑通，16 GB 显卡也不例外。K/V 缓存超出显存时无缝延展至系统内存，不截断、不滑窗：RTX 5090（`NVFP4`）在 256K 达 **164.5 tok/s**，24 GB RTX 4090 在 256K 仍达 **122.7 tok/s**，且 **32K–256K 大海捞针 100% 精准召回**（[v0.2.0 实测](../benchmarks.md#1-qwen38-27b-long-context-coding-benchmark)）。
- **逐卡调优，告别反复试超参** — 内置 11 款主流 NVIDIA 显卡的实测 kernel 调优表，启动时根据可用显存与内存自动配置上下文长度、KV 精度与显存/内存配比，不再手动折腾卸载层数、batch 和缓存类型。
- **免安装、免依赖、约 57 MB** — CUDA 与 C++ 运行库全部静态内置，只需 NVIDIA 驱动（≥ 580），不装 CUDA Toolkit、Python 或 Docker。Windows 是单个约 57 MB 的 `.exe`（双击即运行），Linux 为压缩包。
- **图像输入** — 具备视觉能力的模型配合 `mmproj` 投影器即可读图：命令行（`--image`）、OpenAI 与 Anthropic API、控制台粘贴或拖放均可。
- **自带图形面板 & 终端监控** — 启动自动打开多语言 Web 控制台（中/英/日/韩/德/法/西 7 种语言），带实时状态引导与完整本地化 API 文档；可直接检索加载 Hugging Face 或本地模型、发起对话，并实时查看速度、显存/内存分布、会话上下文、请求详情与日志（[面板说明](../monitor.md)）。
- **一键接入多个 Coding Agent** — 同时兼容 OpenAI（Chat Completions / Responses）与 Anthropic Messages 协议，支持工具调用与多轮前缀缓存；在面板 **Integration** 页即可**一键选目录拉起 Claude Code 或 Codex CLI**（Windows），或复制命令接入 **DeepSeek Harness、OpenCode、Pi**。
- **精度可验证，绝不暗中掉点** — 全算子对标 `float64` 参考实现回归测试；默认 FP8 KV 在 27B 上相对 16-bit 的 KL 仅 **0.0009–0.0016**；投机解码与普通解码**逐 token 完全一致**（[精度与数值契约](../design/numerics.md)）。

## 更新动态

- **`v0.2.1`** — **四卡实测速度、Inkling-2-Mini 与 DSpark、图像输入**：
  - **Qwen3.8-27B + DFlash2** 写代码：RTX 5090 达 **565.5 tok/s**（普通解码的 **6.03×**），RTX 4090 **305.7**，16 GB 的 RTX 5070 Ti **332.8**，RTX 3090 **234.1**。5090 上 `NVFP4` 的 W4A4 预填充达 **15,496 tok/s**。
  - **Inkling-2-Mini + DSpark** — 新增稀疏 MoE 模型，DSpark 自动挂载：5090 上写代码 **438.9 tok/s**（普通解码的 **1.92×**），4090 **322.4**，5070 Ti **276.2**，3090 **222.0**。
  - **视觉** — 具备视觉能力的模型配合 `mmproj` 投影器即可输入图像：命令行、OpenAI 与 Anthropic API、控制台粘贴 / 拖放。
  - **依然零依赖** — Windows 单文件 `.exe`，现约 57 MB。
- **`v0.2.0`** — **FLYX 首次开源发布**：提供约 59 MB 零依赖 Windows 单文件 `.exe` 与 Linux 压缩包，开箱支持 GGUF / NVFP4 原生全上下文推理与显存/内存自动延展、内置精确投机解码（MTP / DFlash2）、7 语言 Web 控制台，以及一键接入 Claude Code / Codex CLI。

## 实测性能

v0.2.1 解码速度（tok/s，贪心解码），四卡实测。**普通解码**不使用草稿器；两列**投机解码**分别在对话与代码提示上开启草稿器，括号内为相对普通解码的加速比。

**Qwen3.8-27B + DFlash2** — 对话 / 混合：3 条提示 × 128 token；代码：1 条提示 × 128 token。

| GPU | 量化 | 普通解码 | 投机解码 · 对话 / 混合 | 投机解码 · 代码 |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | 93.8 | **315.5** (3.36×) | **565.5** (6.03×) |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | 49.4 | **171.6** (3.47×) | **305.7** (6.18×) |
| **RTX 3090** (24 GB) | `UD-Q4_K_XL` | 42.8 | **131.2** (3.06×) | **234.1** (5.46×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 68.8 | **154.4** (2.24×) | **332.8** (4.84×) |

**Inkling-2-Mini + DSpark** — 对话 / 混合：8 条提示 × 128 token；代码：8 条提示 × 384 token。

| GPU | 量化 | 普通解码 | 投机解码 · 对话 / 混合 | 投机解码 · 代码 |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `UD-IQ3_XXS` | 225.5 | **259.8** (1.15×) | **438.9** (1.92×) |
| **RTX 4090** (24 GB) | `UD-IQ3_XXS` | 166.7 | **202.9** (1.22×) | **322.4** (1.92×) |
| **RTX 3090** (24 GB) | `UD-IQ3_XXS` | 126.2 | **136.5** (1.08×) | **222.0** (1.75×) |
| **RTX 5070 Ti** (16 GB) | `UD-IQ3_XXS` | 150.9 | **169.3** (1.12×) | **276.2** (1.80×) |

<sub>**×** 为同一组提示上相对普通解码的加速比，按未取整的速度计算；**普通解码**一列取自对话 / 混合测试。凡记录了 token ID 的测试，投机解码的输出均与普通解码逐 token 一致（仅 Windows 上的 Qwen 混合测试未记录 ID）。除 RTX 5070 Ti 为原生 Windows 11 外，其余均为 Linux。</sub>

<sub>**其他实测：** Qwen3.8-27B 内置的 **MTP** 草稿器（无需额外文件）在混合提示上达 **286.6 / 158.0 / 121.7 / 167.2 tok/s**（5090 / 4090 / 3090 / 5070 Ti）。**RTX 5090 `NVFP4`：** W4A4 预填充 **15,496 tok/s**；128K 上下文下 MTP 解码 **203.8 tok/s**（普通解码：73.4）；256K 下 **164.5 tok/s**，大海捞针召回 **100%**（v0.2.0 实测）。硬件、驱动与完整命令见 [docs/benchmarks.md](../benchmarks.md#v021-four-gpu-verified-benchmarks)。</sub>

## 快速开始

### 1. 下载即用（推荐）

从 [GitHub Releases](https://github.com/enwove/flyx/releases) 下载对应平台包，只需 NVIDIA 驱动（≥ 580）：

- **Windows**：下载单个 `flyx-0.2.1-windows-x86_64-cuda13.exe`，**双击即启动服务并打开浏览器面板**（若 8080 端口被占会自动顺延空闲端口）；也可在 PowerShell 中运行 `.\flyx-0.2.1-windows-x86_64-cuda13.exe serve`。
- **Linux / WSL2**：解压发布包后运行：

```sh
./bin/flyx serve
# 或直接按 Hugging Face 仓库名启动（自动下载到标准 HF 缓存目录）：
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

再试两条命令——DSpark 草稿器与视觉投影器会随模型一同下载：

```sh
# Inkling-2-Mini，自动挂载 DSpark
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
# 让模型看一张图片
flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --image photo.png --chat -p "Describe the image."
```

在浏览器 `http://127.0.0.1:8080` 的 **Model** 页下载或选择本地模型，随后到 **Integration** 页直接一键启动 **Claude Code / Codex**（或复制其他 Agent 的接入命令）。

| 客户端 / 接口 | 地址 |
|---|---|
| OpenAI SDK（Chat Completions、Responses、Completions） | `http://127.0.0.1:8080/v1` |
| Anthropic SDK 与 Claude Code（`ANTHROPIC_BASE_URL`） | `http://127.0.0.1:8080` |
| Web 控制台 | `http://127.0.0.1:8080/` |
| 实时遥测 JSON（`flyx monitor`） | `http://127.0.0.1:8080/v1/telemetry` |

<details>
<summary><b>从源码编译</b>（CUDA 13、CMake 3.24+、Ninja、g++ 11+ / MSVC 2022）</summary>

```sh
cmake --preset dev && cmake --build --preset dev    # 产物位于 ~/flyx-build/bin

# 直接按 Hugging Face 名称运行或启动服务
~/flyx-build/bin/flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --chat -p "用中文写一首关于 GPU 的俳句。"
~/flyx-build/bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port 8080
```

> **WSL2 提示：** 请将构建目录与模型文件放在 Linux 文件系统下（如 `~/models`、`~/flyx-build`）；放在 `/mnt/<盘符>` 会严重拖慢模型加载速度。

</details>

<details>
<summary><b>Python SDK 与 curl 示例</b></summary>

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好！"}]}'
```

```python
import flyx

eng = flyx.Engine("~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", lib="~/flyx-build/bin/libflyx.so")
reply = eng.session().chat_full(
    [{"role": "user", "content": "介绍一下 FLYX。"}],
    enable_thinking=False,
    max_new_tokens=64,
)
print(reply["content"])
```

</details>

## 支持的模型与硬件

| 维度 | 支持范围 | 说明 |
|---|---|---|
| 模型架构 | Qwen3.5 / Qwen3.8（`qwen35`）、Qwen3（`qwen3`）、Inkling-2-Mini（`inkling2_loop`，稀疏 MoE）、Llama（MiniCPM5-1B） | Llama 仅支持 `.flyx` |
| 投机草稿器 | MTP、DFlash2、DSpark、提示查找 | MTP 与 DSpark 自动挂载；DFlash2 为配套文件（`--drafter-file`，或控制台中的 **Draft with it**）；DSpark 配合 Inkling-2-Mini 使用 |
| 视觉 | 配合 `mmproj` 投影器输入图像，投影器自动从模型所在的 Hub 仓库下载 | 目前支持 Qwen3.5 / Qwen3.8；Inkling-2-Mini 仅支持文本 |
| 模型格式 | GGUF（`F32`、`F16`、`BF16`、`Q4_0–Q8_0`、`Q2_K–Q6_K`、`IQ1–IQ4`、`MXFP4`, `Q2_0`, GGML NVFP4）；`.flyx`（BF16、NVFP4；格式 2 可带目标模型自带的 LoRA 适配器（V-LoRA，不合并运行）；DV 草稿器，仅在指定时起草：`--drafter dv`，[设计](../design/dv.md)） | 分片 GGUF 的所有文件须放在同一目录。 |
| 显卡 | RTX 30 / 40 / 50 系、L4、L40S、A100、H100（`sm_80`–`sm_120`；Linux 发布包另含 `sm_100`） | 单进程单卡（`CUDA_VISIBLE_DEVICES`）；驱动 ≥ 580 |
| 平台与形态 | Windows（原生 MSVC）、Linux、WSL2 | Windows 提供单文件 `.exe` / ZIP / wheel；Linux 提供 tar.gz / `manylinux` wheel |

## 文档导航

AI Agent 请直接从 [llms.txt](../../llms.txt) 索引进入。

- **使用与集成：** [docs/server.md](../server.md) · [docs/models.md](../models.md) · [docs/monitor.md](../monitor.md) · [docs/benchmarks.md](../benchmarks.md)
- **编译与测试：** [docs/building.md](../building.md) · [docs/testing.md](../testing.md)
- **架构与数值契约：** [docs/design/architecture.md](../design/architecture.md) · [docs/design/numerics.md](../design/numerics.md) · [docs/design/gguf.md](../design/gguf.md) · [docs/design/tree-verification.md](../design/tree-verification.md) · [docs/design/dv.md](../design/dv.md)
- **贡献与规范：** [AGENTS.md](../../AGENTS.md) · [CONTRIBUTING.md](../../CONTRIBUTING.md) · [SECURITY.md](../../SECURITY.md) · [docs/prior-art.md](../prior-art.md)

## 联系我们

- **Discord 社区：** [加入 FLYX Discord](https://discord.gg/ZtUPXBxYhN)
- **微信交流群：** [点击查看最新入群二维码](https://enwove.ai/flyx/wechatqr)
- **问题反馈与建议：** [GitHub Issues](https://github.com/enwove/flyx/issues)

商务合作与技术交流请联系：[contact@enwove.ai](mailto:contact@enwove.ai)。

## 开源协议

Apache-2.0。第三方组件与算法致谢详见 [NOTICE](../../NOTICE) 与 [docs/prior-art.md](../prior-art.md)。
