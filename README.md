<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/flyx-readme-banner-dark.svg">
    <img alt="FLYX: fast, lightweight local inference." src="docs/brand/flyx-readme-banner.svg" width="100%">
  </picture>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2d6a4f" alt="Apache-2.0 License"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555" alt="Windows and Linux">
  <a href="https://discord.gg/ZtUPXBxYhN"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Join FLYX Discord"></a>
  <a href="https://enwove.ai/flyx/wechatqr"><img src="https://img.shields.io/badge/WeChat-Join%20Group-07C160?logo=wechat&logoColor=white" alt="WeChat Group QR"></a>
</p>

<p align="center">
  <b>English</b> ·
  <a href="docs/i18n/README_zh.md">中文</a> ·
  <a href="docs/i18n/README_ja.md">日本語</a> ·
  <a href="docs/i18n/README_ko.md">한국어</a> ·
  <a href="docs/i18n/README_es.md">Español</a> ·
  <a href="docs/i18n/README_fr.md">Français</a> ·
  <a href="docs/i18n/README_de.md">Deutsch</a>
</p>

---

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#performance">Performance</a> ·
  <a href="docs/models.md">Models</a> ·
  <a href="docs/server.md">CLI &amp; API</a> ·
  <a href="llms.txt">llms.txt</a>
</p>

**FLYX** is a local LLM engine with zero dependencies. Run supported GGUF models directly and serve browsers, scripts and coding agents through OpenAI- and Anthropic-compatible APIs.

<p align="center">
  <img src="docs/media/dashboard-light.png" width="900" alt="FLYX built-in web dashboard showing live throughput, long-context sessions, VRAM and RAM usage, and recent agent requests">
  <br><sub>Built-in web dashboard: model management, chat, live throughput, memory breakdown, and one-click agent setup.</sub>
</p>

## Key Features

- Long context uses both VRAM and host RAM. The planner sizes each session to the memory available and reports the additional bytes needed when increasing one budget would fit the requested window.
- **Tuned per GPU, zero manual flags** — Ships measured kernel tuning tables for 11 NVIDIA GPUs and automatically sizes context length, K/V precision, and GPU/RAM placement from available memory. No guessing offload counts, batch sizes, or cache types.
- **No install, no dependencies, ~57 MB** — The single Windows executable includes the CUDA and C++ runtimes.
- **Built-in web dashboard & terminal monitor** — Starts straight into a browser UI (7 languages) with guided next-step hints to search/download Hugging Face or local models, chat, read localized API docs, and watch live throughput, VRAM/RAM split, sessions, requests, and logs ([monitor guide](docs/monitor.md)).
- **One-click coding agent integration** — Speaks OpenAI (Chat Completions, Responses) and Anthropic Messages with tool calls and multi-turn prompt reuse. Launch **Claude Code** or **Codex CLI** straight into your project folder from **Integration** (Windows), or copy ready-to-run commands for **DeepSeek Harness, OpenCode, and Pi**.
- Kernel tests compare CUDA results against float64 references. Release benchmarks report the workloads and devices on which speculative output IDs matched plain decoding. [Numerics](docs/design/numerics.md).

## Updates

- **v0.2.1** — Inkling-2-Mini adds sparse MoE inference, optimized decode kernels and automatic pairing with its Markov DSpark drafter. Qwen3.5/3.8 accepts images with a matching projector. The Hub discovers companions, and **Draft with it** attaches a selected drafter. Elastic loading adapts the drafter to available VRAM. Sliding-window DFlash2 uses a ring KV cache. The zero-PTX Windows candidate is 57.14 MB, down from 58.96 MB in v0.2.0.
- **`v0.2.0`** — **Initial open-source release of FLYX**: standalone ~59 MB zero-dependency Windows `.exe` & Linux tarball, native full-context GGUF & NVFP4 inference with automatic GPU/RAM K/V extension, built-in speculative decoding (MTP / DFlash2), 7-language web dashboard, and one-click Claude Code / Codex CLI integration.

## Performance

Qwen3.8-27B decode in tokens/s. The first three rates use three mixed prompts × 128 tokens. The last rate uses the first code prompt × 128 with lookup disabled.

| GPU | Qwen3.8-27B | Plain | MTP | DFlash2 | DFlash2 on code |
|---|---|---:|---:|---:|---:|
| RTX 5090 · Linux | NVFP4 | 93.8 | 286.6 | 315.5 | 565.5 |
| RTX 4090 · Linux | UD-Q4_K_XL | 49.4 | 158.0 | 171.6 | 305.7 |
| RTX 3090 · Linux | UD-Q4_K_XL | 42.8 | 121.7 | 131.2 | 234.1 |
| RTX 5070 Ti · Windows | UD-IQ3_XXS | 68.8 | 167.2 | 154.4 | 332.8 |

Inkling-2-Mini (UD-IQ3_XXS), plain → Markov DSpark (Q8_0), tokens/s. Each pair produced identical output IDs.

| GPU | chat_short · 8 × 128 | code_short · 8 × 384 |
|---|---:|---:|
| RTX 5090 · Linux | 225.5 → 259.8 | 228.5 → 438.9 |
| RTX 4090 · Linux | 166.7 → 202.9 | 167.9 → 322.4 |
| RTX 3090 · Linux | 126.2 → 136.5 | 126.9 → 222.0 |
| RTX 5070 Ti · Windows | 150.9 → 169.3 | 153.6 → 276.2 |

[Measurements and commands](docs/benchmarks.md#v021-four-gpu-verified-benchmarks)

## Quick Start

### 1. Download and run (recommended)

Grab the latest release from [GitHub Releases](https://github.com/enwove/flyx/releases) — requires only an NVIDIA driver (≥ 580):

- **Windows:** Double-click `flyx-0.2.1-windows-x86_64-cuda13.exe` (or run `.\flyx-0.2.1-windows-x86_64-cuda13.exe serve` in PowerShell). The server starts and opens the dashboard automatically (picking another free port if `8080` is busy).
- **Linux / WSL2:** Extract the tarball and run:

```sh
./bin/flyx serve
# Or load a Hugging Face GGUF directly into the standard HF cache:
./bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL
```

Linux Hub downloads use system libcurl.

The first command pairs Inkling with DSpark. The second sends an image to Qwen using the matching Hub projector.

```sh
flyx serve unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS
flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --image photo.png --chat -p "Describe the image."
```

Open the dashboard at `http://127.0.0.1:8080` → pick or download a model in **Model** → launch **Claude Code / Codex** directly (or copy any agent command) from **Integration**.

| Client / Endpoint | Base URL |
|---|---|
| OpenAI SDKs (Chat Completions, Responses, Completions) | `http://127.0.0.1:8080/v1` |
| Anthropic SDKs & Claude Code (`ANTHROPIC_BASE_URL`) | `http://127.0.0.1:8080` |
| Web Dashboard | `http://127.0.0.1:8080/` |
| Telemetry JSON (`flyx monitor`) | `http://127.0.0.1:8080/v1/telemetry` |

<details>
<summary><b>Build from source</b> (CUDA 13, CMake 3.24+, Ninja, g++ 11+ / MSVC 2022)</summary>

```sh
cmake --preset dev && cmake --build --preset dev    # outputs in ~/flyx-build/bin

# Run or serve directly by Hugging Face name
~/flyx-build/bin/flyx run unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --chat -p "Write a haiku about GPUs."
~/flyx-build/bin/flyx serve unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL --port 8080
```

> **WSL2 note:** Keep build trees and model files on the Linux filesystem (`~/models`, `~/flyx-build`); `/mnt/<drive>` degrades model load times severely.

</details>

<details>
<summary><b>Python usage &amp; curl example</b></summary>

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
import flyx

eng = flyx.Engine("~/models/Qwen3.8-27B-UD-Q4_K_XL.gguf", lib="~/flyx-build/bin/libflyx.so")
reply = eng.session().chat_full(
    [{"role": "user", "content": "Name the capital of France."}],
    enable_thinking=False,
    max_new_tokens=64,
)
print(reply["content"])
```

</details>

## Supported Models & Hardware

| Dimension | Supported | Notes |
|---|---|---|
| Architectures | GGUF: Qwen3 (`qwen3`), Qwen3.5/3.8 (`qwen35`), Inkling-2-Mini (`inkling2_loop`) | Qwen3.5/3.8 + `qwen3vl_merger` · Inkling-2-Mini + Markov DSpark · DFlash2 / DSpark: `dflash` · Llama: `.flyx` |
| Model formats | GGUF (`F32`, `F16`, `BF16`, `Q2_0`, `Q4_0–Q8_0`, `Q2_K–Q6_K`, `IQ1–IQ4`, `MXFP4`, GGML `NVFP4`); `.flyx` (BF16, NVFP4; format 2 with the target's own LoRA adapters, V-LoRA, run unmerged; a DV drafter, drafting only when named: `--drafter dv`, [design](docs/design/dv.md)) | Keep all split GGUF parts in the same folder. |
| GPUs | RTX 30 / 40 / 50 series, L4, L40S, A100, H100 (`sm_80`–`sm_120`; `sm_100` in Linux release) | Single GPU per process (`CUDA_VISIBLE_DEVICES`); driver ≥ 580 |
| Platforms | Windows (native MSVC), Linux, WSL2 | Single-file `.exe` / ZIP / wheel on Windows; tarball / `manylinux` wheel on Linux |

## Documentation

For AI agents, start with [llms.txt](llms.txt).

- **Use & Integrate:** [docs/server.md](docs/server.md) · [docs/models.md](docs/models.md) · [docs/monitor.md](docs/monitor.md) · [docs/benchmarks.md](docs/benchmarks.md)
- **Build & Test:** [docs/building.md](docs/building.md) · [docs/testing.md](docs/testing.md)
- **Design & Contracts:** [docs/design/architecture.md](docs/design/architecture.md) · [docs/design/numerics.md](docs/design/numerics.md) · [docs/design/gguf.md](docs/design/gguf.md) · [docs/design/tree-verification.md](docs/design/tree-verification.md) · [docs/design/dv.md](docs/design/dv.md)
- **Contribute & Governance:** [AGENTS.md](AGENTS.md) · [CONTRIBUTING.md](CONTRIBUTING.md) · [SECURITY.md](SECURITY.md) · [docs/prior-art.md](docs/prior-art.md)

## Contact Us

- **Discord:** [Join the FLYX Discord](https://discord.gg/ZtUPXBxYhN)
- **WeChat:** [Scan the WeChat group QR code](https://enwove.ai/flyx/wechatqr)
- **Issues & Feedback:** [GitHub Issues](https://github.com/enwove/flyx/issues)

For collaborations and partnerships, please contact us at [contact@enwove.ai](mailto:contact@enwove.ai).

## License

Apache-2.0. Third-party attributions are listed in [NOTICE](NOTICE) and [docs/prior-art.md](docs/prior-art.md).
