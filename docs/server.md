# CLI, HTTP Server, and APIs

Single reference for the `flyx` CLI, `flyx serve` HTTP endpoints, environment variables, `--backend-opt` keys, and the C / Python APIs. `flyx-server` (symlink in Linux releases) aliases `flyx serve`.

---

## 1. CLI Subcommands

| Command | Synopsis | Description |
|---|---|---|
| `serve` | `flyx serve [MODEL] [serve-opts] [engine-opts]` | Start OpenAI + Anthropic HTTP API server and dashboard. Without `MODEL`, restores the last served model (`~/.config/flyx/serve.json` or `%APPDATA%\flyx\serve.json`), or starts unloaded (HTTP 503 until loaded via `/v1/load` or dashboard). |
| `run` | `flyx run MODEL (-p TEXT \| -f FILE) [--image FILE]... [--chat] [--no-think] [-n N] [opts]` | Single-shot generation. Use `-f FILE` for prompts > 128 KB; `--image` attaches local images (implies `--chat`). |
| `chat` | `flyx chat MODEL [opts]` | Interactive multi-turn terminal chat. |
| `bench` | `flyx bench MODEL [-n N] [--prompts FILE.jsonl] [--json [FILE]] [opts]` | Benchmark decode `tok/s`, tokens/round, and GPU telemetry across 3 fixed prompts, or across a prompt set (`--prompts`). `--json` emits one JSON record on stdout, or into `FILE`. |
| `tokenize` | `flyx tokenize MODEL "TEXT"` | Print token IDs for `TEXT`. |
| `template` | `flyx template MODEL MESSAGES.json [--no-think] [--chat-template-kwargs JSON]` | Render a JSON message array through the model's chat template. |
| `monitor` | `flyx monitor [URL] [--key KEY] [--interval S] [--once [--for S]]` | Read-only terminal monitor polling `GET /v1/telemetry` (default `http://127.0.0.1:8080`; `http://` URLs only). |
| `--version` | `flyx --version` | Print `flyx X.Y.Z` (also `server.version` in `GET /v1/telemetry`). `--help` after any command prints its usage on stdout with status 0; an option the command does not know is named in one line, status 2. |

---

## 2. Engine Flags (`serve`, `run`, `chat`, `bench`)

All commands that load a model share the parser in `apps/common/engine_flags.cpp`.

| Flag | Values | Default | Behavior |
|---|---|---|---|
| `--ctx` | `N` (`1..16777216`) \| `max` | `8192` (CLI) / `max` (`serve`) | Tokens per session. `serve` defaults to `max`: the model's trained context (or `262144` if unnamed), capped to what GPU + host RAM hold. CLI defaults to `8192` (shrinking down to `4096` if VRAM is tight). A context cut by memory is logged with which memory cut it and what would give more (other programs' share of the GPU, `--kv-ram`), as a warning under 32,768 tokens; weights and a session's fixed state stay on the GPU whatever host RAM there is, so a load can be refused with RAM to spare, and the refusal then says how much GPU memory is missing. |
| `--prefill-chunk` | `N` (≥ 1) | `1024` to `4096` by free GPU memory | Prompt rows per prefill pass. Buffers are shared across all sessions. |
| `--kv` | `fp8` \| `16` | `16` where 32 GB of GPU memory remain after the weights and on Llama models, else `fp8` | Target KV cache dtype: `fp8` (`e4m3` + per-position/head float32 scale, half memory/bandwidth) or `16` (FP16). Llama models (MiniCPM5-1B) default to `16`: FP8 moved their next-token distributions visibly. |
| `--kv-ram` | `GB` \| `auto` | `auto` | Pinned host RAM budget for KV positions beyond GPU capacity. `auto` takes available RAM minus `--ram-reserve` (default `max(1/16 of total RAM, 2 GiB)`; 32 GB machines keep 2 GiB, 64 GB 4 GiB; the remainder goes to `--host-cache`). `0` disables host KV offload. |
| `--kv-gpu` | `N` (≥ 0) | auto (max that fits) | Cap on KV positions kept on GPU; overflow goes to host RAM. |
| `--vram-reserve` | `GB` \| `auto` | `auto` | GPU VRAM kept free outside FLYX's ledger. `auto` keeps `max(384 MiB, total/40)` (`832 MiB` min on WSL2). Set `0` on a dedicated/headless Linux GPU to let FLYX use all available VRAM; backends report their own minimum floor (`256 MiB` on native Windows CUDA for WDDM, `0` on Linux/WSL2). |
| `--ram-reserve` | `GB` \| `auto` | `auto` | Host RAM kept free outside `--kv-ram` (`auto` = `max(total/16, 2 GiB)`). Set `0` on a dedicated VM/host to let FLYX use all available RAM. |
| `--kv-copy-gpu` | `N` (≥ 0) | auto (max that fits) | With host KV, max layers keeping their StrataFlux low-bit summary copy on GPU; remaining layers keep copies in host RAM and stage ahead. |
| `--kv-method` | `auto[:BITS]` \| `exact` \| `strata[:BITS]` | `auto` (`BITS=2`) | Decode KV read mode. `BITS` in `2..4` (default `2`). `auto`: exact attention up to `kv_gpu`, then StrataFlux once KV spills to host RAM; `exact`: exact attention everywhere; `strata`: StrataFlux everywhere. A file with a DV drafter, or a Qwen3 or Llama model with a DSpark drafter (in its file or `--drafter-file`), reads exactly under `auto`, drafting or not (their trees read every position, and naming the drafter changes no token); an EAGLE-4 drafter drafts its trees under `auto` only where the target's switch leaves the whole context exact, else its chains alone (the switch the target's either way). `strata` with DV drafting or draft trees is refused, and keeps a DSpark or EAGLE-4 drafter to chains. |
| `--backend` | `cuda` \| `cpu_ref` | the build's (`cuda`; `cpu_ref` in a build without it) | Backend shared library to load. A CUDA backend that cannot run here (no GPU, compute capability below 8.0, an old driver) is an error, not the float64 reference in its place; `--backend cpu_ref` asks for it. Select GPU with `CUDA_VISIBLE_DEVICES`. |
| `--backend-opt` | `K=V` (repeatable) | none | Backend-specific option (also `FLYX_BACKEND_OPTIONS=K=V,...`). Unknown keys fail load. |
| `--drafter` | `mtp` \| `dflash2` \| `dflash` \| `dspark` \| `dspine` \| `eagle4` \| `dv` \| `all` \| `none` | auto | Active drafters. Default enables all `--drafter-file` drafters plus the model's embedded MTP (or embedded block drafter: DFlash2, DFlash, DSpark or DSpine); an EAGLE-4 drafter takes MTP's place, so under the default and `all` MTP does not draft beside it (name `mtp` to draft with MTP instead). Multiple active drafters are selected dynamically per round. A DV drafter (in a DV `.flyx` file, [DV](design/dv.md)) drafts only when named (`dv` or `all`): its warm-up, its trees of 15 and of the file's width, plain decoding and lookup rounds are then chosen every round. |
| `--drafter-file` | `PATH` \| `HF_REPO` (repeatable) | none | Sidecar MTP GGUF (`MTP/mtp-*.gguf`), or a block drafter: DFlash2 (`.flyx`, HF repo directory, or a `general.architecture=dflash` GGUF), z-lab's DFlash (`.flyx` or HF repo directory, e.g. `z-lab/Qwen3-8B-DFlash-b16`), DSpark (DeepSpec's `.flyx` or HF repo directory, e.g. `deepseek-ai/dspark_qwen3_8b_block7` beside Qwen3-8B; llama.cpp's DSpark GGUF, `dspark-*.gguf`, beside an Inkling-2 model), Tencent's DSpine (`.flyx` or HF repo directory, e.g. `tencent/Qwen3-8B-DSpine-b16` beside Qwen3-8B) or EAGLE-4 (`.flyx` or HF repo directory, e.g. `EAGLE-Corp/Qwen3.8-27B-EAGLE4` beside Qwen3.8-27B, in MTP's place): one of DFlash2's block, DSpine or EAGLE-4 a model, and llama.cpp's DSpark beside it. |
| `--lora` | `DIR` \| `HF` | none | A LoRA adapter as Hugging Face PEFT saves it (`adapter_config.json` and `adapter_model.safetensors`): its folder, or a Hub repository (`org/name`, `org/name:FOLDER` for one in a folder of it, `@REV`), run unmerged beside the model's linears in every pass (a Hub adapter whose `adapter_config.json` is listed past 16 MiB, what a load reads, is refused before either file is fetched; any rank; Llama, Qwen3 and Qwen3.5/3.8: attention, MLP and gated delta net linears; GGUF block weights included). The command line's model takes it; a model loaded from the page takes the page's (`"lora"` of `POST /v1/load`). |
| `--mmproj` | `PATH` | auto (HF Hub) / none (local) | Multimodal vision projector GGUF (`mmproj-*.gguf`, e.g., `mmproj-BF16.gguf`) paired with the model file. Auto-fetched from Hugging Face Hub repos when available, for the architectures this build reads vision for (`qwen35`), unless `--no-vision` is set. |
| `--no-vision` | flag | off | Disable loading any vision encoder / `mmproj` projector even if available. |
| `--vision-offload` | flag | on (`disk`) | Shorthand for `--vision-device disk`: offload vision weights to an auto-deleted temp file on disk (0 VRAM, 0 locked host RAM) and stream via OS page cache into GPU staging. |
| `--vision-device` | `disk` \| `host` \| `gpu` \| `auto` | `disk` | Where vision encoder weights live (`disk`: auto-deleted temp file, 0 VRAM & 0 locked host RAM so host KV budget is untouched, 3-stage streamed with bit-identical outputs; `host`: pinned host RAM, 0 VRAM; `gpu`: resident on GPU; `auto`: GPU when >= 8 GB VRAM remain free after weights, else `disk`). |
| `--vision-quant` | `q8` \| `exact` \| `nvfp4` | `q8` | On-the-fly quantization of floating-point vision linear weights (`q8`: Q8_0 ~46% smaller; `exact`: keep file precision; `nvfp4`: ~70% smaller; pre-quantized GGML `mmproj` files stay packed). |
| `--vision-max-tokens` | `N` (`16..max_ctx`) | auto | Cap merged visual tokens per image across CLI, OpenAI, Responses, and Anthropic endpoints. |
| `--width` | `N` (`0..64`) | adaptive | Pin model drafter width (up to drafter max). Prompt lookup still runs unless `--no-lookup`, but not beside a DV drafter: its widest tree of at most `N` drafts then runs every round. |
| `--cycle` | `I,J,...` | off | Measurements and tests: launch these rounds in turn (indices into the rounds `-v` lists), no lookup round, instead of the controller's choice. An index past the rounds fails the load. |
| `--no-spec` | flag | off | Disable all speculative decoding (drafters + lookup). |
| `--no-lookup` | flag | off | Disable prompt-lookup rounds (15 drafts, wider with the `wide` backend option) on repeated text. |
| `--draft-trees` | flag | off (on with a DSpark drafter on a Qwen3 or Llama target, or an EAGLE-4 drafter) | Enable best-first draft trees (DFlash2, DSpark, EAGLE-4) alongside chains, chosen per round by measured cost and acceptance (also `FLYX_DRAFT_TREES=1`). |
| `--no-draft-trees` | flag | off | Chains only, even with a DSpark or EAGLE-4 drafter (also `FLYX_DRAFT_TREES=0`). |
| `--activations` | `PREFILL[/DECODE]`: `q8` \| `bf16` \| `mxfp8` \| `mxfp8/q8` \| `nvfp4/bf16` \| `exact` | as the file has it | The inputs of prefill passes, then of decoding rounds (the same where `/DECODE` is left out): see [models.md, Precision](models.md#precision). `q8`: GGML blocks' 8 bits (their default, llama.cpp's); `bf16`, `mxfp8`, `mxfp8/q8`: gpt-oss's MXFP4 experts as vLLM, as SGLang (sm_120), or SGLang's prefill with llama.cpp's decoding; `nvfp4/bf16`: NVFP4 weights' NVFP4 prefill even where the file declares bf16 (sm_120a, `K % 128 == 0`); `exact`: every format's own rounding everywhere. |
| `--chat-template` | `FILE` | model's template | Override model Jinja chat template from file. |
| `-v` | flag | off | Verbose logging (memory budget bill, load timings, per-request metrics, each CUDA round graph a session keeps and the size it is counted at, and as a session ends its rounds from a graph and without one). |

### Sampling & CLI-Specific Flags

| Flag | Commands | Values / Range | Default | Behavior |
|---|---|---|---|---|
| `--temperature` | all | `0..100` | `0` (CLI) / model card (`serve`) | Sampling temperature (`0` = greedy). |
| `--top-p` | all | `0..1` | `0` (CLI) / model card (`serve`) | Nucleus cutoff (`0` or `1` = off). |
| `--top-k` | all | `≥ 0` | `0` (CLI) / model card (`serve`) | Top-k cutoff (`0` = off). |
| `--presence-penalty` | all | `-2..2` | `0` (CLI) / model card (`serve`) | One-time logit penalty for tokens already in reply. |
| `--frequency-penalty` | all | `-2..2` | `0` | Per-occurrence logit penalty for tokens in reply. |
| `--seed` | CLI only | `uint64` | `0` (random) | Seed for position-keyed Gumbel-max sampling. |
| `--accept-min-p` | CLI only | `0..1` | `0` (off) | Lossy relaxed draft acceptance: accept draft with prob ≥ `P * p_max`. |
| `--accept-typical` | CLI only | `0..1` | `0` (off) | Lossy relaxed draft acceptance: accept draft with prob ≥ `min(E, sqrt(E) * exp(-H))`. |
| `-p TEXT` / `-f FILE` | `run` | string / path | — | Prompt text or file path. |
| `--image FILE` | `run` | path (repeatable) | none | Attach one or more local image files (PNG, JPEG, BMP, GIF, TGA) to the prompt (implies `--chat`; requires a loaded vision encoder via `--mmproj` or Hub auto-fetch). |
| `--chat` | `run` | flag | off | Wrap prompt as a single user message through the chat template. |
| `--chat-template-kwargs JSON` | `run --chat`, `chat`, `template`, `serve` | JSON object | `{}` | Default variables for the loaded template, e.g. `'{"preserve_thinking":true,"reasoning_effort":"medium"}'`. Use a JSON boolean for `enable_thinking`. The renderer supplies `messages`, `tools`, `add_generation_prompt` and the model's own tokens. |
| `--no-think` | `run`, `chat`, `template`, `serve` | flag | off | Disable thinking block by default (`enable_thinking=false`; a template that reads a thinking effort instead, Inkling-2's, is given its `reasoning_effort` `none`, and the reply is held to no thinking section); `bench` always runs without it. |
| `-n N` | `run`, `chat`, `bench` | `≥ 0` | `256` | Maximum new tokens to generate; `0`: the model's own limit if its file has one (`generation.max_new_tokens`, from `generation_config.json`), else up to the end of the context. |
| `--ignore-eos` | `run` | flag | off | Continue past EOS tokens until `-n` or context limit (`bench` always does). |
| `--json [FILE]` | `bench` | flag, or path | off | Emit structured JSON benchmark output on stdout, or into `FILE` when the next argument is not an option. |
| `--rounds FILE` | `run` | path | — | Write one JSONL record per decode round (`position`, `tokens`, `drafts`, `accepted`, and `prob`/`margin` when the drafter reports per-draft confidence). |
| `--prompts FILE` | `bench` | path | — | A prompt set, one JSON object a line (`{"question": ...}`, `{"messages": [...]}` or `{"ids": [...]}`, the prompt as tokens), each prompt in a fresh session to its end token (`-n` defaults to 512 for a prompt set); `--json` then records every prompt's tokens and every round (for `tests/e2e/check_dv.py`). |
| `--suffix S` | `bench --prompts` | string | none | Text after each question (`\n` in `S` a newline, `\\` a backslash, any other backslash itself). |
| `--limit N` | `bench --prompts` | `≥ 0` | all | Run the set's first `N` prompts. |

---

## 3. Server Flags (`flyx serve`)

`--chat-template-kwargs` applies to Chat Completions, Responses, Anthropic Messages, and Anthropic token counts.
`--no-think` overrides the startup object's `enable_thinking` in either flag order. Explicit request fields
(such as `reasoning_effort`, `reasoning.effort`, or Anthropic `thinking` and `output_config.effort`) override the
corresponding defaults. The request's `chat_template_kwargs` object overrides individual keys last. Other defaults
carry through. A startup `reasoning_effort` of `"none"` disables thinking unless overridden. When the resolved
mode conflicts with the startup effort, the template uses its own effort for that mode. An effort set by the
request is preserved. Each template determines which variables it reads.

In Windows PowerShell 5.1, escape the double quotes inside the single-quoted argument when launching a native
executable, for example `--chat-template-kwargs '{\"preserve_thinking\":true,\"reasoning_effort\":\"medium\"}'`.

To connect from another device, open **Server settings → Serve on local network** in the dashboard. Close flyx and launch it again, then copy an **Active API address** into the other device's client. Choose the address on the network the devices share. If the PC's firewall prompts, allow access on the intended network.

The switch saves `lan_access` in `serve.json`. It takes effect on the next launch of `flyx.exe` or `flyx serve`. Turning it off and restarting restores loopback-only access. Until restart, the dashboard shows the current listener and any pending change. An explicit `--host` overrides the preference and locks the switch for that launch. Devices that can reach the port can use inference. Set `FLYX_API_KEY` or pass `--api-key KEY` to require authentication.

| Flag | Values | Default | Behavior |
|---|---|---|---|
| `--host` | IP / hostname | saved LAN setting, initially `127.0.0.1` | Sets the listen address for this launch. `flyx serve --host 0.0.0.0` listens on every IPv4 interface. |
| `--port` | `0..65535` | `8080` | With `--port 0`, the OS assigns a free port and the server prints the dashboard URL at startup. If omitted and 8080 is held by another program, picks the next free port automatically (if held by an existing `flyx serve`, Explorer launches open its dashboard). |
| `--name` | string | filename / HF name | Model ID reported in `/v1/models` and responses. |
| `--models-dir` | `DIR` (repeatable) | none | Extra local directories scanned (depth 6) for `.gguf`/`.flyx` in `/v1/hub/cached` and dashboard (never deleted by `flyx`). |
| `--sessions` | `N` (≥ 1) | `8` (capped by memory) | Target concurrent GPU sessions (allocated on demand). Planner sizes full context for 1 session first, then attempts 2 within memory budgets when tight, admitting up to `N` sessions that fit (reported as `model.max_sessions`). |
| `--host-cache` | `GB` (≥ 0) | `8` | Pinned host RAM budget for parked session checkpoints (capped by RAM remaining after `--kv-ram`). `0` disables host snapshots. |
| `--kv-disk` | `DIR[:GB]` | off | Disk directory + GB cap (default `min(64 GB, 50% free space)`) for incremental prompt-prefix snapshots and spilled host checkpoints. Files are owned by the process and cleaned up on shutdown. |
| `--no-batch` | flag | off | Advance concurrent sessions one round at a time instead of batching rounds across sessions. |
| `--keep-billing-header` | flag | off | Retain Claude Code's per-request `x-anthropic-billing-header` line in system prompt (stripped by default to preserve KV prefix reuse). |
| `--api-key` | `KEY` | `FLYX_API_KEY` / `FLYX_KEY` | Require `Authorization: Bearer KEY` or `x-api-key: KEY` on all API routes and remote management calls. |
| `--no-auth` | flag | off | Explicitly disable API-key requirement for LAN dashboard and management endpoints (trusted networks only; ignores `FLYX_API_KEY` / `FLYX_KEY`; cannot be combined with `--api-key`). Cross-site browser requests are still blocked. |
| `--cors` | `ORIGIN` \| `*` \| `none` (repeatable) | this machine's pages | Origins whose pages a browser lets call the inference API: `https://host[:port]`, an extension's origin (`chrome-extension://ID`), `null` (a page opened from a file; any sandboxed frame has it too). `*`: any page (said aloud at the start when no key is set). `none`: not even the pages of this machine (`localhost`, `127.0.0.1`, `[::1]`, any port), which may by default. See *Browser clients* below. |
| `--max-tokens` | `N` (≥ 0) | `0` | The most tokens a reply takes when its request names no limit (`max_tokens` / `max_output_tokens` absent). `0`: the model's own limit if its file has one, else up to the end of the context, as llama.cpp, Ollama, vLLM and SGLang do. |
| `--queue` | `N` (≥ 1) | `16` | Maximum queued requests waiting for admission. |
| `--queue-timeout` | `S` (seconds) | `600` | Maximum queue wait before rejecting request. |
| `--idle-unload` | `S` (seconds) | `0` (never) | Free GPU sessions after `S` idle seconds (weights stay loaded). |
| `--keep-alive` | `S` (seconds) | `0` (never) | Unload model completely after `S` idle seconds; reloads on next request. |
| `--threads` | `1..4096` | `64` | HTTP worker threads. A request holds one while it is queued or running, so keep it above `--sessions` + `--queue`; connections past the workers wait (as many again), and the ones past that are closed. |
| `--no-monitor` | flag | off | Disable terminal dashboard (auto-disabled if stdout is not a TTY). |
| `--no-browser` | flag | off | Do not auto-open browser dashboard (auto-disabled in SSH/headless sessions). |
| `--no-request-log` | flag | off | Disable in-memory ring buffer of last 50 requests (`GET /v1/requests`). |
| `--no-system-log` | flag | off | Disable in-memory ring buffer of last 2000 stderr lines (`GET /v1/logs`). |
| `--dashboard-dir` | `DIR` | embedded | Serve dashboard static files from `DIR` (e.g., `apps/flyx-server/dashboard`) before embedded assets. |

> **Remembered model settings:** Starting `flyx serve` with no `MODEL` reloads the last model and its saved model-level settings (`--ctx`, drafters, KV placement, `--vram-reserve`, `--ram-reserve`, the LoRA adapter, and vision options `--mmproj`, `--no-vision`, `--vision-device`, `--vision-quant`, `--vision-max-tokens`) from `serve.json`. `serve.json` also persists `lan_access`, `remote_manage`, and a Hub mirror set via `POST /v1/hub/config` (`hub_endpoint`, applied at startup when `HF_ENDPOINT` is unset). To change saved model settings at startup, pass `MODEL` explicitly. Process flags (`--sessions`, `--kv-disk`, `--kv-method`, `--backend-opt`) must be passed on each launch.

---

## 4. Environment Variables

The engine and server register their environment variables in `core/os/env.h`. Explicit CLI flags take precedence where available.

| Variable | Values | Effect |
|---|---|---|
| `FLYX_BACKEND` | `cuda` \| `cpu_ref` | Default backend when `--backend` is unset. |
| `FLYX_BACKEND_DIR` | path | Directory searched for `libflyx-*.so` / `libflyx-*.dll`. |
| `FLYX_BACKEND_OPTIONS` | `K=V,K2=V2` | Prepended before `--backend-opt` flags. |
| `FLYX_DRAFTER` | `mtp` \| `dflash2` \| `dflash` \| `dspark` \| `dspine` \| `eagle4` \| `dv` \| `all` \| `none` | Default drafter mode when `--drafter` is unset. |
| `FLYX_ACTIVATIONS` | as `--activations` | Default activation mode when `--activations` is unset. |
| `FLYX_LOOKUP` | `0` \| `1` | Enable/disable prompt lookup (`0` = `--no-lookup`). |
| `FLYX_DRAFT_TREES` | `0` \| `1` | Enable/disable draft trees (`1` = `--draft-trees`, `0` = `--no-draft-trees`; unset: on with a DSpark drafter on a Qwen3 or Llama target, or an EAGLE-4 drafter on a Qwen3, Qwen3.5/3.8 or Llama target, EAGLE-4's under `--kv-method auto` only where the whole context reads exactly). |
| `FLYX_ADAPTIVE` | `0` \| `1` | Enable/disable intra-round confidence optimal stopping on adaptive drafter plans (`1`, default). Enabled for sequential drafters (MTP) and for drafters (DSpark, with step-wise Markov early stopping and optimal prefix verification, and DFlash2) on targets whose verify cost scales steeply with width (such as sparse MoE `inkling2_loop`, across checkpoints up to the drafter's trained block `{1, 2, 3, 5, 7}`, with sustained high-acceptance runs widening past the trained block when supported); dense targets paired with block drafters (DFlash2) verify the chooser's fixed widths. |
| `FLYX_LEARN_COSTS` | `0` \| `1` | Learn each round's cost at the context from the rounds a session reads alone (`1`, default), the load's timings the start; `0` keeps the load's timings throughout. The output is the same either way. |
| `FLYX_CHAT_NATIVE` | `0` \| `1` | Force Jinja engine (`0`) instead of native chat renderer (`1`, default). |
| `FLYX_API_KEY` / `FLYX_KEY` | string | Default server API key when `--api-key` is unset. |
| `FLYX_MODELS_DIR` | paths (`:` Linux / `;` Win) | Additional local model directories appended to `--models-dir`. |
| `FLYX_LIB` | path | Explicit `libflyx.so` / `flyx.dll` path for Python `flyx.Engine`. |
| `CUDA_VISIBLE_DEVICES` | GPU index list | Selects device 0 used by the CUDA backend. |
| `HF_HOME` / `HF_HUB_CACHE` | path | Hugging Face cache root / hub cache directory. |
| `HF_TOKEN` | token | Auth token for gated HF repositories (falls back to `hf auth login` token). |
| `HF_ENDPOINT` | URL | Hugging Face Hub mirror URL (takes precedence over a saved `hub_endpoint` in `serve.json` at startup; can still be overridden at runtime via `POST /v1/hub/config`). |
| `HF_HUB_OFFLINE` | `1` | Use local HF cache only; make no network requests. |
| `NO_COLOR` / `COLUMNS` | any / int | Terminal monitor formatting overrides. |

The CUDA backend also reads, at each model load, the environment spellings nine of its [Inkling-2 round options](#inkling-2-rounds) had before they were options: `FLYX_<NAME>=V` sets option `<name>`, and `FLYX_BACKEND_OPTIONS` and `--backend-opt` override it. A variable that is unset, empty, or not an integer (the whole value, no spaces) leaves the option's default; one that is not an integer also prints a warning. An integer keeps the meaning it had for the variable:

| Variable | Option | An integer `V` sets |
|---|---|---|
| `FLYX_SC_CLUSTER` | `sc_cluster` | `4` or `8`; any other `V`: `1` |
| `FLYX_MOE_VEC` | `moe_vec` | `1` for nonzero `V`, else `0` |
| `FLYX_WARM` | `warm` | the bits of `V` among `1`, `2`, `4` and `16` |
| `FLYX_EXPERT_PREFETCH` | `expert_prefetch` | `0` or `2`; any other `V`: `1` |
| `FLYX_GUESS` | `guess` | the low four bits of a positive `V`; zero or negative: `0` |
| `FLYX_GUESS_K` | `guess_k` | `V`; zero or negative: `0` |
| `FLYX_SIDE_LOOP_MSG` | `side_loop_msg` | `1` for nonzero `V`, else `0` |
| `FLYX_SIDE_COMPANION` | `side_companion` | `1` for nonzero `V`, else `0` |
| `FLYX_SIDE_KV` | `side_kv` | `1` for nonzero `V`, else `0` |

---

## 5. Backend Options (`--backend-opt K=V`)

Passed verbatim to the backend (`backends/cuda/backend.cpp`). Unknown keys or invalid values fail load immediately.

### Execution, Fusion, and Profiling

| Key | Allowed Values | Default | Effect |
|---|---|---|---|
| `profile` | `0` \| `1` | `0` | Print per-node GPU time (disables CUDA graphs). Also supported by `cpu_ref`. |
| `graphs` | `0` \| `1` | `1` | Capture and replay decode rounds as CUDA graphs (each session's memory budget includes room for its graphs; `0`: none counted). |
| `dv_conditional` | `0` \| `1` | `1` | DV's canvas pass in a round's graph as the body of a conditional node, skipped after a no-op round (`0`: inline, run every round; measurement and tests). |
| `fusion` | `0` \| `1` | `1` | Enable fused region overrides (`0` = generic ops only). |
| `gdn_round` | `0` \| `1` | `1` | Fused 1-kernel GDN round path for ≤ 4 rows. |
| `stacks` | `0` \| `1` | `1` | Combine adjacent mixed-type projection runs in one round launch. |
| `round_glu` | `0` \| `1` | `1` | Fuse SwiGLU into dp4a gate\|up GEMM on GGUF rounds. |
| `side_stream` | `0` \| `1` | `1` | Run narrow stack GEMMs concurrently on a side stream. |
| `lora_lanes` | `0` \| `1` \| `2` \| `4` \| `8` | `0` | Measurement: a LoRA adapter's second product with this many lanes an output (`0`: by its rank). Other bits than the default. |
| `lora_glu` | `0` \| `1` | `1` | The SwiGLU takes gate's and up's LoRA products for ranks of one lane (`0`: in their own launch). |
| `pdl` | `0` \| `1` | GPU table | Programmatic Dependent Launch (`sm_90+`). Default set by GPU tuning table (on for RTX 5090). |
| `wait` | `auto` \| `spin` \| `poll` | `auto` (Win) / `poll` (Linux) | Host wait mode for device completion (`auto` uses a high-resolution timer sleep followed by a short spin on Windows). |
| `tuning` | `table` \| `defaults` \| `NAME` | `table` | Kernel tuning table (`defaults` disables GPU table; `NAME` forces a named table e.g. `rtx4090`). |
| `log_shapes` | `0` \| `1` | `0` | Log each unique GEMM shape once at load/run. |
| `wide` | `auto` \| `0` \| `1` | `auto` | Wide prompt-lookup rounds: 31, 47 or 63 drafts verified at once (up to 64 rows, every row the bits of a 16-row round), taken after two 15-draft lookup rounds in a row are accepted whole (the width by the expected tokens per millisecond, from how often recent lookup drafts were rejected) and dropped when they yield fewer tokens per millisecond. `auto`: on where every layer's projections are NVFP4 and 24 GB of GPU memory remain after the weights; `0`: off; `1`: on (bf16 weights read once for every 16 rows, as NVFP4's; GGML block weights: 16-row strips). Not with penalties, relaxed acceptance or a grammar. The same rows carry a DV drafter's trees past 15 drafts: under `auto` a model that holds DV has the rows of DV's widest round (32 for MiniCPM5-1B's DV files, so lookup rounds of 31 there too), and with `0` DV drafts trees of 15 (its wider ones refused, said at load). Lookup rounds of any width run only where DV does not draft, and a lookup round leaves DV's canvas behind ([DV](design/dv.md)). Each session's buffers grow by the wider rows, and its graph room by the wider plans, both in the session size the session cap divides by (QUASAR's 27B: about 0.5 GB, most of it GDN replay inputs; 23.3 -> 23.8 GB a session at 262144 tokens; MiniCPM5-1B's DV files at 2048 tokens: 0.13 GB with `0`, 0.15 with DV's 32 rows, 0.17 with `1`). |

### Weights, Head, and Precision

| Key | Allowed Values | Default | Effect |
|---|---|---|---|
| `gemm` | `auto` \| `rows` | `auto` | Force CUDA-core row GEMMs (`rows`, debugging). |
| `blocks` | `packed` \| `bf16` | `packed` | `bf16` dequantizes GGUF block weights to bf16 at load (higher VRAM, outside packed contract). |
| `embed` | `auto` \| `host` \| `gpu` | `auto` | Untied embedding table in host RAM (`host`, read over PCIe) or on the GPU (`gpu`); `auto`: `gpu` where 32 GB of GPU memory remain after the weights, else `host`. |
| `vision` | `disk` \| `host` \| `gpu` \| `auto` | `disk` | Vision encoder weight placement (`disk`: auto-deleted flushed temp file, 0 VRAM & 0 locked host RAM, 3-stage streamed via OS page cache + 16 MiB lazy pinned ring; `host`: pinned host RAM, 0 VRAM; `gpu`: GPU resident; `auto`: `gpu` when >= 8 GB free after weights, else `disk`). |
| `vision_quant` | `q8` \| `exact` \| `nvfp4` | `q8` | On-the-fly quantization of floating-point vision linears (`q8`: Q8_0, `exact`: file precision, `nvfp4`: NVFP4; pre-quantized GGML blocks stay packed). |
| `head_w12` | `0` \| `1` | `1` | Lossless 12-bit packed bf16 output head for 1..16 rows (`0` saves ~0.95 GB on 4B, ~5% slower plain). |
| `head_cert` | `0` \| `1` | `1` | Certified FP8-filtered exact greedy head (bit-identical greedy tokens; `0` saves ~0.64 GB on 4B). A softcapped head (Inkling-2) is never certified: capped scores can tie where the raw ones do not, so greedy decoding reads the whole head. |
| `head_fp8` | `0` \| `1` | `0` | Lossy FP8 output head (trades accuracy for decode speed). |
| `draft_head` | `pick` \| `full` | `pick` | Draft candidate scoring: 1-bit prefilter of the top 256 (a DSpark walk's top 1024) + exact score (`pick`) vs full NVFP4 head (`full`). |
| `eagle4_tree` | `BEAMxSTEPS` (BEAM 1..8, 1 + BEAM × STEPS ≤ 64; STEPS 0: by the width, else 7 or more, the trunk's chain) | `4x0` | EAGLE-4's draft trees (of 15 drafts): the nodes each step keeps and expands, and the steps a tree takes (0: two for every three drafts, 10). |
| `draft_weights` | `auto` \| `nvfp4` \| `q8` \| `exact` | `auto` | Drafter weight precision. `auto` selects target-matched precision when VRAM allows (BF16 beside BF16 except where the model asks otherwise for its block drafter: Q8_0 for a DeepSpec DSpark's or a DFlash's, NVFP4 for a DSpine's; Q8_0 beside GGUF, NVFP4 beside NVFP4; MTP uses NVFP4) and drops to `nvfp4` when VRAM is tight; `nvfp4` forces 4-bit streaming quantization; `q8` uses Q8_0; `exact` keeps file precision and uncompacted draft head. |
| `draft_kv` | `fp8` \| `16` | `fp8` | MTP drafter KV cache dtype when target KV is FP8. |
| `act_group` | `32` \| `64` \| `128` \| `256` | `32` | Measurement only: 8-bit activation scale group size (use with `fusion=0,round_glu=0`). |
| `act_fp4` | `0` \| `1` \| `2` | `0` | Measurement only: round staged activations as NVFP4 (`1`) or MXFP8 (`2`). |
| `moe_dp4a` | `default` \| `original` \| `1x1` \| `1x2` \| `2x1` \| `2x2` \| `4x1` \| `4x2` | `default` | Measure packed expert dp4a GEMMs with staged 8-bit activations for IQ3_XXS, IQ2_S, IQ2_XXS, IQ2_XS, IQ4_NL and Q4_K in one configuration (`backends/cuda/variants/candidates.toml`, `moe_dp4a`). The two numbers select output columns per warp (gate/up pairs when fused) and K/32 units in flight per lane; these specialize grouped tiles containing one row. `default` takes the GPU's tuning table's rules for fused gate/up GEMMs (`moe_gate_up` under `variants/<arch>/tables/<sku>.toml`; `candidates.toml` defaults keep original tiles where a table has none) and the original tiles elsewhere; `original` keeps the original tiles everywhere. A release build (`FLYX_TUNE_CANDIDATES=OFF`) compiles non-original configurations only for fused gate/up in the row bands and architectures its tables reach, so forcing any non-original override requires `FLYX_TUNE_CANDIDATES=ON`. |

### Attention, Tiered KV, StrataFlux, and MTP Cache

| Key | Allowed Values | Default | Effect |
|---|---|---|---|
| `attention` | `auto` \| `split` \| `head` | `auto` | Attention kernel selection (`split` = FP32 split prefill instead of FlashAttention; `head` = ungrouped per-head prefill and decode/verify attention, over binary16 K/V only: `--kv 16`). |
| `rows_attention` | `mma` \| `cores` | `mma` | Decode/verify attention on tensor cores (`mma`) or FP32 CUDA cores (`cores`). |
| `rows_attention_from` | `N` (≥ 0) | `8192` | Key count threshold for switching decode attention to tensor cores. |
| `rows_attention_parts` | `1` \| `2` \| `3` | auto (`1` or `3`) | Binary16 split precision parts for `q` and `p` in tensor-core decode attention (`1` on GQA > 4 & HD ≥ 256, else `3`). |
| `rows_attention_q` | `16` \| `fp8` | `16` | Over FP8 KV on `sm_89+`, quantize query `q` to e4m3 and score on FP8 MMA (`sm_90+`) or binary16 tensor cores (`sm_89`, avoiding Ada's 22-bit FP8 accumulator). |
| `stage_ahead` | `0` \| `1` | `1` | Prefetch host KV / hosted StrataFlux copies concurrently with preceding layers. |
| `stage_dma` | `0` \| `1` | `1` | Use async H2D DMA transfers for committed StrataFlux prefixes (single-session decode) and aligned historical FP8 prefill windows. |
| `stage_blocks` | `N` (≥ 1) | `4` | CUDA blocks allocated to kernel-based `kv_stage` copy. |
| `stage_window` | multiple of 64 | `16384` | Device staging window positions for prefill over host KV. |
| `strata_from` | `N` (≥ 0) | Requested GPU position cap or estimated capacity of the text target alone | Position threshold where `--kv-method auto` switches from exact attention to StrataFlux. |
| `strata_denom` | `N` (≥ 0) | `16` (2-bit) / `64` (3–4-bit) | Refine 1 key pair in `N` per row (`0` = no refinement, scan copy only). |
| `strata_least` | `N` (≥ 1) | `32` | Minimum refined key pairs per row. |
| `strata_most` | `N` (≥ 0) | `256` | Maximum refined key pairs per row (`0` = unbounded). |
| `strata_recent` | `N` (≥ 0) | `0` | Extra most-recent positions always refined. |
| `strata_hot` | `0` or pow2 (`32..1048576`) | `2048` | Device LRU cache of refined host KV pairs per KV head (`0` = disabled). |
| `strata_copy_slots` | `2` \| `3` | `2` | Device double/triple-buffer slots for hosted StrataFlux copies when `stage_ahead=1`. |
| `strata_compact` | `0` \| `1` | `0` | Pack 256-dim hosted 3-bit StrataFlux copies (24 words/pos instead of 26). |
| `strata_scan` | `mma` \| `cores` | `mma` | Tensor-core (`mma`) vs FP32 (`cores`, measurement) StrataFlux scan. |
| `pageable_kv` | `auto` \| `on` \| `off` | `auto` | Spill tiered StrataFlux K/V overflow layers past the pinned host budget (WDDM non-local cap on Windows) into OS pageable RAM (`VirtualAlloc`/`mmap`) while keeping low-bit Strata copies and within-budget K/V layers pinned. |
| `pinned_host_mb` | `N` (≥ 0) | unset (auto) | Override the pinned host K/V budget in MiB for `pageable_kv=auto` (unset: the DXGI non-local budget on Windows, unlimited on Linux). |
| `strata_grid` | `range` \| `normal` | `range` | StrataFlux low-bit quantization grid mode. |
| `strata_chunk` | `N` (≥ 1) | `16384` | Selection chunk size in pairs (test/debug knob). |
| `mtp_window` | `N` (≥ 0) | `4096` | Above `8 * N` context, MTP reads last `N` positions exact (`0` = full exact device cache). |
| `mtp_copy` | `0` \| `2..4` | `2` | StrataFlux bit-width for MTP layer beyond `mtp_window` (`0` = disabled). |
| `mtp_sink` | `N` (≥ 0) | `64` | Initial sink positions kept exact for MTP when `mtp_copy=0`. |
| `mtp_ring` | `0` \| `1` | `1` | Ring buffer storage for MTP exact window: the window, one prefill pass and the sink, copied to host memory with every checkpoint as a sliding-window layer's ring is (`0` = linear slots, tests). A rewind whose window later positions overwrote in the ring is refused, and the session recomputes instead. |
| `dflash_ring` | `0` \| `1` | `1` | Ring buffer storage for sliding-window DFlash2 context K/V (`0` = linear slots, tests). A checkpoint, snapshot or rewind whose drafter window later positions overwrote in the ring is refused, and the session recomputes instead. |

### Inkling-2 Rounds

Every option here but `moe_vec` changes only where and when kernels run, never a result bit. Each also has a legacy environment spelling ([Environment Variables](#4-environment-variables)).

| Key | Allowed Values | Default | Effect |
|---|---|---|---|
| `moe_vec` | `0` \| `1` | `0` | NVFP4 experts with float32 inputs on column-at-a-time vector kernels at every width (`1`) instead of bf16 inputs on tensor cores: another arithmetic contract (both are tested against float64, [Numerics](design/numerics.md)), kept for measurement. |
| `sc_cluster` | `1` \| `4` \| `8` | `8` | A fused sublayer's row over a cluster of this many blocks (sm_90 and later, where the device fits the cluster; otherwise, and with `1`, one block). |
| `warm` | sum of `1`, `2`, `4`, `16` | `23` | L2 warming on the expert stream during rounds: the next MoE's router (`1`), loop controller (`2`), residual query and norm (`4`), and the next attention's small weights (`16`). `0`: none. |
| `expert_prefetch` | `0` \| `1` \| `2` | `2` | Guess the next MoE's experts from the residual and prefetch their weights: in rounds of 2 or more rows (`2`), in every round (`1`) or never (`0`). |
| `guess` | `0` to `15` | `5` | One-row rounds: a second guess from the mixed residual, prefetching these planes (bits: gate and up codes `1`, their scales `2`, down codes `4`, their scales `8`). `0`: none. |
| `guess_k` | `N` (≥ 0) | `4` | Prefetch the `N` best-scored guessed experts (`0`: all `top_k`). |
| `side_loop_msg` | `0` \| `1` | `1` | Loop messages on the side stream beside the sublayer's mix (needs `side_stream=1`). |
| `side_companion` | `0` \| `1` | `1` | A sink or tap projection on the side stream beside the wide GEMM that shares its input (needs `side_stream=1`). |
| `side_kv` | `0` \| `1` | `1` | A sliding-window round's K and V projections on the side stream beside the query path (needs `side_stream=1`). |

---

## 6. HTTP Endpoints & Schemas

Base URLs: OpenAI clients use `http://127.0.0.1:8080/v1`; Anthropic clients use `http://127.0.0.1:8080`. Max request body: `64 MiB`.

### Inference Endpoints

| Method & Path | API Family | Supported Request Fields & Output Contract |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI Chat | `messages` (text or array of `text` and base64 `image_url` data URI parts), `tools` (`strict: false` allows arbitrary JSON object args), `tool_choice` (`auto`, `required`, `none`, `{"type":"function","function":{"name":...}}`), `parallel_tool_calls`, `response_format` (`json_object`, `json_schema`), `stream`, `stream_options.include_usage`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `presence_penalty`, `frequency_penalty`, `accept_min_p`, `accept_typical`, `stop`, `reasoning_effort`, `enable_thinking`, `chat_template_kwargs`. Template kwargs override the corresponding request fields. Template rendering, reply grammar, and sampling defaults share the resolved boolean `enable_thinking`. Returns `reasoning_content`, `tool_calls`, and `usage.prompt_tokens_details.cached_tokens`. |
| `POST /v1/responses` | OpenAI Responses | `input` (string or items: messages with `input_text` / base64 `input_image`, `reasoning`, `function_call`, `function_call_output`), `instructions` (current request only; developer messages in `input` persist), `previous_response_id`, `store`, function `tools` (top-level or `namespace`), `tool_choice`, `parallel_tool_calls`, `text.format` (`json_object`, `json_schema`), `reasoning.effort` (`none` disables thinking), `max_output_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream` (`response.created` … `response.completed` / `response.incomplete`), `include: ["reasoning.encrypted_content"]`. Thinking returned as `reasoning` items (`reasoning_text`). Rejects built-in/custom tool types, `background`, `conversation`, `prompt` with HTTP 400. |
| `GET`, `DELETE /v1/responses/{id}` | OpenAI Responses | Fetch or delete a stored response. Serialized JSON store cap is `256 MiB` (FIFO eviction; responses exceeding cap are returned unstored). |
| `GET /v1/responses/{id}/input_items` | OpenAI Responses | Paginated input items: `order=asc\|desc` (default `desc`), `limit=1..100` (default `20`), `after=LAST_ID`. |
| `POST /v1/completions` | OpenAI Legacy | Raw string `prompt`. Omitting `max_tokens` (or `null`) defaults to `16` tokens. |
| `GET /v1/models` | OpenAI | Lists the active model (empty list when unloaded). |
| `POST /v1/messages` | Anthropic Messages | `system`, content blocks (`text`, base64 `image`, `tool_use`, `tool_result`, `thinking`), `tools`, `tool_choice` (`auto`, `any`, `tool`, `none`; `disable_parallel_tool_use`), `thinking` (`enabled`/`adaptive`/`disabled`; absent: no thinking, as Anthropic's API, unless the server's `--no-think` or `--chat-template-kwargs` set `enable_thinking` / `reasoning_effort`; `budget_tokens`: the reply's thinking ends at that many tokens, the format's closing marker written and the answer following), `output_config.format` (`json_schema`; incompatible with assistant prefill), `stop_sequences`, `stream`, `max_tokens`. Reused tokens reported in `usage.cache_read_input_tokens`. **Prewarm mode:** `max_tokens: 0` with `stream: false` prefills and caches the prompt (rejects enabled thinking, `output_config.format`, or required tool use with HTTP 400). |
| `POST /v1/messages/count_tokens` | Anthropic Messages | Returns prompt token count for a Messages payload without generating (including expanded vision tokens when a vision encoder is loaded). |

For image input, load a supported `qwen35` target and its matching `qwen3vl_merger` projector. The Hub can fetch it automatically, or select a local companion with **Use as projector**. The dashboard accepts pasted or dropped images. In a user message, send one of these content blocks alongside a text instruction (replace `BASE64` with the encoded PNG bytes).

| Endpoint | Image content block |
|---|---|
| `/v1/chat/completions` | `{"type":"image_url","image_url":{"url":"data:image/png;base64,BASE64"}}` |
| `/v1/responses` | `{"type":"input_image","image_url":"data:image/png;base64,BASE64"}` |
| `/v1/messages` | `{"type":"image","source":{"type":"base64","media_type":"image/png","data":"BASE64"}}` |

### Management, Telemetry, and Hub Endpoints

`GET /health` answers anyone who reaches the port. `GET /v1/telemetry` and `GET /v1/load` take the API key when one is set (HTTP 401 without it); without `--api-key`, they answer direct loopback clients and direct, unforwarded LAN clients sending a trusted `Host` matching the socket/listener address or configured/system hostname (HTTP 403 otherwise; cross-site browser requests are blocked). The others (`/v1/requests`, `/v1/logs`, `POST`/`DELETE /v1/load`, `/v1/unload`, `/v1/hub/*`, `/v1/settings/network`) are for this machine, with `--api-key` for the key's holder, or from a direct, unforwarded LAN client when bound to a non-loopback address with `--no-auth` or the saved `remote_manage` toggle enabled (HTTP 403 otherwise; cross-site browser requests are always blocked, loopback listeners require a loopback `Host`, and keyless LAN management requires a `Host` matching the socket/listener address or configured/system hostname while non-browser LAN inference clients keep flexible `Host` handling). `POST /v1/hub/config` always requires this machine or `--api-key`. This machine means a connection from the loopback address that no proxy passed on: a request carrying `Forwarded`, any `X-Forwarded-*` header, `X-Real-IP`, `Via`, `CF-Connecting-IP` or `True-Client-IP` is never treated as loopback or unforwarded LAN. An opaque TCP tunnel or a proxy that strips forwarding headers cannot be detected from HTTP headers alone, so always set `--api-key` behind a reverse proxy or tunnel.

| Method & Path | Behavior & Response Fields |
|---|---|
| `GET /health` | `{"status":"ok", "model": ID\|null, "queued": N, "loaded": bool, "load": state}` |
| `GET /v1/telemetry?recent=N` | Schema version 1 live metrics snapshot (`recent` default `20`, max `100`). See [docs/monitor.md](monitor.md). |
| `GET /v1/requests?limit=N` | Last ≤ 50 finished requests (in-memory only; supports `ETag` / `If-None-Match` → 304). Fields: `enabled`, `requests[]` (`id`, `api`, `model`, `started_at`, `finished_at`, `stream`, `status`, `error`, `usage`, `ttft_ms`, `request`, `reasoning`, `content`, `tool_calls`, optional `timing`, `cancel_reason`, `client_hints`). Clips large request/reply strings (64 KiB cap; down to 1 KiB if body > 256 KiB); of a reply's tool calls the first 64 are kept, each argument string clipped at 64 KiB. |
| `GET /v1/logs?after=SEQ&limit=N` | Last ≤ 2000 stderr lines (8 KiB/line, in-memory; default `limit=500`). Returns `enabled`, `lines[{seq, t, text}]`, `next`, `missed`. |
| `GET /v1/load` | Current load/served status: `state` (`none`, `fetching`, `loading`, `ready`, `failed`), `progress`, `error`, `fallback` (always `null`: a model that fails to load is not replaced by the one served before; kept for clients that read the field), `degraded` (list of `{kind, from, to, detail}` VRAM adjustments made during elastic load; empty when full settings fit), `notes`, `drafters` (actively loaded drafters), `suggested` (the verified drafters for the served model it was not given, as `/v1/hub/files` gives `known_drafters`: `[{repo, kind, base, exact, relation, size, tok_s}]`, `tok_s` for the model served), `vision_projector` (`{from, size}` or `null`), active options (`loaded`: the load as it was asked, for a reload), `loaded_file` (the model file served, an absolute path, whatever name or path the load was given), and `remembered` saved settings. |
| `POST /v1/load` | Async model load/switch (HTTP 202; HTTP 409 if another load is active). Body: `{"model", "drafters", "ctx" (N\|"max"), "drafter", "speculative", "kv" ("fp8"\|"16"\|"auto"), "kv_ram", "kv_gpu" (positions\|"auto": as many as fit, whatever the server's `--kv-gpu`; null/absent: the server's own), "vram_reserve" (GB\|"auto"), "ram_reserve" (GB\|"auto"), "width", "draft_trees" (`true`, `false`, or `null`/absent: the drafter's default, as saved), "lookup", "draft_head", "draft_weights" ("auto"\|"nvfp4"\|"q8"\|"exact"), "lora" (a PEFT adapter's folder or Hub name; null: none), "vision", "mmproj", "vision_device" ("disk"\|"host"\|"gpu"\|"auto"), "vision_quant" ("q8"\|"exact"\|"nvfp4"), "vision_max_tokens", "name"}` (`"kv": "auto"`: the engine's default cache type, whatever the server's `--kv`; the dashboard's auto choice sends it). Fetches while current model serves; on load failure after unloading, stays in `state: "failed"` with no model served. Saved settings in `serve.json` always keep the requested configuration even if tight VRAM caused elastic load adjustments. `DELETE /v1/load` aborts an in-progress fetch. |
| `POST /v1/unload` | Stops active requests and unloads model (`state: "none"`, inference returns 503). Preserves `serve.json`. HTTP 409 if loading. |
| `GET /v1/device/processes?kind=vram\|ram` | On-demand list of other applications occupying GPU VRAM (`kind=vram`, default) or host RAM (`kind=ram`): `kind`, `supported`, `can_close`, `device`, `mem_total`, `mem_used`, `mem_free`, `own_bytes`, `others_bytes`, `system_bytes`, `idle_igpu`, `apps[]` (`id`, `name`, `process`, `bytes`, `local_bytes`, `processes`, `closable`, `reason`). |
| `POST /v1/device/close` | Closes (`{"id", "force": false, "kind": "vram"\|"ram"}`, default graceful `WM_CLOSE`/`SIGTERM`) or force-quits (`"force": true`) an application currently listed by `GET /v1/device/processes`. Loopback requests only (HTTP 403 otherwise); refuses system processes, ancestor processes running FLYX, and other users' processes (HTTP 403). Answers `{"status": "closed" \| "still_running" \| "no_window", "windows_signaled", "name", "snapshot"}`. |
| `POST /v1/agents/launch` | Starts a coding agent against the served model in a new Windows PowerShell window on the server's machine (body `{"agent": "claude" \| "codex", "dir"}`). For Claude Code, the optional `permission_mode` accepts `default`, `acceptEdits`, `auto`, or `bypassPermissions`. Omitting it uses Claude Code's settings. Without `dir`, the desktop's folder dialog asks for the project folder. The server writes the command itself (the Integration page's, Codex's profile as `-c` overrides, no file of the user's written); an API key reaches the window as `FLYX_KEY` in its environment, never on a command line. Answers `{"launched": true, "dir"}`, or `{"launched": false}` when the dialog is cancelled. Loopback requests only (HTTP 403 otherwise); HTTP 409 with no model served, a dialog already open, or (Codex) a model id with one of `" & \| < > ^ %`, which PowerShell or `codex.cmd`'s cmd.exe cannot pass on; Codex is run as `codex.cmd` or `codex.exe`, so a system that runs no PowerShell scripts still starts it; HTTP 501 where no window can be opened (not Windows, SSH, a service). `telemetry.server.agent_launch` lists the agents it can start. |
| `GET /v1/settings/network` | Returns the saved `lan_access`, saved/active `remote_manage`, `remote_manage_effective` (true only when `serving_lan` is true, no API key is set, and `--no-auth` or `remote_manage` is active), `no_auth_from_cli`, `api_key_set`, effective `listen_host` and `port`, `serving_lan`, `host_from_cli`, `restart_required`, client `api_urls`, and `settings_error` (null, or why `serve.json` was not read: its lock held by another flyx server, or the file unreadable, not JSON or no object, which a save then leaves as it is; the values are then the last this server read or wrote). Both network settings operations require management access. |
| `POST /v1/settings/network` | Updates `lan_access` (takes effect on next launch; HTTP 409 if `--host` was passed) and/or `remote_manage` (takes effect immediately while `serving_lan` is active, else after LAN access is enabled and flyx is restarted; HTTP 409 if `--no-auth` or `--api-key` is active), persists in `serve.json`, and returns the updated network settings. |
| `GET /v1/hub/config` | Returns the active Hugging Face Hub endpoint configuration: `{"endpoint", "default_endpoint", "source" ("setting"\|"HF_ENDPOINT"\|"default"), "offline"}`. |
| `POST /v1/hub/config` | Updates the active Hugging Face Hub endpoint (`{"endpoint": URL}` or `""` to reset), normalizes/validates the URL, persists it as `hub_endpoint` in `serve.json`, and returns the updated config JSON. Requires loopback or `--api-key` (never keyless LAN); `HF_TOKEN` is never forwarded to runtime-saved mirrors or `http://` URLs. HTTP 400 on invalid URLs. |
| `GET /v1/hub/search?q=TEXT&sort=S&limit=N&cursor=C&arch=A` | Lists HF repos holding GGUF files whose names match `q` and, after them on each page, the matching drafters whose files are safetensors (DFlash2, DFlash, DSpark, DSpine, EAGLE-4 configs: a drafter is one by what the Hub says of its config or GGUF architecture, never by words of the query). The two are lists of their own, each in `sort`'s order and up to `limit` long: a page can hold twice `limit` rows (one more where a `q` naming a repository, `org/name`, is looked up directly on the first page) and is not one list sorted across both. Each repository is on a page once, and an `org/name` repository shown on the first page is on no later page; the paging keeps no other state, so a repository the Hub's listing moves between two requests (or a drafter the Hub lists without its GGUF architecture, in both lists) can come again. `next_cursor` (opaque) resumes both; a drafter list the Hub does not answer beside GGUF results ends with that page, and the only list left unanswered returns the Hub's error (`429` with `Retry-After`, `502`), the same `cursor` good to ask again; a cursor of this search's form that is not one is `400` (`sort`: `trending` (default), `downloads`, `likes`, `updated`, `created`; `limit` 1-100, default 20; `cursor` from previous `next_cursor`): `{"query", "family", "architectures", "arch", "endpoint", "next_cursor", "models": [{"id", "downloads", "likes", "updated", "created", "pipeline", "gated", "cached", "architecture", "runs", "drafter"}]}`. Without `q`, browses supported architectures (`"family": true`), optionally narrowed to `arch`. Returns typed HTTP errors (`400`, `403`, `404`, `429` with `Retry-After`, or `502` when unreachable / `HF_HUB_OFFLINE`). |
| `GET /v1/hub/files?repo=ORG/NAME[@REV]` | Lists a HF repo's GGUF quants (preserving `@REV` in `ggufs[].name` and `companions[].name` when pinned), sizes, cache status (`ggufs[].disk`: the cached file's path, else `null`), `mtp` and `mmproj` defaults, full `companions` list (`[{name, path, file, kind, drafter_kind, size, cached, disk, default}]` covering drafters, `kind` `mtp`, `dflash2`, `dflash`, `dspark`, `dspine` or `eagle4` (from a cached file's header, else its name), `drafter_kind` the same from a cached file's header alone (`""` otherwise: the mode a load may be told), and `mmproj` projectors; no projector is listed, and `mmproj` is `null`, for a repository whose architecture this build reads no vision for; the default DSpark drafter is the one a load takes by default), `known_drafters` (the verified drafters for its base models, `[{repo, kind, base, exact, relation, size, tok_s}]`: `exact` for one made for this repository, else inferred from a base model it names, `relation` how the Hub says it was made from it; each tier the fastest first, `tok_s` this machine's own mean decode speed drafting with it for this repository's models, `null` before a run: [models.md](models.md)), `drafter` (a drafter repository's `size` and `kind`), `drafter_repo` (`true` for `general.architecture=dflash` GGUF drafter repos), `endpoint`, `info` (`downloads`, `likes`, `updated`, `license`, `pipeline`, `gated`, the GGUFs' `architecture`, `context`, `parameters`; `null` when answered from the cache), `runs` (whether this build opens that architecture; `null` if unknown) and `build_architectures`. |
| `GET /v1/hub/cached` | Lists all cached HF snapshots (`org/name:QUANT` or `org/name:QUANT@COMMIT`, with `"mmproj": "in-repo"` (only for a model of an architecture this build reads vision for) or `"dflash2": "in-repo"` for companions in the same snapshot, and `"mtp": "in-repo"` for a separate MTP layer in any cached snapshot of the repository) and local `--models-dir` files (each tagged with its configured `"root"`), plus `"dirs"` (canonicalized `--models-dir` roots) and `"win_paths"` (server path semantics); a vision projector, cached or local, is listed with `"kind": "mmproj"`. Every GGUF is classified by its header (read once and cached), its name deciding only where the header cannot be read: a drafter is listed with `"kind": "drafter"` and its `"drafter_kind"` (`mtp`, `dflash2`, `dflash`, `dspark`, `dspine` or `eagle4`): a drafter's directory, an `mtp-*.gguf`, and any GGUF whose header says `general.architecture=dflash` (`dspark` with `dflash.has_markov_context`), cached or local; and its `"drafter_mode"`, the kind its header (or a drafter directory's config) proves, which is what a load is told (`""` where only the name says: an `mtp-*.gguf` of an ordinary model with no MTP layers is `drafter_kind` `mtp`, `drafter_mode` `""`); a model's entry its `known_drafters`, as `/v1/hub/files` gives them, for the repositories its file says it comes from. LoRA adapters are listed with `"kind": "lora"` (a PEFT folder, or a cached repository's `org/name[:FOLDER]`), which `"lora"` of `POST /v1/load` takes. |
| `GET /v1/hub/drafter?name=NAME` | The drafter a path or a Hugging Face name holds, as its own fields say: `{"name", "kind"}` (`mtp`, `dflash2`, `dflash`, `dspark`, `dspine`, `eagle4`; `null` when they say none). A path is read from its header (a GGUF's `general.architecture`, a drafter directory's `config.json`, a `.flyx` file's `draft`), a Hub name from what the Hub says of the repository (a drafter repository's `config.json`) and, for a GGUF, from the header of the file it selects once that is in the cache (`null` until then: a `dflash` GGUF is DFlash2's or DSpark's by its header alone, and the dashboard then leaves the mode to the loader); nothing is fetched, and no name is read for a kind. `400` for a `name` that is neither, `404` for a Hub name the Hub does not have. |
| `GET /v1/hub/loras?base=ORG/NAME` or `?model=NAME` | The LoRA adapters Hugging Face lists as made for a base model (`filter=base_model:adapter:BASE`), most downloaded first: `{"base", "from", "models": [{"id", "downloads", "likes", "updated", "gated", "cached"}], "next_cursor"}`. With `model` (a Hub name or a file of the HF cache) the base is that repository's model card's first `base_model` (`"from": "model card"`), or the repository itself; for another file `"base": null`. `limit` (1–100, default 30), `cursor`. HTTP 502 when the Hub cannot be asked (offline, unreachable). |
| `GET /v1/hub/lora?repo=ORG/NAME[@REV]` | A repository's adapters, one per folder with an `adapter_config.json` (at most 32): `{"repo", "revision", "adapters": [{"folder", "name", "rank", "scale", "base", "targets", "linears", "layers", "size", "why"}]}`. `name` is what `--lora` and `"lora"` take; `why` is `null` when `--lora` reads it, else the reason a load would give: the config (at most 64 KiB) checked as a load checks it, and the tensors' names, dtypes, rank and pairs (each linear's `lora_A` with its `lora_B`, as a load pairs them) from the safetensors header alone (two ranged requests). Both are read into memory: nothing is downloaded to the cache. `base` and `targets` are the config's strings (a regex `target_modules` is one), cut at 120 bytes. |
| `GET /v1/hub/dir?path=PATH` | Browses a folder of the server's machine: `{"path", "parent", "kind": "dir", "dirs": [{"name", "path"}], "models"}` with its subfolders and the `.gguf`/`.flyx` model files in it and two folders down (entries as in `/v1/hub/cached`, loadable by `name`). A model file's path answers `"kind": "file"` with that file. Without `path`: `"kind": "start"` with where to begin (`--models-dir` folders, the home directory, on Windows the local disks). Remote requests refuse Windows UNC/device paths, mapped network drives (`DRIVE_REMOTE`), and reparse points / symlinks targeting them outside `--models-dir`. HTTP 404 for a path that is neither. |
| `POST /v1/hub/dir` | Opens the desktop's own folder dialog on the server's machine (body `{"title", "start"}`, both optional) and waits: answers as `GET /v1/hub/dir` for the folder chosen, `{"path": null}` when cancelled. Loopback requests only (HTTP 403 otherwise); HTTP 501 where no dialog can be shown (SSH session, no display, a service; Linux needs `zenity` or `kdialog`); HTTP 409 while one is open. |
| `DELETE /v1/hub/cached?name=NAME` | Deletes a cached HF snapshot + unreferenced blobs (`{"deleted", "freed"}`). Refuses active models (409) and `--models-dir` files (400). |
| `GET /`, `GET /dashboard` | Browser dashboard UI (non-browser clients at `/` get JSON endpoint index). |

To load Inkling and discover its DSpark companion, post `{"model":"unsloth/Inkling-2-Mini-GGUF:UD-IQ3_XXS"}` to `/v1/load`. To choose files explicitly, pass their paths in the `drafters` array and the mode in `drafter`. A 202 starts the operation. Poll `GET /v1/load` until `ready` or `failed`. **Draft with it** applies a companion to the chosen target. Local and cached GGUF classification uses header metadata, including the Markov discriminator. Remote listings can use repository metadata and filenames until the file is available locally.

### Semantics & Error Codes Agents Should Expect

| Condition | Behavior / HTTP Status |
|---|---|
| Unloaded server | `503` until a model is loaded. |
| Context length exceeded | `400` before stream starts: OpenAI `context_length_exceeded` (`"This model's maximum context length is ..."`), Anthropic `"prompt is too long: N tokens > M maximum"`. If queued > 10 s on a streamed request, delivered as an in-stream error event. |
| Unsupported modality / tool format | `400` for remote image URLs, images sent when no vision encoder (`--mmproj`) is loaded, audio/video/document inputs, or `tools` on a model whose template has no supported tool-call format. Request fields not listed above (`logprobs`, `logit_bias`, `repetition_penalty`, `min_p`, the legacy `functions`, ...) are ignored. |
| Auth failure | `401` (invalid/missing API key); `403` (remote client accessing local-only management endpoint without API key). |
| Request superseded (same conversation retry) | Running request cancelled (`cancel_reason: "superseded"`). A legacy `/v1/completions` prompt never supersedes another request, nor is it superseded: raw text carries no conversation, so two clients sending the same or overlapping prompts each get their own reply. Anthropic queued/non-streaming returns `409` (`conflict_error`, `x-should-retry: false`); OpenAI Responses returns `status: "incomplete"` / `response.incomplete`. |
| Client isolation headers | Claude Code session/subagent headers and Codex `thread-id` headers isolate independent threads even with identical prompt histories. |
| Default sampling (when request omits params) | Qwen3.5 model card defaults: thinking ON → `temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0`; thinking OFF → `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`. For tool/code use without thinking, pass `presence_penalty: 0` (or `--presence-penalty 0`). |
| Grammar enforcement | Enabled automatically when `tools` or `response_format` / `output_config.format` is set. Compiled once per tool/schema set and cached; checked after each speculative round with rollback on rejected tokens. |
| Tools or a schema no grammar can be made from | `400` (`invalid_request_error`, the message beginning `reply grammar:`) when the request requires the grammar (`response_format`, `tool_choice` `required` or a named tool); with `tool_choice: auto` the request runs unconstrained and the server logs why. |
| Browser clients (CORS) | A page of an allowed origin (see `--cors`) may call `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/models` and `/health`: its preflight (`OPTIONS`, which carries no key) is answered `204`, Chrome's `Access-Control-Request-Private-Network` included, and answers carry `Access-Control-Allow-Origin` with that origin. A page of any other origin gets `403`, whose message and the server's log name the `--cors` value that would let it in. The model panel, the request log, the logs and the telemetry are never another origin's, whatever `--cors` says. A request without an `Origin` (an SDK, `curl`) is as before. A page of a public site also needs the browser's own leave to reach this machine: Chrome and Edge ask the user before a public page may call a loopback address (Local Network Access), whatever the server answers; a page served from this machine is not asked. |
| Request limits | Body over 64 MiB: `413` (chunked bodies included). JSON nested more than 64 levels deep: `400`. Request target over 8 KiB: `414`. A client that sends nothing for 30 s is disconnected. |
| Server stopped serving before the request ran | `503` for a request still queued when the model is unloaded or switched, or that waited in the queue past `--queue-timeout`: send it again. |

---

## 7. Coding Agent Integrations

The dashboard's **Integration** page (`http://127.0.0.1:8080/`) generates exact shell commands pre-filled with your host, port, API key, loaded model ID, and context size for **Claude Code**, **Codex CLI**, **DeepSeek Harness**, **OpenCode**, and **Pi**. On Windows, with the browser on the server's machine, **Open in PowerShell** starts Claude Code or Codex CLI in a project folder you pick (`POST /v1/agents/launch`). For any other agent, its **Others** entry gives a text to paste to the agent (base URLs, model ID, context, how the key is passed) by which it sets itself up.

The Claude Code guide defaults to `bypassPermissions`, so tool calls generally run without asking for approval. Choose `acceptEdits` to allow project edits and prompt for other commands as needed. The permission picker applies to both the copied command and **Open in PowerShell**. Choosing `auto` lets a model review commands. On a single-slot server, those reviews can displace the active conversation and cause another prefill.

Quick environment templates:

```sh
# Claude Code (Anthropic Messages API)
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_AUTH_TOKEN=local-key   # or server's --api-key
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096

# OpenAI-compatible SDKs / Codex / OpenCode
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=local-key
```

> For long Claude Code sessions on a single GPU slot, pass `--kv-disk ./flyx-cache:40` to `flyx serve`. `flyx` writes incremental prefix snapshots (honoring Anthropic `cache_control` breakpoints) so title/permission sub-requests do not force a full re-prefill.

---

## 8. Python and C APIs

### Python (`python/flyx/__init__.py`)

Zero external dependencies (`ctypes` over `libflyx.so` / `flyx.dll`, `FLYX_API_VERSION = 3`). Automatically locates the native library inside wheels or unpacked release archives; override via `lib=` or `FLYX_LIB`.

```python
import flyx

eng = flyx.Engine("~/models/qwen35-4b.flyx", lib="~/flyx-build/bin/libflyx.so")
s = eng.session()

city_schema = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
weather_tool = {"type": "function", "function": {"name": "get_weather", "parameters": city_schema}}

# Tool call (schema-constrained)
reply = s.chat_full(
    [{"role": "user", "content": "Look up the weather in Paris."}],
    tools=[weather_tool],
    tool_choice="required",
    parallel_tool_calls=False,
    enable_thinking=False,
    max_new_tokens=128,
)
print(reply["tool_calls"])

# Structured JSON output
reply = s.chat_full(
    [{"role": "user", "content": "Give the capital city of France as JSON."}],
    response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": city_schema}},
    enable_thinking=False,
    max_new_tokens=128,
)
print(reply["content"])
```

Concurrency & lifecycle:
- Multiple sessions on one `Engine` may generate concurrently on separate threads; one `Session` serves one generation at a time.
- `s.close()` is thread-safe and callable from a stream callback (cancels active generation and releases session on return). Closing `eng` keeps existing sessions alive until they close. Operations on closed handles raise `flyx.FlyxError`.

### C API (`include/flyx/flyx.h`)

Check `flyx_api_version() == FLYX_API_VERSION` (`3u`) before using struct layouts. Core entrypoints:
- Engine lifecycle: `flyx_load(path, &opts, &err)`, `flyx_unload(engine)`, `flyx_backend_name(engine)`, `flyx_tokenize(engine, text, out, cap)`.
- Session lifecycle: `flyx_session_new(engine)`, `flyx_generate(session, &req, on_event, user, &usage, &err)`, `flyx_cancel(session)` (thread-safe), `flyx_session_free(session)`.
- Events (`flyx_event_kind`): `FLYX_EVENT_TEXT` (1), `FLYX_EVENT_DONE` (2), `FLYX_EVENT_REASONING` (3), `FLYX_EVENT_TOOL_CALL` (4). Return non-zero from `on_event` to cancel.
- Wait for `flyx_generate` and concurrent calls on a session to return before calling `flyx_session_free`; open sessions hold a reference keeping the engine loaded past `flyx_unload`.
