# Monitoring, Telemetry, and Dashboard

`flyx serve` exposes real-time metrics over `GET /v1/telemetry`, renders a live terminal dashboard (unless `--no-monitor` or non-TTY), and serves a built-in web UI at `GET /` (`/dashboard`, auto-opened unless `--no-browser` or headless/SSH).

---

## 1. `GET /v1/telemetry` Schema (Version 1)

Read-only JSON snapshot (`Cache-Control: no-store`). Query parameter: `?recent=N` (`0..100`, default `20`). Requires API key when `--api-key` is set. Unknown/unavailable fields are `null`.

```json
{
  "schema_version": 1,
  "sampled_at": 1763500000.12,
  "server": {
    "name": "flyx-server",
    "version": "0.2.0",
    "pid": 12345,
    "started_at": 1763499900.0,
    "uptime_s": 100.12,
    "api_key_required": false,
    "agent_launch": ["claude", "codex"],
    "cors": { "loopback": true, "any": false, "origins": [] }
  },
  "model": {
    "id": "Qwen3.8-27B-UD-IQ3_XXS",
    "loaded": true,
    "context": 262144,
    "device": "NVIDIA GeForce RTX 5070 Ti",
    "backend": "cuda",
    "drafter": "mtp",
    "draft_weights": "nvfp4",
    "max_width": 7,
    "chat_format": "qwen35",
    "max_sessions": 2,
    "vision": true,
    "vision_device": "disk",
    "vision_quant": "q8",
    "vision_max_tokens": 0,
    "memory": "weights: model 10.83 GB, workspace: kernels 0.33 GB | 11.17 GB of 13.78 GB budget | one session (262144 tokens) 2.61 GB on the GPU (K/V of 4096 tokens) and 10.41 GB in host memory",
    "load_s": 3.4,
    "degraded": []
  },
  "state": {
    "phase": "decode",
    "queued": 0,
    "running": 1,
    "label": "chat",
    "id": "chatcmpl-...",
    "session": 1,
    "prompt_tokens": 1024,
    "reused_tokens": 900,
    "prefilled": 1024,
    "generated": 128,
    "tokens_per_round": 3.2,
    "rounds": 39,
    "drafted": 117,
    "accepted": 85,
    "decode_tok_s": 145.0,
    "prefill_tok_s": 3850.0,
    "max_new_tokens": 512,
    "elapsed_s": 0.92
  },
  "rates": { "decode_tok_s": 145.0, "tokens_per_round": 3.2, "acceptance": 0.73,
             "output_per_second": [0, 0, 96, 141, 138], "prefill_per_second": [0, 124, 0, 0, 0],
             "rounds_per_second": [0, 0, 30, 44, 43], "drafted_per_second": [0, 0, 90, 132, 129],
             "accepted_per_second": [0, 0, 66, 97, 95] },
  "memory": {
    "ledger_used": 14800000000,
    "ledger_budget": 15900000000,
    "runtime": 540585984,
    "host_mapped": 9987850240,
    "sessions": [{ "id": 1, "tokens": 1152 }]
  },
  "device": {
    "name": "NVIDIA GeForce RTX 5070 Ti",
    "util": 98, "mem_util": 85, "mem_used": 14900000000, "mem_total": 17095000000,
    "temp_c": 62, "power_w": 240.0, "power_limit_w": 300.0,
    "sm_clock_mhz": 2850, "mem_clock_mhz": 14000,
    "pcie_gen": 5, "pcie_width": 16, "fan": 35, "clock_limits": "power cap", "process_host": 11210000000
  },
  "process": { "cpu_s": 12.4, "rss": 1570000000, "ram": 12780000000 },
  "system": { "ram_total": 34000000000, "ram_available": 18000000000, "cpus": 16, "cpu_s": 51234.5 },
  "totals": {
    "requests": 10, "errors": 0, "cancelled": 1,
    "prompt_tokens": 5000, "reused_tokens": 4100, "output_tokens": 1200,
    "tool_calls": 4, "grammar_fixes": 0
  },
  "recent": [
    {
      "finished_at": 1763499990.5, "label": "messages", "status": "stop", "id": "...",
      "session": 1, "prompt_tokens": 500, "reused_tokens": 420, "output_tokens": 80,
      "queue_ms": 0.2, "ttft_ms": 42.1, "prefill_ms": 38.0, "decode_ms": 510.0,
      "decode_tok_s": 156.9, "tokens_per_round": 3.4, "drafted": 69, "accepted_drafts": 56,
      "tool_calls": 1, "grammar_fixes": 0
    }
  ],
  "load": { "state": "ready", "model": "Qwen3.8-27B-UD-IQ3_XXS", "error": null, "fallback": null, "degraded": [], "vision_projector": { "from": "mmproj-BF16.gguf", "size": 912345678 } }
}
```

Field semantics (defined in `apps/flyx-server/telemetry.cpp`):
- `model.draft_weights`, `model.degraded`, `load.degraded`: Active drafter linear weight precision (`"nvfp4"`, `"q8"`, `"exact"`, or `null` when no neural drafter is active or on `cpu_ref`), and the list of elastic VRAM adjustments (`[{"kind", "from", "to", "detail"}]`) applied during load (empty when the requested settings fit).
- `model.vision`, `model.vision_device`, `model.vision_quant`, `model.vision_max_tokens`: Whether a vision encoder is loaded (`true`/`false`), where its weights live (`"disk"`, `"host"`, `"gpu"`, `"auto"`), its linear quantization mode (`"q8"`, `"exact"`, `"nvfp4"`), and any per-image token cap (`0` = auto). `load.vision_projector` reports `{"from", "size"}` for a companion `mmproj` file, or `null` when absent or vision is off.
- `model.context_limit`: Present only when memory cut the context under the one wanted (`--ctx max`: the model's own; no `--ctx`: the default). `by` is `gpu` or `host`; `wanted` the context it would have had; `short` whether what it got is under 32,768 tokens; `left` the bytes that side had (GPU: what the sessions share after the weights and the workspace; host: what K/V may take); `gpu_tokens` the K/V positions placed on the GPU (`0..context`, the rest in host RAM). For `gpu`: `others` is what other programs and the driver hold of the GPU, `runtime` flyx's own CUDA runtime beside the model where the system counts a process's share (Windows), else `null` with that runtime inside `others`; `closable` whether other programs hold enough for closing them to matter. Example: `{"by": "gpu", "wanted": 262144, "short": true, "left": 823806362, "gpu_tokens": 8192, "others": 3600134144, "closable": true, "runtime": 540585984}`.
- `server.cors`: Which pages a browser lets call the inference API from another origin (`--cors`): `loopback` this machine's own pages, `any` every page, `origins` the ones named.
- `model.memory`: The engine's memory report, one line of text. `rates.output_per_second`: tokens generated in each of the last 60 whole seconds, oldest first (60 entries; the example shows five). `rates.prefill_per_second`: prompt tokens computed (not reused) in the same seconds, each prefill pass distributed proportionally across the seconds it ran; a prefill shorter than a second shows as the tokens it computed, not as its rate (that is `state.prefill_tok_s`, and `recent[]`'s `prompt_tokens - reused_tokens` over `prefill_ms`).
- `memory.runtime`: flyx's own GPU memory beside the ledger (the CUDA context and kernel modules), as the system counted it at the load where it counts a process's share (Windows); `null` elsewhere. `ledger_used + runtime` is what flyx holds of `device.mem_used`.
- `process.ram`: the host memory flyx holds in all. `process.rss` (Linux's resident set, Windows' working set) does not count `memory.host_mapped`, the sessions' K/V past the GPU's part, which the GPU reads over the bus from page-locked memory mapped for it alone (10 GB for the 27B at 253K tokens): `ram` is `rss + host_mapped`, and where the system counts every page-locked byte the process holds for the GPU (`device.process_host`, Windows' "Shared GPU memory" of the process) that figure plus the working set past its other page-locked part, because Windows' working set holds page-locked pages only until it is emptied or trimmed (they stay locked): short of the real figure by at most the pageable rest. Measured on an RTX 5070 Ti: 1 GiB of `cudaHostAlloc` counts in the resident set (WSL2), the working set and `process_host`; 1 GiB of K/V host memory (`cuMemCreate`) only in `process_host`. `device.clock_limits`: what holds the GPU's clocks down, as text (`""` when nothing does). `load`: as `GET /v1/load` answers (`progress` only while `state` is `fetching`).
- `state.phase`: `unloaded` | `loading` | `idle` | `prefill` | `decode`. When multiple requests run concurrently, `state` reflects the oldest running request.
- `state.decode_tok_s`: Per-request rate since the first emitted token group (`null` until the second delivery). `rates.decode_tok_s` is aggregate decode throughput across all running requests over the recent window.
- Speculative decoding: `state.rounds`, `state.drafted`, `state.accepted` are the running request's rounds verified so far, the drafts they verified (the rounds' widths, lookup rounds among them) and the drafts accepted; `recent[].drafted` and `recent[].accepted_drafts` a finished request's. `rates.tokens_per_round` (`1 + accepted / rounds`) and `rates.acceptance` (`accepted / drafted`, in `0..1`, `null` without drafts) aggregate across all running requests over the same recent window as `rates.decode_tok_s`. `rates.rounds_per_second`, `drafted_per_second` and `accepted_per_second` count the same for every running request in each of the last 60 seconds: a second's tokens a round is `1 + accepted / rounds`, its acceptance `accepted / drafted`.
- `state.prefill_tok_s`: The request's prompt tokens computed (not reused) a second, from its start to its first token (while it prefills: to now, counting the passes sent to the device); `null` when it reused the whole prompt. Measured as `recent[].prefill_ms` is.
- `recent[].ttft_ms`: Milliseconds from request arrival to first non-empty reasoning, text, or tool-call event (`null` for prewarms or requests rejected before output).
- `recent[].cancel_reason`: Present (`"superseded"`) when replaced by a newer request for the same conversation.

---

## 2. Terminal Monitor (`flyx monitor`)

Attach a read-only terminal UI to any running server (default `http://127.0.0.1:8080`):

```sh
flyx monitor http://127.0.0.1:18080 --key YOUR_KEY --interval 1
flyx monitor http://127.0.0.1:8080 --once --for 4   # single plain-text frame after 4 s of history
```

| Mode | Hotkeys |
|---|---|
| `flyx serve` embedded panel | `l` full log · `o` open browser dashboard · `space` freeze · `q` twice stop server |
| `flyx monitor` standalone | `q` quit monitor (server keeps running) · `space` freeze · `r` refresh |

---

## 3. Built-in Web Dashboard

Located in `apps/flyx-server/dashboard/`, written as zero-build ES modules using vendored Preact + `htm` (`third_party/preact/`, `third_party/htm/`) and embedded directly into the `flyx` executable.

| Path | Role |
|---|---|
| `index.html`, `style.css`, `app.js` | Shell, light/dark themes, keyboard navigation (`1`–`6`, `[`), telemetry poller |
| `pages/` | `overview.js`, `model.js`, `requests.js`, `chat.js`, `logs.js`, `integration.js` |
| `components/` | `topbar.js`, `sidebar.js`, `conversation.js`, `agent_setup.js` |
| `lib/` | `html.js`, `api.js`, `telemetry.js`, `i18n.js`, `format.js`, `charts.js`, `markdown.js`, `ui.js`, `actions.js`, `chats.js` |
| `i18n/` | Localization dictionaries: `zh`, `es`, `fr`, `de`, `ja`, `ko` |

When `--api-key` is active, auto-opening the browser passes the key in the URL fragment (`#key=...`) so it never appears in HTTP request lines.

What the pages do:

| Page | Shows |
|---|---|
| Overview | Live phase; throughput (decode tokens out on the left axis, prompt tokens computed on the right, with the decode and prefill speeds of now, the running request or the last one); speculative decoding (tokens a round on the left axis, the share of drafts accepted on the right, with both of now or the last request); recent requests with each one's cache hit (prompt tokens reused), "View all" with the count since start; engine facts; device: load and power as bars, temperature, clocks, PCIe; host: VRAM and RAM as bars of flyx's part and the rest in use, CPU. With no model loaded, a button to the model library. A context memory cut short (`model.context_limit.short`) is flagged in the engine facts with its cause, and elastic VRAM adjustments (`model.degraded`) surface a global warning banner with Manage VRAM and View details actions. |
| Model | Two views behind tabs. **Current model**: state, facts (including active drafter precision), Reload/Unload actions and load progress of what is served, and its load settings across sub-tabs (General: context, K/V, API model id; Speculative decoding, with status dot reflecting whether a drafter is actually loaded, active drafter details inside the tab, and `draft_weights` auto/nvfp4/q8/exact under Advanced; Vision encoder: projector file, active projector details, storage, precision, max tokens/image; plus the equivalent command line; a change is applied by reloading). When tight VRAM triggered elastic load adjustments (`degraded`), a warning banner lists each adjustment, shows what is active now, and offers inline Manage VRAM and one-click Restore full settings. Where memory cut the context under the model's own, a line says which memory did (the GPU's part left for sessions and what other programs hold of it, or the host memory K/V may take) and what would give more; as a warning when the context is short. With nothing loaded: only *load the last model again* and *explore the model library*. **Model library**: *Hugging Face* (multi-architecture browsing with filter pills and pagination via `GET /v1/hub/search`, configurable Hub mirror via `GET`/`POST /v1/hub/config`, a repository's quantizations, companion drafters and `mmproj` projectors with up to one drafter + one projector selectable together from `GET /v1/hub/files`; standalone model repositories of an unsupported architecture are not loadable) and *On this machine* (folders with distinct icons for models, drafters, and `mmproj` projectors in each: the Hugging Face cache, `--models-dir`, and folders added through the system's folder dialog, `POST /v1/hub/dir`). Switching to a different model does not silently carry the previous model's custom companions or API model id; companion files can be applied directly or paired with a model load in one step. |
| Requests, Logs | The request log with a conversation reader; the server's log lines. |
| Chat | A streamed conversation with the served model (with image upload, paste, and drag-and-drop when a vision encoder is loaded). Conversations are kept in the browser (`localStorage`, at most 100 and about 3 MB; the history button lists, reopens and deletes them); the server stores none. |
| Integration | Setup for Claude Code, Codex CLI, DeepSeek Harness, OpenCode and Pi, a text to hand any other agent, SDK snippets, and the API reference. For the agents in `server.agent_launch` (Claude Code and Codex CLI on Windows), with the browser on the server's machine: **Open in PowerShell** asks for the project folder in a dialog (the one used last in this browser, or another chosen in the system's folder dialog) and starts the agent there in a new window (`POST /v1/agents/launch`). |

The sidebar keeps decode speed, tokens per round, TTFT, VRAM, RAM, context, queue, GPU and CPU load in view on every page. Folders added to the library and the chat history are the browser's own (per browser and address).

### Live Dashboard Development (No Rebuild)

```sh
flyx serve MODEL --dashboard-dir apps/flyx-server/dashboard
```

Validation tools:
- Missing translations check: `python tools/dashboard_sweep.py --lang ja --dir apps/flyx-server/dashboard`
- Offline Integration guide check (all 7 languages, keyed/open): `python tests/apps/flyx-server/check_dashboard_integration.py` (uses Playwright + PyYAML; pass `--browser msedge` for Edge)
- Offline Model page check (load settings, shell quoting of the command line, the library): `python tests/apps/flyx-server/check_dashboard_model.py`
- Offline Chat check (reply streams, the browser-kept history): `python tests/apps/flyx-server/check_dashboard_chat.py`
- Live endpoint & asset serving check: `python tests/apps/flyx-server/check_server.py --url http://127.0.0.1:8080`
