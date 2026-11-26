# Testing

This document explains how to install the test dependencies and run the tests: test-data generation, kernel checks,
GPU tracing and end-to-end verification. Read [AGENTS.md](../AGENTS.md) before changing any test.

---

## 1. Testing Setup

```sh
pip install -r tests/requirements.txt    # test fixtures & HF reference (pinned, CPU torch; Python 3.12+)
ctest --preset quick                     # while editing: unit and tiny tests, the smoke tests (section 2)
ctest --preset dev --output-on-failure   # GPU + CPU suite, every test but DV's
ctest --preset cpu --output-on-failure   # CPU-only (no GPU required)
cmake --preset dev-full && ctest --preset full --output-on-failure   # DV's tests too (the same build tree)
```

End-to-end verification (http request to the qwen model server):

```sh
python tests/e2e/check_real_model.py --hf ~/models/Qwen3.5-4B --model ~/models/qwen35-4b.flyx --lib ~/flyx-build/bin/libflyx.so
python tests/apps/flyx-server/check_server.py --url http://127.0.0.1:8080
```

CI runs on hosted runners without a GPU: every pull request the `cpu` preset on Ubuntu (`tools/ci/cpu.sh`), and
DV's tests (`cpu-dv`) or the CUDA build, compiled and not run (`cuda-compile`), when it touches their paths; every
night `cpu-full`, the CPU tests on Windows and the CUDA build with DV's tests registered; and a published release's
assets on seven Linux distributions and Windows. The GPU suite (`tools/ci/gpu.sh`) runs on a maintainer's machine:
there is no GPU runner. See [tools/ci/README.md](../tools/ci/README.md).

---

## 2. Tiers, Labels and Presets

Every test has one **tier**, what it needs and so about what it costs, and any number of **aspects**. The test presets
select by these labels (`ctest -L` / `-LE` select by them too: `ctest --preset dev -L '^golden$'`). An `-LE` beside a
preset that excludes a label itself (`dev`, `gpu`, `cpu` and their Windows twins exclude `^dv$`) drops both exclusions
(CTest 4.4): give it with `--test-dir` instead of the preset.

| Label | Kind | Meaning | Given by |
|---|---|---|---|
| `unit` | tier | no fixture, no model file: seconds | inferred: no `FIXTURES_REQUIRED` |
| `tiny` | tier | the tiny models `make_assets` makes, not a whole golden run: seconds | inferred: a `FIXTURES_REQUIRED` |
| `golden` | tier | a whole `golden_test` or `gguf_e2e_test` run: seconds to minutes | `flyx_add_golden_test`; the `gguf_*` runs by hand |
| `tools` | tier | a bench or tools script's own test, not the engine's | `flyx_add_py_test` for scripts under `bench/`; others by hand |
| `gpu` | aspect | needs a CUDA device | every test registered in the CUDA block of `tests/CMakeLists.txt`; `vision` and `residual_decision`, registered before it but taking the CUDA backend too, by hand at its end |
| `gpu_serial` | aspect | measures the device's free memory, so runs alone (`RUN_SERIAL`) | `flyx_add_golden_test(... SERIAL)` |
| `smoke` | aspect | a golden or GPU test the quick preset runs anyway | by hand, at the end of `tests/CMakeLists.txt` |
| `dv` | aspect | DV's tests (DV is not in production), registered with `FLYX_TEST_DV=ON` alone | every test of `tests/dv.cmake` |
| `server` | aspect | starts a `flyx serve` | `flyx_add_py_test` for scripts under `tests/apps/flyx-server/` |
| `quick` | derived | `unit` or `tiny`, neither `gpu` nor `dv`; or `smoke` | `flyx_finish_test_labels()` |

`flyx_finish_test_labels()` runs at the end of `tests/CMakeLists.txt` and `bench/CMakeLists.txt`. It gives a test
without a tier one, adds `quick`, gives every `gpu` test a slot of the GPU (`RESOURCE_GROUPS gpus:1`) and
`gpu_preflight` before it (`FIXTURES_REQUIRED`, below), and refuses the
configure (`FATAL_ERROR`, naming the test) on a test with two tiers, an unknown label, `RUN_SERIAL` without
`gpu_serial`, `gpu_serial` without `gpu`, `smoke` on a DV test, or a DV test (named as one: `dv`, `_dv`, `dv31_`, ...;
or needing DV's assets) outside `tests/dv.cmake`. Labels are only ever appended
(`set_property(TEST ... APPEND PROPERTY LABELS ...)`), so a label given in one place is never lost in another.

A new test usually needs nothing: its tier and aspects follow from where and how it is registered. Label it by hand
when that is not enough: a test registered outside the CUDA block that takes the CUDA backend (`gpu`), a golden run
not made by `flyx_add_golden_test` (`golden`), a new smoke test (keep them a few seconds each: `quick` takes as long
as its longest tests, `cuda_ops`, about 35 s on an RTX 5070 Ti, and `golden_cuda_pdl`, 24 to 30 s). The inference cannot
see how long a test takes: a slow test inferred `tiny` lands in `quick`, so look at its time in the JUnit report
(`ctest --output-junit`): `python tools/check_test_tiers.py REPORT.xml` names every quick test over 20 s (the nightly
CI runs warn of them).

| Test preset (Windows: `-windows` suffix) | Runs | When |
|---|---|---|
| `quick` / `cpu-quick` | label `quick` | after each build while editing (a minute or two with the GPU's smoke tests) |
| `dev` (= `gpu`) / `cpu` | every test but DV's (`-LE ^dv$`) | before a pull request; CI's `cpu` |
| `full` / `cpu-full` | every test, DV's too | a change to DV's code (AGENTS.md gate 3), before a release; CI's nightly `cpu-full` (and `cpu-dv`: DV's alone) |

**DV's tests are opt-in.** DV is not in production, so its tests (`tests/dv.cmake`) and their fixtures (`make_tiny.py
--group dv`, test `make_dv_assets`) are registered only in a tree configured with `FLYX_TEST_DV=ON`: the configure
presets `dev-full`, `cpu-full`, `dev-windows-full` and `cpu-windows-full` turn it on in the same build tree as `dev`,
`cpu`, `dev-windows` and `cpu-windows` (a configure of seconds, nothing to compile: DV's code is built either way),
and configuring with the plain preset again turns it off. The test presets `full` and `cpu-full` name those configure
presets. In a tree without DV's tests, `dv_not_configured` stands in for them: skipped by plain `ctest`, not selected by
`dev`, `gpu` or `cpu`, and failed by `full` (which sets `FLYX_TEST_REQUIRE_DV`), so that a full run cannot pass short.
Six of DV's CUDA tests are known red in a tree on Linux (draft agreement under its floor, no token wrong; the reason is
in `tests/dv.cmake`) and disabled there: they show as Not Run (Disabled), and `-DFLYX_TEST_DV_KNOWN_RED=ON` runs them.

The fixtures come in two groups, each with its own digest and stamp in the assets directory (`.inputs-core`,
`.inputs-dv`): `make_assets` makes core's (`make_tiny.py --group core`), `make_dv_assets` DV's from them. DV's
draw their random weights from generators of their own, so core's files are the same bytes with or without DV's.

**The GPU's slots.** Every test preset runs 8 tests at once and hands the GPU tests out over
`tests/gpu_resources.json`: one GPU of 8 slots, a slot a test, the same as `-j 8` alone gave them. CPU tests take no
slot, so `-j` can go higher than the slots. Without a preset (and without `--resource-spec-file`) ctest ignores the
slots. Eight suit a 16 GB card: on an RTX 5070 Ti, sampled every second, the `gpu` preset's tests held at most 3.2 GB
of it at once and `full`'s 4.8 GB, beside the 0.3 GB the idle desktop held (the `cuda_device_memory*` tests, which run
alone, fill what is free on purpose whatever the slots). A card of 8 GB, or a desktop holding several GB of one, wants
fewer. Write a file of your own outside the source tree, `my_gpu.json` here:

```json
{ "version": { "major": 1, "minor": 0 }, "local": [ { "gpus": [ { "id": "0", "slots": 4 } ] } ] }
```

and pass it to each run in the preset's stead (`ctest --preset gpu --resource-spec-file /path/to/my_gpu.json`), or
once, in a test preset of your `CMakeUserPresets.json` that inherits `gpu` and sets `execution.resourceSpecFile`.
Four is an estimate (half the peaks above if the memory halves with the slots), not a measurement on an 8 GB card.

**`gpu_preflight`.** Every GPU test requires the fixture `gpu_preflight` (`tests/gpu_preflight.py`), so it runs first,
in a fraction of a second, whenever a GPU test is selected. It fails when the GPU already has more than 2560 MiB in use
before the GPU tests start: tests beside another program's memory fail for its reasons (an engine load refused, "Not
enough VRAM ... used by other apps"; a `cuda_device_memory*` test reading the other program's allocations as its own).
`FLYX_TEST_GPU_MAX_USED_MIB` in the environment changes the limit (`FLYX_TEST_GPU_MAX_USED_MIB=4096 ctest --preset
gpu`; in PowerShell `$env:FLYX_TEST_GPU_MAX_USED_MIB = 4096` first). The GPU checked is the first of
`CUDA_VISIBLE_DEVICES`, else `nvidia-smi`'s GPU 0. Without `nvidia-smi` on `PATH` (or a GPU it can read) it is skipped
and the GPU tests run. When it fails, the CPU tests still run, every GPU test is Not Run, and the output says who holds
the GPU (`nvidia-smi`'s processes; on Windows, where `nvidia-smi` gives no process's memory, Windows' GPU Process Memory
counter):

```text
 4/93 Test  #93: gpu_preflight ....................***Failed    1.34 sec
gpu_preflight: GPU 0 (NVIDIA GeForce RTX 5070 Ti): 13387 of 16303 MiB already in use before the GPU tests, more than 2560 MiB (FLYX_TEST_GPU_MAX_USED_MIB).
Beside it engine loads would be refused ("Not enough VRAM ... used by other apps") and the tests measuring free memory (cuda_device_memory*) would fail.
In use on GPU 0 (NVIDIA GeForce RTX 5070 Ti), MiB (Windows' GPU Process Memory counter, read after the check):
    12844  pid 52228   D:\flyx-wt\b-pr38\bin\flyx.exe
      154  pid 54032   C:\Windows\explorer.exe
      ...
  ... and 19 more, 15 MiB or less each
  (pid 2652 dwm.exe, the desktop's compositor: 8361 MiB counted, the surfaces it shares with other programs included)
Stop what holds it or wait for it, or allow more: FLYX_TEST_GPU_MAX_USED_MIB=<MiB> (docs/testing.md, section 2).

      Start  98: gguf_cuda_pdl
Failed test dependencies: gpu_preflight
 5/93 Test  #98: gguf_cuda_pdl ....................***Not Run   0.00 sec
...
The following tests FAILED:
         56 - vision (Not Run)                                  gpu quick smoke unit
         93 - gpu_preflight (Failed)                            gpu unit
         ...
```

---

## 3. Test Suite Map

| Test(s) | Scope |
|---|---|
| `layering`, `numerics`, `docs` | Static gates: directory include boundaries (`tools/check_layering.py`), `docs/design/numerics.md` launcher/test coverage (`tools/check_numerics.py`), public doc links and hygiene (`tools/check_docs.py`). |
| `gpu_preflight` | Before every GPU test (their fixture): fails, naming the processes, when the GPU already has more than `FLYX_TEST_GPU_MAX_USED_MIB` (2560) in use; skipped without `nvidia-smi` (section 2). |
| `make_assets`, `make_gguf_assets` | Fixture generators: tiny Qwen3.5 + DFlash2 (BF16/NVFP4), + DSpine (and with a window of 8) and a tiny DSpark as DeepSpec saves one, with float64 reference golden outputs, tiny Qwen3 with `transformers` golden outputs, and GGUF fixtures across all GGML block types. |
| `reference_vs_transformers` | Checks `python/reference` float64 reference against HuggingFace `transformers`. |
| `cuda_ops` | Every CUDA kernel (`tests/backends/cuda/op_test_*.cu`; DSpine's bridge and walk in `op_test_dspine.cu`) against `ops/<op>/ref.h` and `ops/sampling/sampling.h` (seeded; override via `FLYX_OPS_SEED`). |
| `golden_cpu_ref*`, `golden_cuda*` | End-to-end token-for-token checks: plain, MTP, DFlash2 (and DSpine in its place, `_dspine`, `_dspine_window`, a drafter of one layer `_dspine1`; on the tiny Qwen3, `_qwen3_dspine`), lookup drafts, forced long accepted runs, rollback, checkpoint restore, edited prompts, seeded sampling, stop strings, penalties, grammar enforcement, tree rounds, FP8 KV, tiered host KV, 2-bit StrataFlux & `auto` switch, fused vs unfused (`fusion=0`), PDL on/off, Qwen3 second architecture, sliding-window DFlash2, adaptive stopping on and off; DSpark's drafts against the float64 reference beside Qwen3.5 and Qwen3, spec == plain greedy and sampled (coupled drafts), its draft trees, its bf16 projections (`draft_weights=exact`). |
| `cuda_device_memory*` (`_dv*`: `FLYX_TEST_DV`) | Serial tests verifying engines and sessions release all allocated VRAM back to the device, and that sessions' CUDA round graphs (every round kind, drafted trees, MTP's window and batched rounds included) stay inside the room `seq_bytes` counts, measured by the device's free memory (`_tiers`: K/V and StrataFlux copies in host memory, across the `auto` switch), and that a session going between batched and lone rounds keeps its graphs (`_dv`, `_dv_lora`: the tiny Llama's DV files, every kind of DV round, its warm-ups batched too, beside a V-LoRA; `_dv_graphs`, `_dv_lora_graphs`: DV's round graphs within what they are counted at and under twice what they took, the canvas pass the body of a conditional node and inline (backend option `dv_conditional=0`), and the conditional nodes with their bodies on their own: the sessions as many as free memory holds for three engines, 32 to 64, or the test says why it checks nothing). |
| `gguf_reader`, `ggml_blocks`, `gguf_models`, `gguf_cpu_ref`, `gguf_cuda*` | GGUF container validation, exact per-block float64 dequant vs `gguf` Python package, layout equivalence vs `.flyx`, packed GPU execution across all block types + GGML NVFP4 + sidecar MTP/DFlash2. |
| `inkling2_cpu_ref`, `inkling2_cuda` | Tiny sparse MoE and Markov DSpark model execution, fused/generic rounds, mixed quantization and fixed-seed comparisons with plain decode. |
| `model_source` | Target/companion classification and discovery, including the Markov header and adjacent DSpark files. |
| `draft_ring`, `window_ring`, `gguf_cuda_unfused`, `gguf_cuda_pdl` | Window wrap, refused restores after overwrites and DFlash2 execution with packed GGUF weights; the rings checkpoints copy (sliding-window layers', the MTP layer's): their slots and rewind floor, and the MTP ring's checkpoints restored and saved after it went round with a fresh prefill's drafts. |
| `elastic_load`, `elastic_load_cuda`, `cuda_loader` | VRAM-constrained load policy, protected target arithmetic and CUDA loader estimates, storage and precision. |
| `dflash2_models` | The shared DFlash2 drafter (`models/drafters/`), and DeepSpec's DSpark (taken beside the tiny Qwen3.5, refused with DFlash2's convolutions or a block past 15): config validation before tensor binding, drafter GGUFs beside the tiny Qwen3 and Llama (block, taps, context nodes), a soft-capped selector and a tap after the output norm, drafter files refused for what they say or for the model they were made for, a DSpark file refused beside a model whose programs do not tap for it. |

| `tokenizer`, `bpe_merges`, `gguf_qwen2_*` | BPE merge exactness, pre-tokenizer splits, streaming UTF-8 decode. |
| `flyx_reader`, `flag_numbers`, `known_drafters`, `drafter_speeds`, `cli_usage`, `hub`, `hub_concurrent`, `drafter_dir` | Malformed `.flyx` refusal, strict CLI numeric parsing, the verified drafters list (made for the model, then inferred; the kinds the build runs; a list not version 1 throughout refused whole) and this machine's speeds with them (a malformed file replaced), the command line where it is asked something or cannot be followed (`--version`, `--help`, an unknown option), Hub SHA-256 & multi-process lock/resume, drafter kinds of cached repos, direct DFlash2 and DSpine directory loading. |
| `chat`, `chat_qwen35`, `chat_replies`, `reply_grammars`, `xgrammar_local_changes` | Template rendering vs `transformers`, native Qwen3.5 renderer vs Jinja on 3000 generated conversations, XML/JSON reply parsing, XGrammar structural tags, and the vendored XGrammar's differences from upstream (`third_party/xgrammar/README.flyx.md`) on vocabularies made by hand: mask and acceptance for every token at every step. |
| `request`, `reply_text`, `round_chooser`, `memory_budget`, `snapshot_store`, `strata_snapshot`, `parked_edit`, `telemetry`, `response_store`, `request_log` | Core request pipeline, stop-string boundary handling, speculation width/drafter chooser, VRAM/RAM tier planner, host/disk snapshot stores, telemetry snapshot. |
| `scheduler_*`, `session_host_budget`, `model_host`, `capi`, `python_binding`, `convert_*` | Multi-session scheduling, concurrent generation, prewarm/superseding semantics, refused-session guards, a model switch that loads only once the old model is closed, C API & Python `ctypes` binding, converter UTF-8/CLI checks. |
| `server_hardening` | What a hostile or broken client sends a live `flyx serve`, and what a browser brings: none of it ends the process, fills its memory, holds its workers or reads what it should not (`tests/apps/flyx-server/hardening.py` lists the checks). |
| `api_compat`, `streaming_tool_calls`, `queued_request_*`, `client_conversations`, `codex_threads`, `remembered_model`, `kv_method_flags`, `cli_utf8_paths` | Live `flyx serve` and CLI integration tests (OpenAI Chat/Responses, Anthropic Messages, cancellation, session isolation, remembered `serve.json`). |
| `gemm_tables` | `tools/codegen/gen_gemm_tables.py` validation, table inheritance, and invalid-table rejection (`cuda_ops` additionally verifies compiled table reachability via `test_gemm_tables_compiled`, rows 1..64, MoE `moe_gate_up` rule reachability via `test_moe_gate_up_rules`, and a drafting pass's K partition by K alone in every table via `test_gemm_draft_tables`). |
| `golden_cpu_ref_llama*`, `golden_cuda_llama*`, `tokenizer_minicpm5`, `llama_models` (`llama_models_dv`: with DV's file too, `FLYX_TEST_DV`), `converter` | The tiny Llama (MiniCPM5-1B's layers: no q/k norms, 16 query heads on 2 KV heads of 128) against `transformers` in float64 (plain decoding, unfused, `rows_mma`, NVFP4, prompt passes of up to 64 rows, batched rounds), MiniCPM5's pre-tokenizer, Llama descriptions and refusals, the converter's Llama path. |
| `golden_cuda_llama_dv*`, `golden_cpu_ref_llama_dv*`, `dv_models`, `dv_trees`, `dv_canvas`, `dv_convert*`, `reference_dv*`, `dv_counts` (`FLYX_TEST_DV`) | DV: plain decoding's tokens in every mode (15 and 31 drafts, beam and pool, 8/16/32 slots, causal, two rounds in flight, graphs on and off, rewind, restore, stop strings, sampled and with penalties, FP8 and host K/V, rounds mixed with lookup rounds, wide ones too (`wide=1`), and `wide=0`'s trees of 15, rounds of several sequences, a chat's checkpoint at a message boundary inside a prefill pass restored, adaptive stopping on and off, none planned for DV); DV's drafts against `flyx_ref/dv.py` in float64; the description and its refusals; the float64 tree and canvas pieces; the converter byte for byte; the Python reference against bifill-dv's functions; `check_dv.py`'s counts. |
| `golden_cuda_llama_lora*`, `golden_cuda_llama_dv31_lora*` and `dv_convert_lora` (`FLYX_TEST_DV`), `lora_convert`, `residual_decision` | The V-LoRA: the tiny Llama with a target LoRA against `transformers` with it merged (float64), on both backends, unfused and `rows_mma`, its NVFP4 linears, DV beside it; `.flyx` format 2 and the converter; each backend's bf16 prefill residual decision (`-v`), taken by the NVFP4 Llama with NVFP4 inputs and never with adapters (a device without FP4 tensor cores is held to its refusal of NVFP4 inputs instead). |
| `golden_cuda_quant_q4_k`, `golden_cuda_quant_q8_0`, `tree_bench_context` | Tree rounds of up to 64 rows on quantized Qwen3 GGUFs held to their own decoding; `tree_bench` timing every width from the same context. |
| `round_plans`, `plans_test`, `calibrate_test`, `round_costs_plausible`, `scheduler_cuda_dv` and `dv_choice_replay` (`FLYX_TEST_DV`) | DV's rounds chosen per round: the plans (widths, warm-up, refusals), costs at the context and their re-timing, a DV file's rounds in requests that run at once, and the chooser's replay on the traces in `bench/replay/dv_traces/`. |
| `bench_prompts`, `same_runs` | `flyx bench --prompts` (a prompt set, its JSON record) and `bench/ab/same_runs.py` (two runs' answers and rounds compared). |

---

## 4. CUDA Graph Tracing on Windows (CUPTI)

When configured with `FLYX_TESTS=ON` and CUPTI headers + `cupti.lib` present (optional `cufilt.lib` + `nv_decode.h` for symbol demangling), CMake builds `flyx-trace`. Place the matching CUPTI runtime DLL on `PATH`:

```powershell
cmake --build $BuildDir --target flyx-trace
$env:CUDA_INJECTION64_PATH = (Resolve-Path "$BuildDir\bin\libflyx-trace.dll").Path
$env:FLYX_TRACE = Join-Path $PWD 'rounds.tsv'
& "$BuildDir\bin\flyx.exe" run $Model -p hello --no-spec --ignore-eos -n 128
Remove-Item Env:\CUDA_INJECTION64_PATH, Env:\FLYX_TRACE
python bench\trace\gaps.py rounds.tsv --sequence
```

The TSV logs GPU kernel, copy, and memset intervals (`%p` expands PID for multi-process runs). Measure end-to-end throughput separately without CUPTI injection.

---

## 5. Coding Agent & Dashboard Checks

### Dashboard Tests

```sh
# Render Integration page across all 7 languages (open + keyed) without running a server
python tests/apps/flyx-server/check_dashboard_integration.py    # Playwright + PyYAML; --browser msedge for Edge

# Replay complete, interrupted, and error SSE streams in the Chat tab, and its browser-kept history
python tests/apps/flyx-server/check_dashboard_chat.py

# Model page: load settings and what a reload posts, the command line through this system's shells,
# the library (Hugging Face, this machine's folders), a server with nothing loaded
python tests/apps/flyx-server/check_dashboard_model.py
```

### Coding Agent Integration Tests

Verified agents: **Claude Code** (Anthropic Messages), **Codex CLI** (OpenAI Responses), **DeepSeek Harness**, **OpenCode**, and **Pi** (OpenAI Chat Completions), on both Linux/WSL2 and native Windows (loopback open and `--api-key`).

To verify an agent end-to-end:
1. Start `flyx serve MODEL` and copy the client configuration from `http://127.0.0.1:8080/` → **Integration**.
2. In an empty temporary workspace, write a random string to `probe.txt` (keep the string out of the prompt).
3. Ask the agent to read `probe.txt` and return its exact content; verify both directions of the tool protocol in the agent's reply and in `GET /v1/requests`.
4. Note for Claude Code: set `CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096` if the client defaults to an oversized `max_tokens` for custom local model IDs. For Codex CLI on native Windows, use `-c windows.sandbox=mxc --sandbox read-only` if default shell probing is restricted.
