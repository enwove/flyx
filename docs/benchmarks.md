# Benchmarks

Reproducible long-context coding, `flyx bench` decode, drafter, and Needle-in-a-Haystack measurements for `flyx` on consumer NVIDIA GPUs with exact hardware, models, and commands.

---

## v0.2.1 four-GPU verified benchmarks

Linux captures used CUDA 13.0.88 on Ubuntu 24.04 with PTX JIT disabled. The hardware table below describes the original `5fa1a32` runs. The RTX 4090 and 3090 Inkling decode results come from a second capture on `078022fc`. Windows measurements used native Windows 11 (WDDM), with Inkling and the dashboard on `5fa1a32`, and Qwen, the 20K ring and cold start on `42e3734` before the Inkling decode changes. The Windows candidate passed 170/170 GPU-suite tests. Its standalone executable contained 115 cubins and zero PTX images and weighed 57,144,832 bytes (57.14 MB), compared with 58,960,384 bytes in v0.2.0. Those four-GPU captures predate the version bump; the RTX 5070 Ti 256K coding rerun below used the tagged `v0.2.1` release binary (`flyx-0.2.1-windows-x86_64-cuda13.exe`, commit `7f0de79`).

| GPU | Driver | Power limit | Qwen3.8-27B GGUF | GPU weight GB | Effective / rated GB/s | Share |
|---|---|---:|---|---:|---:|---:|
| RTX 5090 32 GB | 615.71.09 | 575 W | NVFP4-MTP-N4_0 | 14.77 | 1,385 / 1,792 | 77.3% |
| RTX 4090 24 GB | 595.91.07 | 450 W | UD-Q4_K_XL | 16.81 | 830 / 1,008 | 82.4% |
| RTX 3090 24 GB | 595.71.05 | 350 W | UD-Q4_K_XL | 16.81 | 719 / 936 | 76.9% |
| RTX 5070 Ti 16 GB | 610.88 | 300 W | UD-IQ3_XXS | 10.28 | 707 / 896 | 78.9% |

Effective bandwidth estimates bytes read per token from GPU weight size times plain decode rate. The range was 76.9%–82.4% of rated bandwidth. Verbose Linux runs recorded graph execution for every round, with zero eager fallbacks, including on the 3090 using the compatible sm_80 cubin.

### Qwen3.8-27B

Mixed decode uses the three built-in prompts, 128 tokens each. Linux ran their JSONL equivalents to capture IDs (one warm-up, then the median of three runs). Windows ran the built-in benchmark once per mode. Its mixed record omits token IDs. Linux mixed IDs and all four cards' separate single-code-prompt OFF/ON IDs matched. Each Qwen file includes MTP. DFlash2 used the same Q4_K_M companion on all cards. Rates are tokens/s.

| GPU | Mixed plain | Mixed MTP | Mixed DFlash2 | Mixed all | First code prompt plain / DFlash2 | 5,791-token prefill |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 | 93.8 | 286.6 | 315.5 | 323.9 | 93.8 / 565.5 | 15,496 |
| RTX 4090 | 49.4 | 158.0 | 171.6 | 165.6 | 49.5 / 305.7 | 2,678 |
| RTX 3090 | 42.8 | 121.7 | 131.2 | 125.0 | 42.8 / 234.1 | 954 |
| RTX 5070 Ti | 68.8 | 167.2 | 154.4 | 156.1 | 68.8 / 332.8 | — |

On the 6,293-token Inkling prefill workload (same input file, different tokenizer), the 3090, 4090 and 5090 reached 1,829, 3,855 and 5,409 tok/s respectively. These plain prefill timings come from the original candidate.

The 5090 used W4A4 prefill on the NVFP4 target. On the same prompt, explicit `--activations exact` (BF16 inputs) gave 3,602 tok/s on the first pass and 3,598 on the second. At 128,000 input tokens, plain decode reached 73.4 tok/s and MTP reached 203.8 tok/s with identical output text over 256 tokens.

### RTX 5070 Ti 256K coding rerun (v0.2.1 native Windows 11)

Measured with the official GitHub Release binary `flyx-0.2.1-windows-x86_64-cuda13.exe` (`57,144,832 B` / `57.14 MB`, tag `v0.2.1`, commit `7f0de79`) on an NVIDIA GeForce RTX 5070 Ti (`16,303 MiB` VRAM, `300 W` TDP, driver `610.88`), **native Windows 11 Pro** (`10.0.26300`), and **32 GB system RAM**, running `unsloth/Qwen3.8-27B-GGUF` (`Qwen3.8-27B-UD-IQ3_XXS.gguf`, `10.93 GB`; `10.83 GB` GPU weights + `0.42 GB` host embedding + `0.35 GB` MTP on GPU).

- **Automatic 262K (`--ctx max`) K/V placement (`14.89 GB` WDDM budget):** `16,384` tokens K/V on GPU (`3.71 GB` session VRAM) + `245,760` tokens in host RAM (`9.70 GB` host memory incl. checkpoints); exact K/V reads up to `123,203` tokens (`strata_from=123203`), then 2-bit StrataFlux + exact hot reads *(vs `v0.2.0` WSL2 baseline: `2,048` GPU KV + `260,096` RAM KV, exact up to `34,816`)*.
- **Decode (`260,051` prompt tokens, 512 new tokens, greedy MTP):** **`113.5 tok/s`** (`512` tokens in `4,501.9 ms`, `106` rounds, `4.82 tok/round`, `6.3 drafts/round`, `lookup 3`) — **1.85× faster (+85%)** than the `v0.2.0` WSL2 baseline (`61.2 tok/s`).
- **Prefill:** **`787.2 tok/s`** (`330.37 s` / `330,368.9 ms` for `260,051` tokens; vs `v0.2.0` WSL2 `382.49 s` / `679.9 tok/s`).
- **Needle recall (50% depth at 256K):** **PASS** (verbatim recall: `# Calibration verification passphrase: emerald-falcon-4829`).

| Metric | v0.2.0 WSL2 (`b950329b`) | v0.2.1 Native Win11 (`7f0de79`) | Delta |
|---|---:|---:|---:|
| WDDM / GPU KV budget | `13.51 GB` (`2,048` GPU + `260,096` RAM) | `14.89 GB` (`16,384` GPU + `245,760` RAM) | `8×` GPU KV tokens |
| Exact K/V threshold | `34,816` tokens | `123,203` tokens (`strata_from=123203`) | `3.54×` exact span |
| 256K prefill (`~260K` tokens) | `382.49 s` (`679.9 tok/s`) | **`330.37 s` (`787.2 tok/s`)** | **`+15.8%`** |
| 256K decode (512 new, MTP) | `61.2 tok/s` (`3.70 tok/rnd`) | **`113.5 tok/s` (`4.82 tok/rnd`, `6.3 drafts/rnd`)** | **`1.85× (+85%)`** |
| 256K needle recall (50% depth) | pass | **pass** | — |

Only the 256K coding prompt (with its 50%-depth needle) was rerun on `v0.2.1` native Windows 11; the `1K`/`128K` coding rows and `5/5 + 4/4` NIAH grid in Sections 1 and 3 remain from the `v0.2.0` WSL2 baseline.

```sh
python bench/prompts/make_coding_prompt.py --flyx ./flyx.exe --model Qwen3.8-27B-UD-IQ3_XXS.gguf --tokens 260000 -o prompt_256k.txt
./flyx.exe run Qwen3.8-27B-UD-IQ3_XXS.gguf --drafter mtp --ctx max --chat --no-think -n 512 --ignore-eos -v -f prompt_256k.txt
```

### Inkling-2-Mini and Markov DSpark

The target was `Inkling-2-Mini-UD-IQ3_XXS.gguf`, the companion `dspark-Inkling-2-Mini-Q8_0.gguf`. Each cell shows plain → DSpark decode tok/s on the same GPU, with exact KV reads and lookup disabled. All four GPUs matched prompt IDs, output IDs and output lengths between plain and DSpark for every prompt in the warm-up and three measured runs. The RTX 5090 and 5070 Ti used `5fa1a32`. The 4090 and 3090 used `078022fc`, which rounds the scaled ShortConv result before adding the residual. On each of those two GPUs, the fused and generic paths matched on all eight chat prompts, and all four GPU tests passed (`cuda_ops`, `inkling2_cuda`, `cuda_loader`, `elastic_load_cuda`).

| GPU | chat_short, 8 × 128 | code_short, 8 × 384 |
|---|---:|---:|
| RTX 5090 | 225.5 → 259.8 | 228.5 → 438.9 |
| RTX 4090 | 166.7 → 202.9 | 167.9 → 322.4 |
| RTX 3090 | 126.2 → 136.5 | 126.9 → 222.0 |
| RTX 5070 Ti | 150.9 → 169.3 | 153.6 → 276.2 |

The RTX 4090 recheck used a 500 W limit and driver 595.91.07. The RTX 3090 recheck used 350 W and driver 580.178.04. The ranges below cover the three measured runs. Speedups use the unrounded rates. The single-prompt rows used the first `chat_short` prompt (`--limit 1 -n 128`) once per mode.

| GPU | Workload | Plain tok/s (min–max) | DSpark tok/s (min–max) | DSpark tokens/round | Speedup |
|---|---|---:|---:|---:|---:|
| RTX 4090 | Chat, 8 × 128 | 166.6–166.7 | 202.7–202.9 | 1.79 | 1.217× |
| RTX 4090 | Code, 8 × 384 | 167.9–168.0 | 321.6–322.6 | 3.38–3.39 | 1.920× |
| RTX 4090 | First chat prompt, 1 × 128 | 166.5 | 257.2 | 2.76 | 1.545× |
| RTX 3090 | Chat, 8 × 128 | 126.1–126.4 | 135.8–139.0 | 1.60–1.67 | 1.082× |
| RTX 3090 | Code, 8 × 384 | 126.8–127.0 | 220.9–222.7 | 3.48–3.49 | 1.749× |
| RTX 3090 | First chat prompt, 1 × 128 | 126.1 | 180.1 | 2.82 | 1.429× |

### DFlash2 ring memory and Windows latency

With a 17,993-token prompt and 20,480-token context, single-session memory (decimal GB, flat → ring) fell from 1.87 → 1.51 on the 3090 and 4090, 1.82 → 1.46 on the 5070 Ti (0.36 GB saved), and 1.82 → 1.52 on the 5090. Flat and ring output IDs matched. With 59,976 prompt tokens and a 65,536-token context on the 5090, the same comparison gave 4.24 → 3.02 GB (1.22 GB saved), with identical text and accepted tokens each round.

On the 5070 Ti, three fresh Inkling plain processes with warm OS file cache took 2,381, 2,353 and 2,347 ms from spawn to first visible text (median **2,353 ms**). In a separate live dashboard request the active drafter read **dspark · q8**. The server decoded **237.8 tok/s** (128 tokens, 3.34 tokens/round). Client time to first streamed content was 106 ms.

### Reproduce the captures

Set `Q` to the Qwen file in the hardware table (akopytko's NVFP4 GGUF or Unsloth's UD GGUF), `D` to z-lab's `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, and `I` and `S` to the two Inkling files above from Unsloth. From the repository root, run the following in Bash. For the Windows mixed timing, omit `--prompts mixed3.jsonl` to use the built-in workload. For repeated measurements, warm up each variant once and alternate their order over three repetitions.

```bash
cat > mixed3.jsonl <<'JSONL'
{"question": "Write a Python function that parses an ISO 8601 date string without using datetime, with tests."}
{"question": "Explain how a CPU cache works to a new engineer, with a concrete example of a cache miss."}
{"question": "Copy this list exactly, one item per line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey."}
JSONL
export CUDA_DISABLE_PTX_JIT=1
common=(--backend cuda --no-vision --ctx 4096 --temperature 0 --seed 42 --ignore-eos)
for mode in plain mtp dflash2 all; do
  case "$mode" in
    plain) opts=(--no-spec --no-lookup) ;;
    mtp) opts=(--drafter mtp) ;;
    *) opts=(--drafter "$mode" --drafter-file "$D") ;;
  esac
  flyx bench "$Q" "${common[@]}" "${opts[@]}" --prompts mixed3.jsonl -n 128 --json "mixed-$mode.json"
done
for mode in plain dflash2; do
  opts=(--no-spec)
  if [[ $mode == dflash2 ]]; then opts=(--drafter dflash2 --drafter-file "$D"); fi
  flyx bench "$Q" "${common[@]}" --prompts bench/prompts/code_short.jsonl --limit 1 --no-lookup "${opts[@]}" -n 128
done
for set in chat_short code_short; do
  n=128
  if [[ $set == code_short ]]; then n=384; fi
  for mode in plain dspark; do
    opts=(--no-spec)
    if [[ $mode == dspark ]]; then opts=(--drafter dspark --drafter-file "$S"); fi
    flyx bench "$I" "${common[@]}" --kv-method exact --no-lookup --prompts "bench/prompts/$set.jsonl" --limit 8 -n "$n" "${opts[@]}" --json "$set-$mode.json"
  done
done
runner=(--backend cuda --no-vision --chat --no-think --seed 42 --temperature 0 --ignore-eos)
for model in "$Q" "$I"; do
  flyx run "$model" "${runner[@]}" --no-spec --no-lookup --ctx 8192 -f bench/prompts/prompt_6k.txt -n 16
done
flyx run "$Q" "${runner[@]}" --no-spec --ctx 8192 -f bench/prompts/prompt_6k.txt -n 32 --activations exact -v
python bench/prompts/make_coding_prompt.py --flyx flyx --model "$Q" --tokens 128000 -o long128k.txt
flyx run "$Q" "${runner[@]}" --ctx 131072 -f long128k.txt --no-spec --no-lookup -n 256 -v
flyx run "$Q" "${runner[@]}" --ctx 131072 -f long128k.txt --drafter mtp -n 256 -v
python bench/prompts/make_coding_prompt.py --flyx flyx --model "$Q" --tokens 60000 -o long60k.txt
for ring in 0 1; do
  flyx run "$Q" "${runner[@]}" --ctx 65536 -f long60k.txt --drafter dflash2 --drafter-file "$D" --no-lookup --backend-opt dflash_ring="$ring" -n 256 -v
done
```

For the 20K ring comparison, build the single JSONL prompt from the three repository fixtures in their original order. The prefix and blank lines are part of the input.

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path("bench/prompts")
files = ["long.txt", "prompt_6k.txt", "summarize_tree_verification.txt"]
body = "\n\n".join((root / name).read_text(encoding="utf-8") for name in files)
question = "Summarize the following text and code in detail:\n\n" + body
Path("ring.jsonl").write_text(json.dumps({"question": question}) + "\n", encoding="utf-8")
PY
for ring in 0 1; do
  flyx bench "$Q" --backend cuda --no-vision --ctx 20480 --kv-method exact --no-lookup --temperature 0 --seed 42 --ignore-eos --prompts ring.jsonl -n 128 --drafter dflash2 --drafter-file "$D" --backend-opt dflash_ring="$ring" --json "ring-$ring.json" -v
done
```

To repeat the Windows startup measurement, time the process through its first visible reply with the OS cache warm. For the live measurement, start the server below and stream a chat completion for `Write a detailed Python ISO 8601 parser with unit tests. Start with code.` with thinking disabled, temperature 0, seed 42 and a 128-token limit.

```bash
flyx run "$I" --backend cuda --ctx 4096 --no-spec --no-lookup --chat --no-think --temperature 0 --seed 42 --ignore-eos -p "Reply with the single letter A." -n 4 -v
flyx serve "$I" --drafter dspark --drafter-file "$S" --no-lookup --ctx 4096 --host 127.0.0.1 --port 18080 --no-browser --no-monitor
```

## Historical v0.2.0 results

Sections 1–3 below preserve the original six-card measurements on `b950329b`, including the WSL2 run, alongside the updated `v0.2.1` native Windows 11 RTX 5070 Ti 256K rerun (`7f0de79`, **`113.5 tok/s`**) in Sections 1 and 3. The independent DV study in section 5 uses a different model and workload.

## 1. Qwen3.8-27B Long-Context Coding Benchmark

Except for the `v0.2.1` native Windows 11 RTX 5070 Ti 256K rerun (`flyx-0.2.1-windows-x86_64-cuda13.exe`, tag `v0.2.1`, commit `7f0de79`), all rows below were measured on the same `release-linux` build (`flyx 0.2.0`, commit `b950329b`, CUDA `13.0`, per-GPU tuning tables) at full native context (`--ctx max` = `262,144` tokens) with built-in **MTP** speculative decoding (plus `MTP + DFlash2` on RTX 5090 `NVFP4`):

- **RTX 5090 (Blackwell native FP4):** [`akopytko/Qwen3.8-27B-NVFP4-GGUF`](https://huggingface.co/akopytko/Qwen3.8-27B-NVFP4-GGUF) `Qwen3.8-27B-NVFP4-MTP-N4_0.gguf` (`NVFP4`, `15.72 GB` file, `15.20 GB` GPU weights; 504 GGML NVFP4 tensors executing on native `sm_120a` W4A4 FP4 tensor cores in prefill + built-in MTP).
- **24 GB GPUs (RTX 4090, RTX 3090):** [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-Q4_K_XL` (`17.66 GB` file, `17.24 GB` GPU weights).
- **16 GB GPUs (RTX 5080, RTX 5070 Ti, RTX 4080):** `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` (`10.93 GB` file; `10.72 GB` GPU weights on `v0.2.0`, `10.83 GB` GPU weights + `0.42 GB` host embedding + `0.35 GB` MTP on GPU on `v0.2.1` Win11).

### Workload & Reproducible Commands

Each prompt embeds a calibration passphrase at 50% depth inside tracked C++ repository sources (`1,022` / `131,081` / `260,049` tokens on `v0.2.0`; `260,051` tokens on the `v0.2.1` rerun) and asks the model to recall the passphrase and write a fresh 512-token pure-Python ISO 8601 date/timestamp parser with unit tests (greedy decoding; no repeated Python code in the prompt so prompt-lookup hits stay at 0–3 rounds).

```sh
# 1. Generate deterministic 1K / 128K / 256K prompts from this repository
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 1024   -o prompt_1k.txt
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 131072 -o prompt_128k.txt
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 260000 -o prompt_256k.txt

# 2. Run greedy coding benchmark at full native 262K context
./bin/flyx run MODEL.gguf \
  --drafter mtp --ctx max --chat --no-think -n 512 --ignore-eos -v -f prompt_128k.txt
```

### Hardware & Automatic 262K Memory Placement

| GPU | Model | VRAM / TDP / Driver | Automatic 262K (`--ctx max`) K/V placement |
|---|---|---|---|
| **RTX 5090** | `NVFP4` (`15.72 GB`) | `32,607 MiB` · `575 W` · driver `615.71.09` | Full `262,144` tokens exact FP8 K/V on GPU (`9.64 GB` VRAM); W4A4 NVFP4 prefill |
| **RTX 5080** | `UD-IQ3_XXS` (`10.93 GB`) | `16,303 MiB` · `360 W` · driver `610.57.04` | `43,008` tokens on GPU (`4.49 GB`) + `219,136` in RAM (`8.55 GB`); exact to `75,776`, then 2-bit StrataFlux + exact hot reads |
| **RTX 5070 Ti** (Win11, v0.2.1) | `UD-IQ3_XXS` (`10.93 GB`) | `16,303 MiB` · `300 W` · driver `610.88` · 32 GB RAM | `16,384` tokens on GPU (`3.71 GB`, `14.89 GB` WDDM budget) + `245,760` in RAM (`9.70 GB` incl. checkpoints); exact to `123,203` (`strata_from=123203`), then 2-bit StrataFlux + exact hot reads |
| **RTX 5070 Ti** (WSL2, v0.2.0) | `UD-IQ3_XXS` (`10.93 GB`) | `16,303 MiB` · `300 W` · driver `610.88` | `2,048` tokens on GPU (`2.42 GB`, `13.51 GB` WSL2/WDDM budget) + `260,096` in RAM (`10.62 GB`); exact to `34,816`, then 2-bit StrataFlux + exact hot reads |
| **RTX 4090** | `UD-Q4_K_XL` (`17.66 GB`) | `24,564 MiB` · `450 W` · driver `590.48.01` | `88,064` tokens on GPU + `174,080` in RAM; exact to `120,832`, then 2-bit StrataFlux + exact hot reads |
| **RTX 4080** | `UD-IQ3_XXS` (`10.93 GB`) | `16,376 MiB` · `340 W` · driver `580.178.04` | `43,008` tokens on GPU (`4.49 GB`) + `219,136` in RAM (`8.55 GB`); exact to `75,776`, then 2-bit StrataFlux + exact hot reads |
| **RTX 3090** | `UD-Q4_K_XL` (`17.66 GB`) | `24,576 MiB` · `300 W` cap · driver `595.58.03` | `92,160` tokens on GPU (`6.14 GB`) + `169,984` in RAM (`6.94 GB`); exact to `124,928`, then 2-bit StrataFlux + exact hot reads |

### Coding Results Across RTX 50 / 40 / 30 Series (Greedy, 512 New Tokens)

| GPU | Model & Drafter | Context (prompt tokens) | Prefill | Prefill (`tok/s`) | Decode (`tok/s`) | `tok/rnd` (`drafts/rnd`) | Needle recall |
|---|---|---|---:|---:|---:|---:|:---:|
| **RTX 5090** (32 GB) | `NVFP4` + MTP | 1K (`1,022`) | **`0.06 s`** | **`17,090.3`** | **350.2** | `5.27` (`6.8`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP | 128K (`131,081`) | **`29.70 s`** | **`4,414.0`** | **228.7** | `5.27` (`7.0`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP | 256K (`260,049`) | **`103.09 s`** | **`2,522.5`** | **164.5** | `4.96` (`6.8`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP & DFlash2 | 1K (`1,022`) | `0.10 s` | `9,883.9` | **365.0** | `5.06` (`6.7`) | pass |
| **RTX 5090** (32 GB) | `NVFP4` + MTP & DFlash2 | 128K (`131,081`) | `30.06 s` | `4,361.1` | **273.1** | `5.74` (`7.0`) | pass |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.44 s` | `2,312.7` | **225.9** | `5.06` (`6.2`) | pass |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `100.64 s` | `1,302.4` | **161.0** | `5.27` (`6.7`) | pass |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `283.49 s` | `917.3` | **105.8** | `3.84` (`5.6`) | pass |
| **RTX 5070 Ti** (16 GB, Win11, v0.2.1) | `UD-IQ3_XXS` + MTP | 256K (`260,051`) | `330.37 s` | `787.2` | **113.5** | `4.82` (`6.3`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2, v0.2.0) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.63 s` | `1,611.2` | **167.4** | `4.48` (`5.0`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2, v0.2.0) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `135.09 s` | `970.3` | **104.6** | `5.21` (`6.5`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2, v0.2.0) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `382.49 s` | `679.9` | **61.2** | `3.70` (`5.0`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 1K (`1,022`) | `0.37 s` | `2,763.7` | **193.6** | `5.94` (`9.0`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 128K (`131,081`) | `85.10 s` | `1,540.4` | **155.9** | `5.94` (`8.3`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 256K (`260,049`) | `241.20 s` | `1,078.2` | **122.7** | `5.06` (`6.9`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.55 s` | `1,864.3` | **172.3** | `5.11` (`6.1`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `127.00 s` | `1,032.2` | **111.0** | `4.29` (`5.9`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `358.46 s` | `725.5` | **99.0** | `4.69` (`6.4`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 1K (`1,022`) | `1.15 s` | `888.1` | **135.8** | `5.49` (`7.6`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 128K (`131,081`) | `254.71 s` | `514.6` | **97.5** | `5.55` (`7.3`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 256K (`260,049`) | `694.67 s` | `374.3` | **68.1** | `4.41` (`6.4`) | pass |

Note: The `RTX 5070 Ti (16 GB, Win11, v0.2.1)` row (`113.5 tok/s`, `512` tokens in `4,501.9 ms`, `106` rounds, `lookup 3`, `1.85×` vs `v0.2.0` WSL2 `61.2 tok/s`) ran the official `flyx-0.2.1-windows-x86_64-cuda13.exe` binary on native Windows 11 Pro (`10.0.26300`, 32 GB RAM). Only the 256K prompt (with 50%-depth needle recall) was rerun on `v0.2.1` Win11; the `1K` and `128K` RTX 5070 Ti rows above are from the earlier `v0.2.0` WSL2 run (`b950329b`).

---

## 2. Standard `flyx bench` Across GPUs & Drafters (Qwen3.8-27B)

Measured on the exact same build (`b950329b`) and machines as Section 1 across `flyx bench`'s 3 fixed mixed prompts (33..79 tokens, 128 new tokens each, greedy `tok/s`; `--ctx 4096` on 16 GB GPUs):

```sh
./bin/flyx bench MODEL.gguf --no-spec -n 128 --json
./bin/flyx bench MODEL.gguf --drafter mtp -n 128 --json
./bin/flyx bench MODEL.gguf --drafter dflash2 --drafter-file z-lab/Qwen3.8-27B-DFlash2 -n 128 --json
./bin/flyx bench MODEL.gguf --drafter all     --drafter-file z-lab/Qwen3.8-27B-DFlash2 -n 128 --json
```

### RTX 5090 (32 GB, `575 W`, driver `615.71.09`): Drafter Modes (`NVFP4`)

| Model | `flyx` plain | `flyx` + MTP | `flyx` + DFlash2 | `flyx` + both | Best Speedup |
|---|---:|---:|---:|---:|---:|
| `NVFP4` (`15.72 GB`) | **94.0** | **282.7** (`4.10 tok/rnd`) | **311.6** (`4.10 tok/rnd`) | **310.3** (`4.28 tok/rnd`) | **3.3x** |

### All 6 Consumer GPUs: Plain vs Built-In MTP (`flyx bench -n 128`)

| GPU | Model | `flyx` plain (`tok/s`) | `flyx` + MTP (`tok/s`) | Speedup |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` | **94.0** | **282.7** (`4.10 tok/rnd`) | **3.0x** |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` | **71.0** | **183.9** (`3.97 tok/rnd`) | **2.6x** |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` | **58.6** | **150.4** (`3.93 tok/rnd`) | **2.6x** |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` | **49.6** | **154.2** (`4.43 tok/rnd`) | **3.1x** |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` | **53.9** | **140.9** (`3.97 tok/rnd`) | **2.6x** |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` | **41.0** | **108.2** (`4.23 tok/rnd`) | **2.6x** |

Important caveats:
- **Drafted speed depends on workload predictability.** Code and structured output accept more drafts per round (`5.0–5.9 tok/rnd`) than mixed/prose prompts (`3.9–4.4 tok/rnd`).
- **Single-user latency focus:** Concurrent requests share decode rounds (`1.12x` aggregate throughput at 2 requests, `1.30x` at 4 requests on 27B 4-bit), but `flyx` is optimized for single-request latency rather than high-concurrency batching.

---

## 3. Long-Context Attention Fidelity (Needle-in-a-Haystack Suite)

To verify that neither default FP8 K/V nor tiered Host-RAM + 2-bit StrataFlux attention degrades long-range retrieval, we verified single-needle recall (`5%`, `25%`, `50%`, `75%`, `95%` depths across `32K–256K` plus `1K / 128K / 256K` 50%-depth recall on every GPU) and 4-needle simultaneous retrieval at `256K` (`ALPHA` / `BETA` / `GAMMA` / `DELTA` embedded at `10%`, `35%`, `65%`, `90%` depth) on `b950329b`: **51/51 (100%) exact recall across all 6 consumer GPUs, zero failures**. On `v0.2.1` native Windows 11 (`7f0de79`), the RTX 5070 Ti 256K coding rerun also passed its 50%-depth needle recall (`1/1`; the 5-depth and 4-needle grids were not rerun on `v0.2.1` Win11).

```sh
# Generate and run the 256K 4-needle NIAH prompt
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 260000 --multi-needle -o niah_256k.txt
./bin/flyx run MODEL.gguf --drafter mtp --ctx max --chat --no-think -n 96 -v -f niah_256k.txt
```

| GPU | Model & K/V Attention Mode | Single-Needle Recall | Multi-Needle at 256K (4 keys) | Total |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` · Full 262K GPU exact FP8 | `3/3` (`1K / 128K / 256K` @ `50%`) | `4/4` (`b950329b`) | **7/7 (100%)** |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` · 43K GPU + 219K RAM, 2-bit StrataFlux > 75K | `5/5` (`256K` @ `5%–95%`) | — | **5/5 (100%)** |
| **RTX 5070 Ti** (16 GB, Win11, v0.2.1) | `UD-IQ3_XXS` · `16,384` GPU + `245,760` RAM, 2-bit StrataFlux > `123,203` | `1/1` (`256K` @ `50%`, `7f0de79`) | — | **1/1 (100%)** |
| **RTX 5070 Ti** (16 GB, WSL2, v0.2.0) | `UD-IQ3_XXS` · 2K GPU + 260K RAM, 2-bit StrataFlux > 34K | `5/5` (`256K` @ `5%–95%`) | `4/4` (`b950329b`) | **9/9 (100%)** |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` · 88K GPU + 174K RAM, 2-bit StrataFlux > 120K | `20/20` (`32K–256K` @ `5%–95%`) | `4/4` (`b950329b`) | **24/24 (100%)** |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` · 43K GPU + 219K RAM, 2-bit StrataFlux > 75K | `3/3` (`1K / 128K / 256K` @ `50%`) | — | **3/3 (100%)** |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` · 92K GPU + 170K RAM, 2-bit StrataFlux > 124K | `3/3` (`1K / 128K / 256K` @ `50%`) | — | **3/3 (100%)** |

---

## 4. Benchmarking Tools in This Repo

| Tool | Purpose & Command |
|---|---|
| `flyx bench` | End-to-end decode speed + tokens/round + GPU clocks/power/throttling (`flyx bench MODEL -n 128 --json`). |
| `bench/prompts/make_coding_prompt.py` | Deterministic `1K–256K` long-context coding and 4-needle NIAH prompt generator from tracked repository sources. |
| `bench/ab/ab.sh` | Interleaved A/B comparison across builds or env vars: `bench/ab/ab.sh decode\|prefill MODEL A B`. |
| `bench/drafters_27b.sh` + `bench/drafters_table.py` | Multi-workload drafter comparison across `code`, `cache`, `list` and `reason` prompts. |
| `bench/prompts/` | Fixed prefill prompt fixtures (`p512.txt`, `long.txt`, `prompt_6k.txt`), and the long chat prompts and the prompt sets DV is measured on. |
| `bench/replay/agent_replay.py` | Multi-turn tool-using OpenAI replay measuring recomputed prompt tokens and TTFT p50/p95. |
| `cuda_bench` | Kernel microbenchmarks (`gemm`, `skinny`, `fp4`, `decode`, `tc`, `w4a4`, `w12`, `pick`, `gdn`, `attn`, `flash`, `mlp`, `blocks`, `q8runs`, `rounds`, `moe`, `strata`, `mma`, `host_read`, `graphs`) and tuning table generator (`cuda_bench tune [FAMILIES] [FILE.gguf...] [NxK...]`; NxK: a model's bf16 GEMMs alone, for a table that inherits the rest). |
| `tree_bench` | Speculative tree verification latency and acceptance: `tree_bench MODEL [--ctx-at P] [--widths 1,15,31,63] [--drafted T]` (every width timed from the same context; trees of up to 63 drafts where the backend lowers them). |
| `flyx bench --prompts` | A prompt set (`{"question"}`, `{"messages"}` or `{"ids"}` a line), every prompt's tokens and rounds in the `--json` record: `flyx bench MODEL --prompts FILE.jsonl [--suffix S] [--limit N] --json OUT.json`; `tests/e2e/check_dv.py` scores and compares such records. |
| `bench/ab/paired_records.sh`, `bench/ab/summarize_records.py`, `bench/ab/same_runs.py` | Two builds' `flyx bench --prompts` runs in turns (waiting for a quiet GPU), their medians, and whether two runs give the same answers and rounds. |
| `bench/ab/sglang_bench.py` | A `flyx bench --prompts` record's prompts, token for token, through an SGLang server: its decode tok/s as flyx counts it, its tokens a round, and how many answers are the record's (section 6). |
| `bench/replay/dv_choice` | DV's per-round choice replayed on recorded rounds (`bench/replay/dv_traces/`), each constant set otherwise: `python -I bench/replay/dv_choice.py table BUILD/bin/dv_choice`. |
| `bench/trace/dv_round_kernels.py` | Where a DV round's time goes, from an `nsys` trace of DV's rounds with graphs off. |
| `bench/trace/` | CUPTI kernel timeline analysis (`gaps.py`, `layer_times.py`, `per_matrix.py`, `prefill_kinds.py`). |

---

## 5. MiniCPM5-1B with DV (RTX 5060 Ti 8 GB)

Historical study using private drafter inputs. The original capture did not record a release commit.

Hardware: RTX 5060 Ti 8 GB, driver `610.88`, CUDA `13.5`, Windows 11 (native, MSVC), the desktop holding about 3.3 GB of the GPU. Model: `openbmb/MiniCPM5-1B` in BF16, its DV files made from bifill-dv's checkpoints on Hugging Face (`panmz/bifill-dv`, a private repository: without access to it the DV rows below cannot be reproduced; [models](models.md) has the commands, [DV](design/dv.md#the-description-and-the-file-abi-42) the files), binary16 K/V (`--kv 16`). Prompts: GSM8K's first 100 test questions as bifill-dv prompts them (`gsm8k_test.jsonl`, made from the public `openai/gsm8k` as below; the instruction below after each), greedy, at most 512 new tokens. Every DV run generates plain decoding's tokens for all 100 (`tests/e2e/check_dv.py`).

`gsm8k_test.jsonl` is GSM8K's test split, one `{"question", "answer"}` object a line in the dataset's order, as bifill-dv writes it (sha256 `ca6e7c5108b7ab7bc920c707e021b21bd9bdaf725e48a179b781ce4be1e9ad13`, its lines ending in CRLF; with LF line ends the same prompts, sha256 `752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac`):

```python
import json
from datasets import load_dataset  # pip install datasets
with open("gsm8k_test.jsonl", "w", encoding="utf-8", newline="\r\n") as f:
    for r in load_dataset("openai/gsm8k", "main", split="test"):
        f.write(json.dumps({"question": r["question"], "answer": r["answer"]}, ensure_ascii=False) + "\n")
```

| File | Drafter | Decode `tok/s` | Tokens / round |
|---|---|---:|---:|
| `minicpm5-1b.flyx` | none (`--no-lookup`) | 226.7 | 1.00 |
| `minicpm5-1b-dv-res2.flyx` (beam 31:8:cov) | DV, 31 drafts every round (`--width 31`) | 320.1 | 5.02 |
| `minicpm5-1b-dv-res2.flyx` | DV, rounds chosen per round | **338.7** | 4.30 |
| `minicpm5-1b-dv-b2h.flyx` (with its V-LoRA) | none (`--drafter none --no-lookup`) | 179.2 | 1.00 |
| `minicpm5-1b-dv-b2h.flyx` | DV, 31 drafts every round (`--width 31`) | 387.2 | 7.08 |
| `minicpm5-1b-dv-b2h.flyx` | DV, rounds chosen per round | **426.1** | 6.37 |

Commands:

```sh
S='\nPlease reason step by step, and put your final answer within \boxed{}.'
C="--prompts gsm8k_test.jsonl --kv 16 --ctx 2048 --limit 100"
flyx bench minicpm5-1b.flyx $C --suffix "$S" --no-lookup --json plain.json
flyx bench minicpm5-1b-dv-res2.flyx $C --suffix "$S" --drafter dv --width 31 --json dv31.json
flyx bench minicpm5-1b-dv-res2.flyx $C --suffix "$S" --drafter dv --json dv.json
flyx bench minicpm5-1b-dv-b2h.flyx $C --suffix "$S" --drafter none --no-lookup --json b2h_plain.json
flyx bench minicpm5-1b-dv-b2h.flyx $C --suffix "$S" --drafter dv --width 31 --json b2h31.json
flyx bench minicpm5-1b-dv-b2h.flyx $C --suffix "$S" --drafter dv --json b2h.json
python -I tests/e2e/check_dv.py --drafted dv31.json --plain plain.json --same-target
python -I tests/e2e/check_dv.py --drafted dv.json --plain plain.json --same-target
python -I tests/e2e/check_dv.py --drafted b2h31.json --plain b2h_plain.json --same-target
python -I tests/e2e/check_dv.py --drafted b2h.json --plain b2h_plain.json --same-target
```

`check_dv.py` passes for each DV run against the plain run of its file's target (`plain.json` for the `-dv-res2`
file, `b2h_plain.json` for the `-dv-b2h` file): the same generated tokens for every question.

After long prompts (the three `bench/prompts/summarize_*.txt`, 3110 to 5949 tokens; 200 tokens, a new engine each) `minicpm5-1b-dv-res2.flyx`'s DV of 31 drafts every round gives 1.75 to 1.95 tokens a round and decodes at 0.31 to 0.42 times plain decoding; with its rounds chosen per round it runs plain decoding's and lookup rounds there (1.00 to 1.02 tokens a round), at about plain decoding's speed. Every text is plain decoding's. DV drafts only when named.

```sh
for p in bench/prompts/summarize_*.txt; do
  flyx run minicpm5-1b.flyx -f $p --chat --no-think -n 200 --kv 16 --ctx 8192 --no-spec --no-lookup
  flyx run minicpm5-1b-dv-res2.flyx -f $p --chat --no-think -n 200 --kv 16 --ctx 8192 --drafter dv --width 31
  flyx run minicpm5-1b-dv-res2.flyx -f $p --chat --no-think -n 200 --kv 16 --ctx 8192 --drafter dv
done
```

---

## 6. Block Drafters on Qwen3.8-27B and Qwen3-8B: DSpine, DFlash2, DFlash, MTP (RTX PRO 6000)

One RTX PRO 6000 Blackwell Server Edition (96 GB, driver `595.112.07`; flyx at #9's two commits, Release,
`cmake --preset dev -DFLYX_CUDA_ARCHS=120-real` with CUDA 13.0's `nvcc`), greedy, thinking off unless said, up to 1024
new tokens a prompt, each prompt in a fresh session, every row of both tables from one run of this build:

- **Qwen3.8-27B**: `QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` (`4a22293a`, its MTP layer kept; `a07bf63e` changes its README
  alone), converted once with `z-lab/Qwen3.8-27B-DFlash2` (`50307d4c`, block 8) and once with
  `tencent/Qwen3.8-27B-DSpine-b16` (`260a6c23`).
- **Qwen3-8B**: `Qwen/Qwen3-8B` (`b968826d`, BF16), converted once with `tencent/Qwen3-8B-DSpine-b16` (`3b1e69d0`) and
  once with `z-lab/Qwen3-8B-DFlash-b16` (`9b41424b`; DFlash, DFlash2's block without its convolutions or selector, the
  baseline the DSpine authors compare against).
- **Prompts**: `bench/prompts/eval_sets.py` (the first 30 of GSM8K, MATH-500, HumanEval, MBPP and MT-Bench's first
  turns; the datasets at `740312ad`, `6e4ed1a2`, `7dce6050`, `4bb6404f`, `e3a795c5`) and
  `bench/prompts/chat_short.jsonl` (24). With thinking on, GSM8K and MBPP as token ids (`--think`): on the 27B the
  first 20 of each, on Qwen3-8B the first 30 (no opener: Qwen3's template writes none). The files' md5:

| set | md5 | set | md5 |
|---|---|---|---|
| `sets/gsm8k.jsonl` | `7c0efd4094ef6297d20e53756b54cfb9` | `sets/mtbench.jsonl` | `d6bf690974020dd78ebcf3f8c9dcc743` |
| `sets/math500.jsonl` | `0b6b13d8ec07f297d3e816121ea91f84` | `sets/chat_short.jsonl` | `abd29de8b71aa54b54b3f3a6f611d768` |
| `sets/humaneval.jsonl` | `ca05864a11736e1b4c9a1ad70faa674a` | `think27/gsm8k_think.jsonl` | `118b57a6ab4d61aeedcd9cc0cef67608` |
| `sets/mbpp.jsonl` | `eb6c14653c0ad6ed0c70c12f0190085b` | `think27/mbpp_think.jsonl` | `dfceded8edf146478b3bdf97b2fb81a5` |
| `think8/gsm8k_think.jsonl` | `8f5a581f7847fa9a2f46784c86302719` | `think8/mbpp_think.jsonl` | `7bf491c7c25897f9fb0883b9fa35368c` |

```sh
for d in openai/gsm8k HuggingFaceH4/MATH-500 openai/openai_humaneval google-research-datasets/mbpp \
         HuggingFaceH4/mt_bench_prompts; do hf download --repo-type dataset $d --local-dir hf/${d#*/}; done
python -I bench/prompts/eval_sets.py hf/ sets/ && cp bench/prompts/chat_short.jsonl sets/
python -I bench/prompts/eval_sets.py hf/ think27/ --n 20 --think Qwen3.8-27B-QUASAR-NVFP4/tokenizer.json --opener $'<think>\n'
python -I bench/prompts/eval_sets.py hf/ think8/ --think Qwen3-8B/tokenizer.json
PYTHONPATH=python/convert python -m flyx_convert --model Qwen3.8-27B-QUASAR-NVFP4 --draft Qwen3.8-27B-DFlash2 -o q27-dflash2.flyx
PYTHONPATH=python/convert python -m flyx_convert --model Qwen3.8-27B-QUASAR-NVFP4 --draft Qwen3.8-27B-DSpine-b16 -o q27-dspine.flyx
PYTHONPATH=python/convert python -m flyx_convert --model Qwen3-8B --draft Qwen3-8B-DSpine-b16 -o q8-dspine.flyx
PYTHONPATH=python/convert python -m flyx_convert --model Qwen3-8B --draft Qwen3-8B-DFlash-b16 -o q8-dflash.flyx
# S: each set of sets/ (and of think27/ or think8/ for the thinking rows); R.json: each run's record
flyx bench q27-dflash2.flyx --prompts S -n 1024 --json R.json --no-spec                 # plain (q27-dspine.flyx too)
flyx bench q27-dflash2.flyx --prompts S -n 1024 --json R.json --drafter mtp
flyx bench q27-dflash2.flyx --prompts S -n 1024 --json R.json --drafter dflash2         # chosen widths
flyx bench q27-dflash2.flyx --prompts S -n 1024 --json R.json --drafter dflash2 --width 7 --no-lookup   # and 15
flyx bench q27-dspine.flyx --prompts S -n 1024 --json R.json --drafter dspine           # and --width 15 --no-lookup
flyx bench q8-dspine.flyx --prompts S -n 1024 --json R.json --no-spec                   # plain
flyx bench q8-dspine.flyx --prompts S -n 1024 --json R.json --drafter dspine            # and --width 15 --no-lookup
flyx bench q8-dflash.flyx --prompts S -n 1024 --json R.json --no-spec                   # plain (DFlash's file)
flyx bench q8-dflash.flyx --prompts S -n 1024 --json R.json --drafter dflash            # and --width 15 --no-lookup
# DSpine from its exact weights (both models, GSM8K):
flyx bench q27-dspine.flyx --prompts sets/gsm8k.jsonl -n 1024 --json R.json --drafter dspine --width 15 \
  --no-lookup --backend-opt draft_weights=exact
# every drafted record against the plain record of its file: the same answers, token for token
python -I bench/ab/same_runs.py R.json PLAIN.json --answers-only
```

Decode `tok/s` (tokens a round). Every drafted answer is plain decoding's, token for token, in every configuration
(`same_runs.py --answers-only`; the 27B: 174 prompts and 40 with thinking on; the 8B: 234).

| Qwen3.8-27B | plain | MTP | DFlash2 | DSpine | DFlash2, 7 drafts | DFlash2, 15 drafts | DSpine, 15 drafts |
|---|---:|---:|---:|---:|---:|---:|---:|
| GSM8K | 81.5 | 308.3 (5.79) | 426.3 (6.75) | 388.4 (6.08) | 419.4 (6.20) | **445.8** (8.26) | 408.7 (7.51) |
| MATH-500 | 81.7 | 297.5 (5.40) | 421.6 (6.69) | 375.9 (5.86) | 413.0 (6.08) | **427.6** (7.87) | 378.3 (6.91) |
| HumanEval | 81.0 | 444.9 (10.55) | 610.6 (11.88) | 618.4 (12.09) | 478.5 (7.35) | 591.1 (11.65) | **634.2** (12.48) |
| MBPP | 80.8 | 285.6 (5.57) | 391.9 (6.26) | 389.9 (6.30) | 395.2 (6.00) | **401.4** (7.67) | 394.1 (7.47) |
| MT-Bench | 81.5 | 196.3 (3.13) | 227.4 (3.35) | 203.7 (3.03) | **229.3** (3.37) | 191.8 (3.51) | 173.4 (3.15) |
| chat | 81.6 | 217.7 (3.54) | **263.4** (3.94) | 238.0 (3.57) | 259.4 (3.81) | 219.4 (4.01) | 203.4 (3.69) |
| GSM8K, thinking | 81.5 | | | | **378.6** (5.60) | 363.7 (6.72) | 376.9 (6.93) |
| MBPP, thinking | 82.0 | | | | 423.0 (6.23) | **437.3** (8.05) | 429.2 (7.85) |

- A round costs DSpine what it costs DFlash2 at each width (the engine's load-time timings: 13.98 / 14.64 / 17.99 ms
  for 3 / 7 / 15 drafts against 13.88 / 14.57 / 18.01; plain decoding's 12.25), and DSpine's NVFP4 layer copies beside
  the NVFP4 target accept as its exact weights do (`draft_weights=exact`; GSM8K at 15 drafts: 7.51 against 7.49
  tokens a round, 408.7 against 354.2 tok/s).
- At 15 drafts, flyx's DFlash2 (its block of 8 stretched to 16 rows) accepts more than DSpine (its block of 16) on
  GSM8K (8.26 against 7.51 tokens a round) and fewer on HumanEval (11.65 against 12.48); at its own 7 drafts DFlash2
  accepts less (6.20 against 7.51; with thinking on 5.60 against 6.93). Its first drafts are accepted less often than
  DFlash2's (GSM8K, the first: 0.912 against 0.948; seven deep: 0.419 against 0.567, DFlash2 at 7 drafts). The faster
  of the two depends on the text: DSpine on HumanEval, DFlash2 on math, MBPP and chat.
- Chosen every round, DSpine's widths stay mostly at 7 drafts where 15 give more (GSM8K: 388.4 against 408.7 tok/s):
  a fresh session's acceptance per position starts from the controller's prior (0.8 - 0.02 i), and a reply of a few
  hundred tokens seldom reaches positions past 7 to learn them.

| Qwen3-8B | plain | DFlash | DSpine | DFlash, 15 drafts | DSpine, 15 drafts | Tencent's plugin (SGLang), 15 drafts |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 87.7 | 417.5 (5.86) | 649.5 (9.16) | 428.8 (6.12) | **684.4** (9.92) | 659.2 (10.00) |
| MATH-500 | 87.9 | 483.5 (6.87) | 581.3 (8.25) | 494.7 (7.06) | **601.1** (8.63) | 592.3 (8.81) |
| HumanEval | 87.9 | 424.8 (5.98) | 514.0 (7.19) | 434.9 (6.18) | **543.8** (7.81) | 531.3 (7.85) |
| MBPP | 87.8 | 418.5 (5.89) | 487.1 (6.79) | 431.4 (6.15) | **520.9** (7.50) | 513.5 (7.58) |
| MT-Bench | 87.6 | 195.8 (2.67) | **262.2** (3.51) | 190.7 (2.71) | 253.1 (3.61) | 242.1 (3.58) |
| chat | 87.8 | 233.9 (3.20) | 322.9 (4.37) | 230.5 (3.26) | **324.7** (4.62) | 306.1 (4.57) |
| GSM8K, thinking | 88.2 | 234.4 (3.18) | 301.5 (4.03) | 232.8 (3.28) | **305.0** (4.34) | 294.5 (4.39) |
| MBPP, thinking | 87.7 | 225.1 (3.06) | **270.2** (3.56) | 221.5 (3.13) | 262.6 (3.73) | 248.5 (3.76) |

- DFlash's round costs what DSpine's does (the load-time timings: 13.65 / 13.91 / 14.75 ms for 3 / 7 / 15 drafts
  against 13.06 / 13.30 / 14.55; plain decoding's 11.54), and DSpine accepts 1.2 to 1.6 times as many tokens a round
  and is that much faster: GSM8K at 15 drafts 9.92 against 6.12 tokens a round, 684.4 against 428.8 tok/s.
- On GSM8K DSpine accepts about what its authors report (at 15 drafts 9.92 tokens a round from its NVFP4 layer copies,
  10.03 from its exact weights; theirs 10.55), DFlash too (6.12; theirs 6.32); on MT-Bench both less (3.61 and 2.71
  against 5.63 and 4.26), in the same ratio.

**Against Tencent's own implementation.** Their SGLang plugin (`Tencent/DSpine` at `d037a9b`; Python 3.12, CUDA 13.0)
on the same GPU, model and drafter snapshots, the same prompts token for token (each flyx plain record's
`prompt_ids`), greedy, one request at a time, its README's Qwen3-8B server but for the attention backends
(FlashAttention 3 does not run on sm_120):

```sh
pip install sglang==0.5.22 torch==2.14.1 flashinfer-python==0.7.1 && pip install --no-deps -e DSpine
python -m sglang.launch_server --model-path Qwen3-8B --speculative-algorithm DSPINE \
  --speculative-draft-model-path Qwen3-8B-DSpine-b16 --speculative-num-draft-tokens 16 --dtype bfloat16 \
  --attention-backend flashinfer --speculative-draft-attention-backend triton --page-size 1 \
  --mem-fraction-static 0.65 --context-length 8192 --max-running-requests 8 --cuda-graph-bs-decode 1 2 4 8 \
  --trust-remote-code
python -I bench/ab/sglang_bench.py plain_gsm8k.json --json sglang_gsm8k.json
```

Its column was measured at #6 on another RTX PRO 6000 Blackwell Server Edition (driver `620.70.06`). flyx on that host,
DSpine at 15 drafts: at #6 (`c97fd62f`) GSM8K 692.1, MATH-500 614.5, HumanEval 555.0, MBPP 531.9, MT-Bench 258.5,
chat 331.8 tok/s, 1.1-2.2% above this table's at the same tokens a round; the DSpine change rebased onto `main` at
`527f0ba6` 682.5, 604.2, 545.8, 523.5, 254.4 and 326.2, within 0.5% of this table's (the 1-2% came with `main`). Against the plugin at 15 drafts flyx is 1.4-6.1% faster on every set with this table's
numbers (8.3% on MT-Bench with widths chosen every round) and 3.6-8.4% with #6's from the plugin's own host, at about
the same tokens a round: its round is shorter. Every flyx answer is plain decoding's; SGLang's speculative answers were
its own plain decoding's in 8 of the 30 GSM8K prompts (`--same-as` a run of the server without `--speculative-*`).
What made the difference on this GPU, measured on GSM8K and HumanEval at 15 drafts (617.6 / 490.8 tok/s before):

- **This GPU's table** (`backends/cuda/variants/sm120/tables/rtxpro6000.toml`, `cuda_bench tune gemm` on the four
  bf16 GEMMs Qwen3-8B runs): the attention out and mlp down projections at 9..16 rows read 1052 and 953 GB/s on the
  RTX 5090's rules, 1255 and 1008 on their own (+2-4%; bit-identical configurations).
- **DSpine's layer copies**: beside a BF16 target the drafter was bf16, its five layers 1.9 GB a round, each layer
  the size of a target layer; Q8_0 copies +6% at the same tokens a round, NVFP4 copies 2-3% more at most 1% fewer.
  DFlash's go to Q8_0 (GSM8K at 15 drafts: 398.0 -> 431.2 tok/s at the same tokens a round).
- **The walk's GEMMs** in one launch of up to 240 rows (+0.5%).

The trace that found them (`CUDA_INJECTION64_PATH=.../libflyx-trace.so FLYX_TRACE=round.tsv flyx bench ...`, then
`python -I bench/trace/gaps.py round.tsv`), before them: a round 15.35 ms, its graph busy 15.2 ms of it, the GEMMs 13.3.

## 7. EAGLE-4 on Qwen3.8-27B BF16 against SGLang's EAGLE-4 (RTX PRO 6000 Server Edition)

One RTX PRO 6000 Blackwell Server Edition (96 GB, 600 W, driver `620.70.06`, a Modal instance; flyx at this change,
`cmake --preset dev` for `sm_120` with CUDA 13.0's `nvcc`, its GEMMs from `variants/sm120/tables/rtxpro6000.toml`);
the six sets of section 6 (the same md5s; 174 prompts: chat 24, the others 30), up to 256 new tokens a prompt,
thinking off, both engines through their OpenAI `/v1/chat/completions` with the same requests
(`bench/ab/chat_bench.py`: request i of a set seed 7 + i, top-p 1, no penalties), every request once untimed first.

- **Model**: `Qwen/Qwen3.8-27B` (`37561816`, BF16), converted with `EAGLE-Corp/Qwen3.8-27B-EAGLE4` (`f58a5985`).
- **SGLang**: the EAGLE-4 release's SGLang (`eagle-harness 0.1.5`, `sglang 0.5.23.dev252+eagle.9`, its
  `EAGLE_HARNESS` worker) on the same GPU and checkpoints, its two documented recipes: the quickstart's
  (`--disable-overlap-schedule --speculative-num-steps 12 --speculative-eagle-topk 4
  --speculative-num-draft-tokens 16`, with `--max-running-requests 8 --mem-fraction-static 0.9`: at its defaults the
  mamba state reservation leaves no KV cache on one 96 GB GPU) and the measured one (`--dtype bfloat16
  --attention-backend flashinfer --page-size 1 --max-running-requests 32 --cuda-graph-max-bs-decode 32
  --cuda-graph-backend-decode full --chunked-prefill-size 16384 --mamba-ssm-dtype bfloat16
  --speculative-draft-model-quantization unquant --speculative-draft-attention-backend flashinfer
  --speculative-attention-mode prefill`, the same 12 steps, top 4, 16 tokens, `{"mode":"serve"}`), each case
  against the better of the two.

```sh
PYTHONPATH=python/convert python -m flyx_convert --model Qwen3.8-27B --draft Qwen3.8-27B-EAGLE4 -o q38-e4.flyx
flyx serve q38-e4.flyx --port 8080 --no-browser --ctx 16384 --presence-penalty 0 --drafter eagle4 --sessions 8
python -m sglang.launch_server --model-path Qwen3.8-27B --port 30000 --context-length 16384 \
  --speculative-algorithm EAGLE_HARNESS --speculative-draft-model-path Qwen3.8-27B-EAGLE4 ...   # a recipe above
python -I bench/ab/chat_bench.py --url URL --conc C --temp T --max-new 256 --json R.json chat_short.jsonl ...
```

One request at a time: decode `tok/s` (tokens after the first over the time from the first to the last, per
request), SGLang's measured recipe (its quickstart recipe: 117.3, 111.4 and 105.8 over all at T = 0, 0.7 and 1), and
the end-to-end throughput (every new token over the wall time, time to the first token included).

| concurrency 1 | flyx T = 0 | SGLang T = 0 | flyx T = 0.7 | SGLang T = 0.7 | flyx T = 1 | SGLang T = 1 |
|---|---:|---:|---:|---:|---:|---:|
| chat (24) | **92.5** | 89.6 | **83.9** | 83.3 | **79.5** | 78.6 |
| GSM8K | **155.4** | 148.1 | **151.1** | 141.9 | **141.9** | 133.6 |
| HumanEval | **254.2** | 206.4 | **242.5** | 204.0 | **232.3** | 183.3 |
| MATH-500 | **145.1** | 136.4 | **142.6** | 130.5 | **129.9** | 123.6 |
| MBPP | **156.8** | 147.9 | **142.6** | 139.0 | **136.1** | 133.4 |
| MT-Bench | **85.2** | 79.9 | **79.9** | 76.4 | 74.0 | **74.4** |
| all 174 | **127.5** | 119.6 | **120.7** | 113.9 | **112.8** | 108.7 |
| throughput | **123.8** | 116.4 | **117.3** | 111.0 | **109.9** | 106.1 |

Eight requests at a time (`--sessions 8`; SGLang's quickstart recipe, its measured one 299.9, 289.3 and 267.9):
throughput, every new token over the wall time of the 174 requests.

| concurrency 8 | flyx | SGLang |
|---|---:|---:|
| T = 0 | **371.4** | 352.2 |
| T = 0.7 | **347.5** | 339.7 |
| T = 1 | **321.1** | 319.9 |

Greedy, every flyx reply is its plain decoding's, alone and eight at a time. flyx's drafts are NVFP4
(`draft_weights=auto` beside a BF16 target; `exact`, its bf16 weights, 9% slower one at a time on chat and MT-Bench,
`q8` 3%), its rounds chosen by cost and acceptance among the drafter's chains of 1 to 15, its trees of 15 drafts (a
beam of 4 for 10 steps, best first greedy; sampled, the candidates scored by how likely each is the target's choice,
[tree verification](design/tree-verification.md)) and prompt-lookup rounds; eight sessions' rounds are batched up to 64
rows (`wide`), the chains drafted in lockstep, trees alone. Its 9..16-row GEMMs over K of 15360 and more split a
column tile's chunks among four blocks (the mlp down at 16 rows 1346 GB/s against 1206), and its prefill GEMMs of
up to 160 rows take a 6-stage pipeline (a chat prompt's prefill about 50 ms).
