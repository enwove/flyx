# Benchmarks

Reproducible long-context coding, `flyx bench` decode, drafter, and Needle-in-a-Haystack measurements for `flyx` on consumer NVIDIA GPUs with exact hardware, models, and commands.

---

## v0.2.1 four-GPU verified benchmarks

Linux captures used CUDA 13.0.88 on Ubuntu 24.04 with PTX JIT disabled. The hardware table below describes the original `5fa1a32` runs. The RTX 4090 and 3090 Inkling decode results come from a second capture on `078022fc`. Windows measurements used native Windows 11 (WDDM), with Inkling and the dashboard on `5fa1a32`, and Qwen, the 20K ring and cold start on `42e3734` before the Inkling decode changes. The Windows candidate passed 170/170 GPU-suite tests. Its standalone executable contained 115 cubins and zero PTX images and weighed 57,144,832 bytes (57.14 MB), compared with 58,960,384 bytes in v0.2.0. These captures predate the version bump.

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

Sections 1–3 below preserve the original six-card measurements on `b950329b`, including the WSL2 run. The independent DV study in section 5 uses a different model and workload.

## 1. Qwen3.8-27B Long-Context Coding Benchmark

All rows below were measured on the same `release-linux` build (`flyx 0.2.0`, commit `b950329b`, CUDA `13.0`, per-GPU tuning tables) at full native context (`--ctx max` = `262,144` tokens) with built-in **MTP** speculative decoding (plus `MTP + DFlash2` on RTX 5090 `NVFP4`):

- **RTX 5090 (Blackwell native FP4):** [`akopytko/Qwen3.8-27B-NVFP4-GGUF`](https://huggingface.co/akopytko/Qwen3.8-27B-NVFP4-GGUF) `Qwen3.8-27B-NVFP4-MTP-N4_0.gguf` (`NVFP4`, `15.72 GB` file, `15.20 GB` GPU weights; 504 GGML NVFP4 tensors executing on native `sm_120a` W4A4 FP4 tensor cores in prefill + built-in MTP).
- **24 GB GPUs (RTX 4090, RTX 3090):** [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) `UD-Q4_K_XL` (`17.66 GB` file, `17.24 GB` GPU weights).
- **16 GB GPUs (RTX 5080, RTX 5070 Ti, RTX 4080):** `unsloth/Qwen3.8-27B-GGUF:UD-IQ3_XXS` (`10.93 GB` file, `10.72 GB` GPU weights).

### Workload & Reproducible Commands

Each prompt embeds a calibration passphrase at 50% depth inside tracked C++ repository sources (`1,022` / `131,081` / `260,049` tokens) and asks the model to recall the passphrase and write a fresh 512-token pure-Python ISO 8601 date/timestamp parser with unit tests (greedy decoding; no repeated Python code in the prompt so prompt-lookup hits stay at 0–3 rounds).

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
| **RTX 5070 Ti** (WSL2) | `UD-IQ3_XXS` (`10.93 GB`) | `16,303 MiB` · `300 W` · driver `610.88` | `2,048` tokens on GPU (`2.42 GB`, `13.51 GB` WSL2/WDDM budget) + `260,096` in RAM (`10.62 GB`); exact to `34,816`, then 2-bit StrataFlux + exact hot reads |
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
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.63 s` | `1,611.2` | **167.4** | `4.48` (`5.0`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `135.09 s` | `970.3` | **104.6** | `5.21` (`6.5`) | pass |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `382.49 s` | `679.9` | **61.2** | `3.70` (`5.0`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 1K (`1,022`) | `0.37 s` | `2,763.7` | **193.6** | `5.94` (`9.0`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 128K (`131,081`) | `85.10 s` | `1,540.4` | **155.9** | `5.94` (`8.3`) | pass |
| **RTX 4090** (24 GB) | `UD-Q4_K_XL` + MTP | 256K (`260,049`) | `241.20 s` | `1,078.2` | **122.7** | `5.06` (`6.9`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 1K (`1,022`) | `0.55 s` | `1,864.3` | **172.3** | `5.11` (`6.1`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 128K (`131,081`) | `127.00 s` | `1,032.2` | **111.0** | `4.29` (`5.9`) | pass |
| **RTX 4080** (16 GB) | `UD-IQ3_XXS` + MTP | 256K (`260,049`) | `358.46 s` | `725.5` | **99.0** | `4.69` (`6.4`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 1K (`1,022`) | `1.15 s` | `888.1` | **135.8** | `5.49` (`7.6`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 128K (`131,081`) | `254.71 s` | `514.6` | **97.5** | `5.55` (`7.3`) | pass |
| **RTX 3090** (24 GB, 300 W) | `UD-Q4_K_XL` + MTP | 256K (`260,049`) | `694.67 s` | `374.3` | **68.1** | `4.41` (`6.4`) | pass |

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

To verify that neither default FP8 K/V nor tiered Host-RAM + 2-bit StrataFlux attention degrades long-range retrieval, we verified single-needle recall (`5%`, `25%`, `50%`, `75%`, `95%` depths across `32K–256K` plus `1K / 128K / 256K` 50%-depth recall on every GPU) and 4-needle simultaneous retrieval at `256K` (`ALPHA` / `BETA` / `GAMMA` / `DELTA` embedded at `10%`, `35%`, `65%`, `90%` depth) on `b950329b`: **51/51 (100%) exact recall across all 6 consumer GPUs, zero failures**.

```sh
# Generate and run the 256K 4-needle NIAH prompt
python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 260000 --multi-needle -o niah_256k.txt
./bin/flyx run MODEL.gguf --drafter mtp --ctx max --chat --no-think -n 96 -v -f niah_256k.txt
```

| GPU | Model & K/V Attention Mode | Single-Needle Recall | Multi-Needle at 256K (4 keys) | Total |
|---|---|---:|---:|---:|
| **RTX 5090** (32 GB) | `NVFP4` · Full 262K GPU exact FP8 | `3/3` (`1K / 128K / 256K` @ `50%`) | `4/4` (`b950329b`) | **7/7 (100%)** |
| **RTX 5080** (16 GB) | `UD-IQ3_XXS` · 43K GPU + 219K RAM, 2-bit StrataFlux > 75K | `5/5` (`256K` @ `5%–95%`) | — | **5/5 (100%)** |
| **RTX 5070 Ti** (16 GB, WSL2) | `UD-IQ3_XXS` · 2K GPU + 260K RAM, 2-bit StrataFlux > 34K | `5/5` (`256K` @ `5%–95%`) | `4/4` (`b950329b`) | **9/9 (100%)** |
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
| `cuda_bench` | Kernel microbenchmarks (`gemm`, `skinny`, `fp4`, `decode`, `tc`, `w4a4`, `w12`, `pick`, `gdn`, `attn`, `flash`, `mlp`, `blocks`, `q8runs`, `rounds`, `moe`, `strata`, `mma`, `host_read`, `graphs`) and tuning table generator (`cuda_bench tune [FAMILIES] [FILE.gguf...]`). |
| `tree_bench` | Speculative tree verification latency and acceptance: `tree_bench MODEL [--ctx-at P] [--widths 1,15,31,63] [--drafted T]` (every width timed from the same context; trees of up to 63 drafts where the backend lowers them). |
| `flyx bench --prompts` | A prompt set (`{"question"}` or `{"messages"}` a line), every prompt's tokens and rounds in the `--json` record: `flyx bench MODEL --prompts FILE.jsonl [--suffix S] [--limit N] --json OUT.json`; `tests/e2e/check_dv.py` scores and compares such records. |
| `bench/ab/paired_records.sh`, `bench/ab/summarize_records.py`, `bench/ab/same_runs.py` | Two builds' `flyx bench --prompts` runs in turns (waiting for a quiet GPU), their medians, and whether two runs give the same answers and rounds. |
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
