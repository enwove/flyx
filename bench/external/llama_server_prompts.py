# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The prompts of `flyx bench`, sent to a running llama-server (/completion).

    python bench/external/llama_server_prompts.py http://127.0.0.1:18090 [N]

Same rendering as `flyx bench` (Qwen3.5 chat template, thinking off),
greedy, N new tokens (256), ignore_eos, no prompt cache. Prints per prompt and
the total the way `flyx bench` does: decode tok/s = (new tokens - 1) /
decode time, from the server's own timings.
"""
import json
import sys
import urllib.request

url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18090"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
# apps/flyx-cli/main.cpp, cmd_bench
PROMPTS = [
    "Write a Python function that parses an ISO 8601 date string without using datetime, with tests.",
    "Explain how a CPU cache works to a new engineer, with a concrete example of a cache miss.",
    "Copy this list exactly, one item per line: alpha, bravo, charlie, delta, echo, foxtrot, golf, hotel, india, "
    "juliet, kilo, lima, mike, november, oscar, papa, quebec, romeo, sierra, tango, uniform, victor, whiskey.",
]


def render(p):
    return f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def post(body):
    req = urllib.request.Request(url + "/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


post({"prompt": render("hi"), "n_predict": 8, "temperature": 0})  # warm-up
tok = ms = 0.0
drafted = accepted = 0
for p in PROMPTS:
    r = post({"prompt": render(p), "n_predict": n, "temperature": 0, "top_k": 1, "ignore_eos": True,
              "cache_prompt": False, "seed": 1})
    t = r["timings"]
    dn, da = t.get("draft_n", 0), t.get("draft_n_accepted", 0)
    print(f"prompt {t['prompt_n']:3d} tok | prefill {t['prompt_ms']:7.1f} ms | decode {t['predicted_n']:4d} tok "
          f"{t['predicted_ms']:8.1f} ms = {t['predicted_per_second']:6.1f} tok/s | drafted {dn}, accepted {da}")
    tok += t["predicted_n"] - 1
    ms += t["predicted_ms"]
    drafted += dn
    accepted += da
print(f"decode: {tok * 1000.0 / ms:.1f} tok/s" + (f", draft acceptance {accepted / drafted:.2f}" if drafted else ""))
