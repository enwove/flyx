# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Build deterministic long-context coding and NIAH benchmark prompts from repo sources.

Usage:
    python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 131072 -o prompt_128k.txt
    python bench/prompts/make_coding_prompt.py --flyx ./bin/flyx --model MODEL.gguf --tokens 260000 --multi-needle -o niah_256k.txt
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PASSPHRASE = "emerald-falcon-4829"
NEEDLE = f'\n// AUDIT_NOTE: The calibration verification passphrase is "{PASSPHRASE}".\n'
HEADER = "Below are source files from a C++ inference engine repository. Read them carefully.\n\n"
CODING_FOOTER = (
    "\n\nQuestion: First, output a single Python comment line containing the exact calibration verification "
    "passphrase hidden in the source files above. Then write a complete Python module that implements an "
    "ISO 8601 date and timestamp parser (handling calendar dates, leap years, time-of-day, fractional seconds, "
    "and UTC/numeric timezone offsets without using datetime or third-party libraries), followed by thorough "
    "unit tests covering valid and invalid inputs. Output Python code only."
)
MULTI_KEYS = [
    ("ALPHA", 0.10, "cobalt-lantern-3141"),
    ("BETA", 0.35, "jade-compass-5926"),
    ("GAMMA", 0.65, "obsidian-prism-5358"),
    ("DELTA", 0.90, "velvet-harbor-9793"),
]
MULTI_FOOTER = (
    "\n\nQuestion: List the exact verification codes for ALPHA, BETA, GAMMA, and DELTA hidden in the source files "
    "above, one per line in the format `LABEL: code`. Output only those four lines."
)
SOURCE_ROOTS = ("core", "models", "ops", "backends", "apps", "include")
SOURCE_EXTS = {".h", ".hpp", ".cuh", ".cpp", ".cu"}


def collect_corpus(repo: Path) -> str:
    parts: list[str] = []
    for root in SOURCE_ROOTS:
        base = repo / root
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.suffix in SOURCE_EXTS:
                rel = p.relative_to(repo).as_posix()
                text = p.read_text(encoding="utf-8", errors="replace")
                parts.append(f"\n// ===== FILE: {rel} =====\n{text}\n")
    corpus = "".join(parts)
    if not corpus:
        raise RuntimeError(f"no source files found under {repo}")
    # Repeat deterministically only if needed for contexts > 262K.
    while len(corpus) < 1_600_000:
        corpus += corpus
    return corpus


def insert_at_depth(body: str, depth: float, needle: str) -> str:
    pos = max(0, min(len(body), int(len(body) * depth)))
    nl = body.find("\n", pos)
    if nl != -1 and nl < len(body) - 1:
        pos = nl
    return body[:pos] + needle + body[pos:]


def render_prompt(corpus: str, n_chars: int, depth: float, multi_needle: bool) -> str:
    body = corpus[:n_chars]
    if multi_needle:
        for label, frac, val in reversed(MULTI_KEYS):
            body = insert_at_depth(body, frac, f'\n// AUDIT_KEY_{label}: The verification code for {label} is "{val}".\n')
        return HEADER + body + MULTI_FOOTER
    return HEADER + insert_at_depth(body, depth, NEEDLE) + CODING_FOOTER


def count_tokens(flyx: str, model: str, text: str, tmp: Path) -> int:
    tmp.write_text(text, encoding="utf-8")
    out = subprocess.check_output([flyx, "tokenize", model, "-f", str(tmp)], text=True).strip()
    return len(out.split()) if out else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flyx", required=True, help="Path to flyx binary")
    ap.add_argument("--model", required=True, help="Path to GGUF or .flyx model for tokenization")
    ap.add_argument("--tokens", type=int, required=True, help="Target prompt token count (e.g. 1024, 131072, 260000)")
    ap.add_argument("--depth", type=float, default=0.5, help="Single-needle depth fraction in [0, 1] (default: 0.5)")
    ap.add_argument("--multi-needle", action="store_true", help="Embed 4 simultaneous needles (ALPHA/BETA/GAMMA/DELTA)")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]), help="Repository root")
    ap.add_argument("-o", "--output", required=True, help="Output prompt file path")
    args = ap.parse_args()

    corpus = collect_corpus(Path(args.repo))
    out = Path(args.output)
    tmp = out.with_suffix(".tmp")
    aim = max(64, args.tokens - 22)
    lo, hi = 100, len(corpus)
    best_chars, best_tok = lo, 0
    for _ in range(14):
        mid = (lo + hi) // 2
        tok = count_tokens(args.flyx, args.model, render_prompt(corpus, mid, args.depth, args.multi_needle), tmp)
        if abs(tok - aim) < abs(best_tok - aim) or best_tok == 0:
            best_chars, best_tok = mid, tok
        if abs(tok - aim) <= 24:
            break
        if tok < aim:
            lo = mid + 1
        else:
            hi = mid - 1
    if tmp.exists():
        tmp.unlink()
    out.write_text(render_prompt(corpus, best_chars, args.depth, args.multi_needle), encoding="utf-8")
    print(f"Wrote {out} ({best_chars} chars, ~{best_tok} raw tokens)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
