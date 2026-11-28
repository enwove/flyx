# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Four long chat prompts (one user message each, thinking off) for long-context measurements, as a prompt set.

Usage:
    python -I bench/prompts/make_long_chat_prompts.py --rev REV --book pg2600.txt --hf HF_DIR [--tokens 200000]
        [--repo REPO] -o OUT.jsonl

The prompts are built from flyx's sources at commit REV of the git checkout REPO (this one by default: the files under
core models ops backends apps include, and the Python files under python tests bench tools, read with `git archive`)
and from Project Gutenberg's #2600 (War and Peace, the plain-text file `2/6/0/2600/2600-0.txt` of its mirrors, checked
against PG_SHA256), each with needles at fixed depths:

    cpp_iso       the C++ and CUDA sources; a passphrase at half depth; then an ISO 8601 parser with tests to write
    cpp_explain   the sources from backends/ on; four codes at 10, 35, 65 and 90%; then the files to explain
    py_edit       the Python sources; a passphrase at a quarter; then one of the files to repeat with docstrings
    book_summary  the novel from Book One; a phrase at three quarters; then a summary chapter by chapter

HF_DIR is the target model's Hugging Face tokenizer (tokenizer.json, tokenizer_config.json and its chat template):
every prompt is the longest text that ends at a line end and whose chat-templated tokens (one user message, a
generation prompt, enable_thinking false) are at most --tokens; the text of the tokenizer's added tokens is broken
with a space wherever it occurs, so that the sources' own special-token literals are read as text.  The output is one
{"question": ...} object a line (`chat_bench.py`, `flyx bench --prompts`).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath

PG_SHA256 = "e4bcf9042609b62c7de72a6f1b311f54c412943a9d641b7efcf79a464b5f31c8"
CPP_ROOTS, CPP_EXTS = ("core", "models", "ops", "backends", "apps", "include"), {".h", ".hpp", ".cuh", ".cpp", ".cu"}
PY_ROOTS = ("python", "tests", "bench", "tools")
MULTI = [("ALPHA", 0.10, "cobalt-lantern-3141"), ("BETA", 0.35, "jade-compass-5926"),
         ("GAMMA", 0.65, "obsidian-prism-5358"), ("DELTA", 0.90, "velvet-harbor-9793")]


def sources(repo: str, rev: str, roots) -> dict[str, str]:
    """The text of every file under roots at rev, by its path (newlines as universal newlines read them)."""
    tar = subprocess.run(["git", "-C", repo, "archive", "--format=tar", rev, *roots], check=True,
                         capture_output=True).stdout
    out = {}
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        for m in t.getmembers():
            if m.isfile():
                text = t.extractfile(m).read().decode("utf-8", errors="replace")
                out[m.name] = text.replace("\r\n", "\n").replace("\r", "\n")
    return out


def tree(files: dict[str, str], roots, keep, mark: str) -> str:
    parts = []
    for root in roots:
        for name in sorted((n for n in files if n.startswith(root + "/")), key=lambda n: PurePosixPath(n).parts):
            if keep(PurePosixPath(name)):
                parts.append(f"\n{mark} ===== FILE: {name} =====\n{files[name]}\n")
    return "".join(parts)


def book(path: Path) -> str:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PG_SHA256:
        raise SystemExit(f"{path}: sha256 is not {PG_SHA256}")
    t = raw.decode("utf-8-sig").replace("\r\n", "\n")
    t = t[t.index("*** START OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***"):]
    t = t[:t.index("*** END OF THE PROJECT GUTENBERG EBOOK WAR AND PEACE ***")]
    return t[t.index("\nBOOK ONE: 1805\n") + 1:]  # past the header and the table of contents


def scrub(text: str, added: list[str]) -> str:
    for t in sorted(added, key=len, reverse=True):
        text = text.replace(t, t[0] + " " + t[1:])
    return text


def at_depth(body: str, frac: float, needle: str, sep: str) -> str:
    pos = body.find(sep, int(len(body) * frac))
    pos = len(body) if pos < 0 else pos
    return body[:pos] + needle + body[pos:]


def fit(count, render, corpus: str, start: int, target: int):
    """The longest body corpus[start:end] (end at a line end) whose prompt has at most target tokens."""
    lo, hi, best = start + 1000, len(corpus), None
    while lo <= hi:
        mid = (lo + hi) // 2
        end = corpus.rfind("\n", start, mid) + 1
        q = render(corpus[start:end])
        n = count(q)
        if n <= target:
            best, lo = (q, n, end), mid + 1
        else:
            hi = mid - 1
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", required=True, help="the commit whose sources the prompts read")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]), help="a git checkout holding REV")
    ap.add_argument("--book", required=True, help="Project Gutenberg #2600 as plain text (sha256 PG_SHA256)")
    ap.add_argument("--hf", required=True, help="the target's Hugging Face tokenizer directory")
    ap.add_argument("--tokens", type=int, default=200000)
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.hf)
    config = json.loads((Path(a.hf) / "tokenizer_config.json").read_text(encoding="utf-8"))
    added = [v["content"] for v in config["added_tokens_decoder"].values()]

    def count(question: str) -> int:
        return len(tok.apply_chat_template([{"role": "user", "content": question}], tokenize=True,
                                           add_generation_prompt=True, enable_thinking=False, return_dict=False))

    files = sources(a.repo, a.rev, CPP_ROOTS + PY_ROOTS)
    cpp = scrub(tree(files, CPP_ROOTS, lambda p: p.suffix in CPP_EXTS, "//"), added)
    py = scrub(tree(files, PY_ROOTS, lambda p: p.suffix == ".py", "#"), added)
    nov = scrub(book(Path(a.book)), added)

    # the file py_edit asks to repeat: the first Python file past 70% of the body its tokens are expected to take
    f_at = py.find("\n# ===== FILE: ", int(0.70 * int(3.38 * a.tokens)))
    f_path = py[f_at:].split(" =====\n", 1)[0].removeprefix("\n# ===== FILE: ")

    def p_cpp_iso(body):
        return ("Below are source files from a C++ inference engine repository. Read them carefully.\n\n"
                + at_depth(body, 0.5, '\n// AUDIT_NOTE: The calibration verification passphrase is '
                                      '"emerald-falcon-4829".\n', "\n")
                + "\n\nQuestion: First, output a single Python comment line containing the exact calibration "
                  "verification passphrase hidden in the source files above. Then write a complete Python module "
                  "that implements an ISO 8601 date and timestamp parser (handling calendar dates, leap years, "
                  "time-of-day, fractional seconds, and UTC/numeric timezone offsets without using datetime or "
                  "third-party libraries), followed by thorough unit tests covering valid and invalid inputs. "
                  "Output Python code only.")

    def p_cpp_explain(body):
        for label, frac, val in reversed(MULTI):
            body = at_depth(body, frac, f'\n// AUDIT_KEY_{label}: The verification code for {label} is "{val}".\n',
                            "\n")
        return ("These are the CUDA and C++ sources of the GPU backend of an inference engine. Study them closely."
                "\n\n" + body
                + "\n\nQuestion: First, list the exact verification codes for ALPHA, BETA, GAMMA, and DELTA hidden "
                  "in the source files above, one per line in the format `LABEL: code`. Then explain in detail, file "
                  "by file, what the source files above implement and how their main functions work together when "
                  "the engine decodes one token.")

    def p_py_edit(body):
        return ("Here are the Python sources of a model converter, its reference implementations and its tests. "
                "Read all of them.\n\n"
                + at_depth(body, 0.25, '\n# AUDIT_NOTE: The release signing passphrase is "silver-meadow-2718".\n',
                           "\n")
                + f"\n\nQuestion: First, output a single Python comment line containing the exact release signing "
                  f"passphrase hidden in the files above. Then output the complete file `{f_path}` from above again, "
                  f"unchanged except that every function and method without a docstring gets a one-line docstring. "
                  f"Output Python code only.")

    def p_book_summary(body):
        return ("Below is the beginning of a long novel. Read it carefully.\n\n"
                + at_depth(body, 0.75, "The archivist's verification phrase for this edition is "
                                       '"amber-orchard-6153".\n\n', "\n\n")
                + "\n\nQuestion: First, write the exact verification phrase of the archivist hidden in the text above "
                  "on a line of its own. Then summarize the text above chapter by chapter, in order, naming the "
                  "characters each chapter involves.")

    rows = []
    for name, render, corpus, start in (("cpp_iso", p_cpp_iso, cpp, 0),
                                         ("cpp_explain", p_cpp_explain, cpp, cpp.find("\n// ===== FILE: backends/")),
                                         ("py_edit", p_py_edit, py, 0), ("book_summary", p_book_summary, nov, 0)):
        q, n, end = fit(count, render, corpus, start, a.tokens)
        assert q == q.strip() and not any(t in q for t in added), name
        assert name != "py_edit" or f"===== FILE: {f_path} =====" in q, f_path
        rows.append({"question": q})
        print(f"{name}: {n} tokens, {len(q)} chars, body {start}..{end}", flush=True)
    with open(a.output, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(a.output, hashlib.md5(Path(a.output).read_bytes()).hexdigest(), "py_edit repeats", f_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
