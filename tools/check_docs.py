# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Public documentation hygiene and anti-drift check (ctest `docs`).

    python tools/check_docs.py [REPO]

Checks:
* Every relative markdown/HTML link (and optional #anchor) in public docs and llms.txt resolves.
* llms.txt indexes every markdown document under docs/ plus core root docs.
* No internal/dev residue in public docs (vast.ai, ninfer, docs/perf/, docs/decisions/,
  decision 00xx, personal paths, CJK text, stray .patch/.txt files under docs/).
* Every engine/server CLI flag (apps/common/engine_flags.cpp, apps/flyx-server/main.cpp) and
  CUDA backend option (backends/cuda/backend.cpp) is documented in docs/server.md.
"""
from __future__ import annotations

import os
import re
import sys


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _slugify(heading: str) -> str:
    s = re.sub(r"`([^`]*)`", r"\1", heading.strip().lower())
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return s


def _headings(text: str) -> set[str]:
    out: set[str] = set()
    counts: dict[str, int] = {}
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = re.match(r"^#{1,6}\s+(.*)$", line)
        if not m:
            continue
        base = _slugify(m.group(1))
        n = counts.get(base, 0)
        counts[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    return out


def _strip_code(text: str) -> str:
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def _public_doc_files(root: str) -> list[str]:
    files = ["llms.txt", "README.md", "README_zh.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md",
             "tools/ci/README.md", "bench/prompts/README.md"]
    for sub in ("docs", ".github"):
        base = os.path.join(root, sub)
        if not os.path.isdir(base):
            continue
        for dp, _, fns in os.walk(base):
            for fn in sorted(fns):
                if fn.endswith(".md"):
                    rel = os.path.relpath(os.path.join(dp, fn), root).replace("\\", "/")
                    files.append(rel)
    return sorted(set(f for f in files if os.path.isfile(os.path.join(root, f))))


def _extract_links(text: str) -> list[str]:
    stripped = _strip_code(text)
    links = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", stripped)
    links += re.findall(r'(?:href|src)="([^"]+)"', stripped)
    # <source srcset="a.svg"> (a <picture>'s other theme): each candidate's URL, without its width or density.
    for srcset in re.findall(r'srcset="([^"]+)"', stripped):
        links += [c.split()[0] for c in srcset.split(",") if c.strip()]
    return links


def main(root: str) -> int:
    root = os.path.abspath(root)
    bad: list[str] = []

    # 1. No stray files or banned directories under docs/
    for banned in ("docs/perf", "docs/decisions"):
        if os.path.exists(os.path.join(root, banned)):
            bad.append(f"{banned}: internal directory must not exist in public repo")
    docs_root = os.path.join(root, "docs")
    if os.path.isdir(docs_root):
        for dp, _, fns in os.walk(docs_root):
            for fn in sorted(fns):
                rel = os.path.relpath(os.path.join(dp, fn), root).replace("\\", "/")
                if not (fn.endswith(".md") or (rel.startswith("docs/brand/") and fn.endswith(".svg")) or
                        (rel.startswith("docs/media/") and fn.endswith(".png"))):
                    bad.append(f"{rel}: unexpected non-markdown file under docs/")

    public_files = _public_doc_files(root)
    heading_cache: dict[str, set[str]] = {}

    def get_headings(path: str) -> set[str]:
        if path not in heading_cache:
            heading_cache[path] = _headings(_read(path))
        return heading_cache[path]

    # 2. Links and hygiene across public docs
    leak_patterns = [
        ("vast.ai", re.compile(r"vast\.ai", re.I)),
        ("NInfer", re.compile(r"\bninfer\b", re.I)),
        ("docs/perf", re.compile(r"docs/perf\b")),
        ("docs/decisions", re.compile(r"docs/decisions\b|(?:\.\./)?decisions/\d{4}")),
        ("decision 00xx", re.compile(r"\bdecisions?\s+0\d{3}\b", re.I)),
        ("personal path", re.compile(r"(?:/home/[a-zA-Z0-9._-]+/|[A-Za-z]:\\Users\\)")),
        ("CJK text", re.compile(r"[\u4e00-\u9fff]")),
    ]

    for rel in public_files:
        full = os.path.join(root, rel)
        text = _read(full)
        cjk_text = re.sub(r'<a href="README_[a-z]+\.md">[^<]+</a>', "", text) if rel == "README.md" else text
        for label, pat in leak_patterns:
            if label == "CJK text" and rel == "README_zh.md":
                continue
            target_text = cjk_text if label == "CJK text" else text
            m = pat.search(target_text)
            if m:
                bad.append(f"{rel}: contains forbidden {label} ({m.group(0)!r})")
        for raw in _extract_links(text):
            if raw.startswith(("http://", "https://", "mailto:")):
                continue
            target_part, _, anchor = raw.partition("#")
            target_path = full if not target_part else os.path.normpath(os.path.join(os.path.dirname(full), target_part))
            if not os.path.exists(target_path):
                bad.append(f"{rel}: broken link -> {raw}")
            elif anchor and target_path.endswith(".md"):
                if anchor not in get_headings(target_path):
                    bad.append(f"{rel}: broken anchor -> {raw}")

    # 3. llms.txt indexes every docs/**/*.md and key root docs
    llms_path = os.path.join(root, "llms.txt")
    if not os.path.isfile(llms_path):
        bad.append("llms.txt: missing root agent index")
    else:
        llms_links = {
            link.partition("#")[0]
            for link in _extract_links(_read(llms_path))
            if not link.startswith(("http://", "https://", "mailto:"))
        }
        expected = {"README.md", "README_zh.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md"}
        expected.update(f for f in public_files if f.startswith("docs/") and f.endswith(".md"))
        for req in sorted(expected - llms_links):
            bad.append(f"llms.txt: missing index link for {req}")

    # 4. Anti-drift: CLI flags, env vars, and CUDA --backend-opt keys documented in docs/server.md
    server_doc = _read(os.path.join(root, "docs/server.md"))
    for src_rel in ("apps/common/engine_flags.cpp", "apps/flyx-server/main.cpp"):
        src = _read(os.path.join(root, src_rel))
        flags = set(re.findall(r'==\s*"(--[a-z0-9-]+)"', src)) - {"--help"}
        if not flags:
            bad.append(f"{src_rel}: failed to extract any CLI flags")
        for flag in sorted(flags):
            if f"`{flag}`" not in server_doc:
                bad.append(f"docs/server.md: missing CLI flag `{flag}` from {src_rel}")

    env_src = _read(os.path.join(root, "core/os/env.h"))
    env_vars = set(re.findall(r'"((?:FLYX|HF)_[A-Z0-9_]+)"', env_src))
    if not env_vars:
        bad.append("core/os/env.h: failed to extract any FLYX_*/HF_* environment variables")
    for var in sorted(env_vars):
        if f"`{var}`" not in server_doc:
            bad.append(f"docs/server.md: missing env var `{var}` from core/os/env.h")

    cuda_src = _read(os.path.join(root, "backends/cuda/backend.cpp"))
    m_specs = re.search(r"const std::vector<OptionSpec>& option_specs\(\)(.*?)return specs;", cuda_src, re.S)
    if not m_specs:
        bad.append("backends/cuda/backend.cpp: failed to locate option_specs()")
    else:
        body = m_specs.group(1)
        keys = set(re.findall(r'(?:_opt\(|^\s*\{)"([a-z0-9_]+)"', body, re.M))
        if not keys:
            bad.append("backends/cuda/backend.cpp: failed to extract any CUDA backend options")
        for key in sorted(keys):
            if f"`{key}`" not in server_doc:
                bad.append(f"docs/server.md: missing CUDA backend option `{key}` from backends/cuda/backend.cpp")

    for b in bad:
        print("FAIL", b)
    print("docs: " + ("ok" if not bad else f"{len(bad)} problems"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")))
