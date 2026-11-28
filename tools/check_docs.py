# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Public documentation hygiene and anti-drift check (ctest `docs`).

    python tools/check_docs.py [REPO]

Checks:
* Every relative markdown/HTML link (and optional #anchor) in public docs and llms.txt resolves.
* llms.txt indexes every markdown document under docs/ plus core root docs.
* No internal/dev residue in public docs (vast.ai, ninfer, docs/perf/, docs/decisions/,
  decision 00xx, personal paths, CJK text, stray .patch/.txt files under docs/).
* docs/server.md covers engine/server CLI flags, runtime environment variables and CUDA backend options, and each
  legacy environment spelling of a CUDA backend option has a row naming its option.
* First-party runtime sources route FLYX_* reads through the registry (core, apps) or backend options (backends);
  only the CUDA backend's table of its options' legacy environment spellings (kEnvAliases) reads names it computes.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
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
    files = ["llms.txt", "README.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md",
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


# The one place a backend reads FLYX_* variables of its own: the CUDA backend's table of the legacy environment
# spellings of some of its options, kEnvAliases ({"FLYX_<NAME>", "<name>", ...}), which resolve_options applies.
_CUDA_BACKEND = "backends/cuda/backend.cpp"


def _cuda_env_aliases(src: str) -> list[tuple[str, str]]:
    m = re.search(r"\bkEnvAliases\[\]\s*=\s*\{(.*?)\n\};", src, re.S)
    return re.findall(r'\{"(FLYX_[A-Z0-9_]+)",\s*"([a-z0-9_]+)"', m.group(1)) if m else []


def _check_raw_env(root: Path, bad: list[str]) -> None:
    # getenv takes the name first, _dupenv_s takes it after the output buffer and length.
    read = re.compile(r'\b(?:getenv\s*\(|_dupenv_s\s*\([^,;]*,[^,;]*,)\s*"(FLYX_[A-Za-z0-9_]+)"')
    # A name the code computes would hide a read from the pattern above: models and backends spell out every name
    # they read, but for the CUDA backend's kEnvAliases (core's readers take their names from core/os/env.h).
    computed = re.compile(r"\bgetenv\s*\(\s*(?![\s\"])")
    for layer in ("core", "models", "backends", "apps"):
        for path in (root / layer).rglob("*"):
            if path.suffix not in {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cu", ".cuh", ".inc", ".inl"}:
                continue
            rel = path.relative_to(root).as_posix()
            if rel == "core/os/env.h":
                continue
            src = path.read_text(encoding="utf-8")
            for match in read.finditer(src):
                line = src.count("\n", 0, match.start()) + 1
                fix = ("make it a backend option (FLYX_BACKEND_OPTIONS, --backend-opt)" if layer == "backends"
                       else "register and resolve it through core/os/env.h")
                bad.append(f"{rel}:{line}: raw read of {match[1]}; {fix}")
            if layer in ("models", "backends") and rel != _CUDA_BACKEND:
                for match in computed.finditer(src):
                    line = src.count("\n", 0, match.start()) + 1
                    bad.append(f"{rel}:{line}: getenv of a computed name; spell the variable out")


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
        is_readme = rel == "README.md" or bool(re.fullmatch(r"docs/i18n/README_[a-z]+\.md", rel))
        cjk_text = re.sub(r'<a href="(?:docs/i18n/)?README_[a-z]+\.md">[^<]+</a>', "", text) if is_readme else text
        for label, pat in leak_patterns:
            if label == "CJK text" and rel in ("docs/i18n/README_zh.md", "docs/i18n/README_ja.md"):
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
        expected = {"README.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md"}
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

    cuda_src = _read(os.path.join(root, _CUDA_BACKEND))
    keys: set[str] = set()
    m_specs = re.search(r"const std::vector<OptionSpec>& option_specs\(\)(.*?)return specs;", cuda_src, re.S)
    if not m_specs:
        bad.append(f"{_CUDA_BACKEND}: failed to locate option_specs()")
    else:
        body = m_specs.group(1)
        keys = set(re.findall(r'(?:_opt\(|^\s*\{)"([a-z0-9_]+)"', body, re.M))
        if not keys:
            bad.append(f"{_CUDA_BACKEND}: failed to extract any CUDA backend options")
        for key in sorted(keys):
            if f"`{key}`" not in server_doc:
                bad.append(f"docs/server.md: missing CUDA backend option `{key}` from {_CUDA_BACKEND}")
    aliases = _cuda_env_aliases(cuda_src)
    if not aliases:
        bad.append(f"{_CUDA_BACKEND}: failed to locate the options' environment spellings (kEnvAliases)")

    env_src = _read(os.path.join(root, "core/os/env.h"))
    env_vars = set(re.findall(r'"((?:FLYX|HF)_[A-Z0-9_]+)"', env_src))
    if not env_vars:
        bad.append("core/os/env.h: failed to extract any FLYX_*/HF_* environment variables")
    _check_raw_env(Path(root), bad)
    env_vars.update(env for env, _ in aliases)
    env_rows: dict[str, str] = {}
    for row in re.findall(r"^\|[^\n]*", server_doc, re.M):
        for var in re.findall(r"`((?:FLYX|HF)_[A-Z0-9_]+)`", row.split("|")[1]):
            env_rows.setdefault(var, row)
    for var in sorted(env_vars - env_rows.keys()):
        bad.append(f"docs/server.md: missing environment variable row for `{var}`")
    for env, option in aliases:
        if env != "FLYX_" + option.upper():
            bad.append(f"{_CUDA_BACKEND}: kEnvAliases spells option `{option}` `{env}`, not `FLYX_{option.upper()}`")
        if keys and option not in keys:
            bad.append(f"{_CUDA_BACKEND}: kEnvAliases gives `{env}` to `{option}`, which is no backend option")
        if env in env_rows and f"`{option}`" not in env_rows[env]:
            bad.append(f"docs/server.md: the `{env}` row must name the backend option it sets, `{option}`")

    # Every FLYX_* setting the table lists is cleared for the tests (test preset test-base, which tests/CMakeLists.txt
    # also applies to a plain ctest): a shell's FLYX_WARM=0 or FLYX_LIB would otherwise reach every test run in it.
    try:
        presets = json.loads(_read(os.path.join(root, "CMakePresets.json")))
        base = next(p for p in presets.get("testPresets", []) if p.get("name") == "test-base")
        cleared = {k for k, v in base.get("environment", {}).items() if v == ""}
    except (ValueError, StopIteration) as e:
        bad.append(f"CMakePresets.json: no test preset test-base to read ({e!r})")
        cleared = None
    if cleared is not None:
        for var in sorted(v for v in env_rows if v.startswith("FLYX_") and v not in cleared):
            bad.append(f'CMakePresets.json: test preset test-base must clear `{var}` (docs/server.md lists it): "{var}": ""')

    for b in bad:
        print("FAIL", b)
    print("docs: " + ("ok" if not bad else f"{len(bad)} problems"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..")))
