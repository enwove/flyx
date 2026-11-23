# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""tools/codegen/gen_gemm_tables.py: a GPU's table carries rules of its own for any family the defaults have, a
table that inherits another takes the families it lacks from it, and tables that break the rules are refused.

    python tests/tools/codegen/gen_gemm_tables_test.py SOURCE_DIR
"""
import contextlib
import importlib.util
import io
import os
import re
import sys
import tempfile

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../.."))
spec = importlib.util.spec_from_file_location("gen_gemm_tables", os.path.join(ROOT, "tools/codegen/gen_gemm_tables.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
CANDIDATES = os.path.join(ROOT, "backends/cuda/variants/candidates.toml")
INF = 2**31 - 1
failures = []

# A GPU with rules of its own for rounds and for the runs of K, and a second GPU of the same chip that inherits them.
OWN = """
sku = "a"
devices = ["GPU A"]
[rules]
blocks_round = [
  { types = ["Q4_K"], rows = [7, 8], tensor_cores = "rows" },
  { rows = [1, 1], columns = 1, ahead = 2 },
  { rows = [2, 2], columns = 2, ahead = 0 },
  { rows = [3, 4], columns = 1, ahead = 0 },
  { rows = [5, 8], columns = 2, ahead = 0 },
  { rows = [9, 16], tensor_cores = "rows" },
]
blocks_q8_runs = [{ n_max = 5120, runs = 2 }, { runs = 1 }]
"""
INHERITS = """
sku = "b"
devices = ["GPU B"]
inherit = "a"
"""


def generate(tables):
    """The generated header for these tables (name -> TOML text), or the generator's refusal."""
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for name, text in tables.items():
            paths.append(os.path.join(d, name + ".toml"))
            open(paths[-1], "w", encoding="utf-8").write(text)
        out = os.path.join(d, "gemm_tables.inc")
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                gen.main(["--candidates", CANDIDATES, "--out", out] + paths)
        except SystemExit as e:
            return None, str(e)
        return open(out, encoding="utf-8").read(), None


def entries(text, array):
    body = re.search(r"inline (?:const|constexpr) \w+ %s\[\] = \{(.*?)\};" % array, text, re.S).group(1)
    return [line.strip().rstrip(",") for line in body.strip().splitlines()]


def check(what, ok, detail=""):
    print(("ok   " if ok else "FAIL ") + what + (f": {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(what)


text, err = generate({"a": OWN, "b": INHERITS})
check("a table with rules of its own generates", text is not None, err)
if text:
    skus = entries(text, "kSkus")
    round_bit, runs_bit = 1 << list(gen.FAMILIES).index("blocks_round"), 1 << list(gen.FAMILIES).index("blocks_q8_runs")
    check("the GPU's table says which families it has",
          skus[0] == f'{{"a", kDevices0, 1, -1, {round_bit | runs_bit}u, false}}', skus[0])
    check("the GPU that inherits has none of its own and names its parent",
          skus[1] == '{"b", kDevices1, 1, 0, 0u, false}', skus[1])
    rounds = entries(text, "kBlocksRound")
    own = [r for r in rounds if r.startswith("{0,")]
    check("the GPU's round rules, its type's exception first", own[:2] == [
        f"{{0, FLYX_DT_GGML_Q4_K, 7, 8, 0, {INF}, -1, 1}}", f"{{0, -1, 1, 1, 0, {INF}, 0, 0}}"], own[:2])
    check("every rule of the GPU's own, none of another's", len(own) == 6 and not any(r.startswith("{1,") for r in rounds))
    base, _ = generate({})
    check("the defaults stay as they are, after the GPUs' rules",
          [r for r in rounds if r.startswith("{-1,")] == entries(base, "kBlocksRound"))
    check("the GPU's runs of K", entries(text, "kBlocksQ8Runs")[:2] == [f"{{0, 5120, {INF}, 2}}", f"{{0, {INF}, {INF}, 1}}"],
          entries(text, "kBlocksQ8Runs")[:2])
    check("the families it lacks come from the defaults alone",
          all(r.startswith("{-1,") for r in entries(text, "kBlocksQ8") + entries(text, "kBlocks")))

override = '{ types = ["IQ3_S"], n_min = 5119, n_max = 5120, rows = [4, 4], columns = 1, ahead = 0 }'
text, err = generate({"a": f'sku = "a"\n[rules]\nblocks_round = [{override}, {{ defaults = true }}]\n'})
check("a measured round configuration can precede the default family", text is not None, err)
if text:
    own = [r for r in entries(text, "kBlocksRound") if r.startswith("{0,")]
    defaults = [r.replace("{-1,", "{0,", 1) for r in entries(text, "kBlocksRound") if r.startswith("{-1,")]
    check("the exact-width exception precedes all defaults in their original order",
          own[0].startswith("{0, FLYX_DT_GGML_IQ3_S, 4, 4, 5119, 5120,") and own[1:] == defaults)

FLASH = """
[[flash_gqa]]
fp8 = true
nh = 24
nkv = 4
hd = 256
rows = 128
keys = 1024
hb = 2
"""
text, err = generate({"a": OWN + FLASH, "b": INHERITS})
check("a staged FP8 attention rule names one SKU and one shape",
      text and entries(text, "kFlashGqa") == ["{0, 24, 4, 256, 128, 1024, 2}"], err)

PLACEMENT = """
[[strata_placement]]
copies = 16
host = 12
bits = 2
nkv = 4
hd = 256
min_capacity = 311296
edge_host = 2
scan_halves = true
"""
text, err = generate({"a": OWN + PLACEMENT, "b": INHERITS})
check("a host-copy placement names exactly the measured SKU and layout",
      text and entries(text, "kStrataPlacement") == ["{0, 16, 12, 2, 4, 256, 311296, 2, 1}"], err)

# A configuration with the rows staged in shared memory (and its warps) is named like any other.
STAGED = """
sku = "a"
[rules]
blocks_round = [
  { rows = [1, 1], columns = 1, ahead = 2 },
  { rows = [2, 2], columns = 4, ahead = 0, warps = 16, staged = 1 },
  { rows = [3, 4], columns = 1, ahead = 0 },
  { rows = [5, 8], columns = 2, ahead = 0 },
  { rows = [9, 16], tensor_cores = "rows" },
]
"""
text, err = generate({"a": STAGED})
check("a table naming a staged configuration generates", text is not None, err)
if text:
    compiled = re.search(r"X\((\d+), 2, 4, 0, 16, 1, 1\)", text)
    rule = [r for r in entries(text, "kBlocksRound") if r.startswith("{0, -1, 2, 2,")]
    check("its rule runs that configuration", bool(compiled and rule and rule[0].endswith(f", {compiled.group(1)}, 0}}")),
          rule)

# The tensor cores with the streams split among blocks: kernel 3, no dp4a configuration.
text, err = generate({"a": STAGED.replace('rows = [9, 16], tensor_cores = "rows"', 'rows = [9, 16], tensor_cores = "split"')})
check("a table naming the split kernel generates", text is not None, err)
if text:
    rule = [r for r in entries(text, "kBlocksRound") if r.startswith("{0, -1, 9, 16,")]
    check("its rule runs kernel 3", bool(rule and rule[0].endswith(", -1, 3}")), rule)

# Programmatic dependent launch by GPU: a table's own, else the one it inherits, else off.
text, err = generate({"a": 'sku = "a"\npdl = true\n', "b": 'sku = "b"\ninherit = "a"\n',
                      "c": 'sku = "c"\ninherit = "a"\npdl = false\n', "d": 'sku = "d"\n'})
check("tables that say pdl generate", text is not None, err)
if text:
    pdl = [s.rsplit(", ", 1)[1].rstrip("}") for s in entries(text, "kSkus")]
    check("pdl: the table's own, the inherited one, an inheriting table's own, none", pdl == ["true", "true", "false",
                                                                                               "false"], pdl)

# A table that also serves the GPUs of its architectures without one of their own.
text, err = generate({"a": 'sku = "a"\narchitectures = ["8.6", "8.0"]\n', "b": 'sku = "b"\narchitectures = ["12.0"]\n',
                      "c": 'sku = "c"\n'})
check("tables that serve architectures generate", text is not None, err)
if text:
    check("each compute capability names its table", entries(text, "kArchTables") == ["{86, 0}", "{80, 0}", "{120, 1}"],
          entries(text, "kArchTables"))
    base, _ = generate({"c": 'sku = "c"\n'})
    check("none: an empty list", entries(base, "kArchTables") == ["{-1, -1}"] and "kArchTableCount = 0" in base)

# A pruned release build compiles the kernels its tables reach (kReach*, by architecture): a rule of 17..64 rows names
# a bf16_mt configuration (k_gemm_skinny_mt, kReachBf16Mt), never a 1..16-row kernel's.
MT_SHAPE = ('[[gemm]]\nformat = "bf16_rows"\nn = 1536\nk = 4608\n'
            'rules = [{ rows = [17, 64], nt = 4, chunks = 16, unroll = 2, parts = 3 }]\n')
plain_t, _ = generate({"a": 'sku = "a"\narchitectures = ["12.0"]\n'})
mt_t, err = generate({"a": 'sku = "a"\narchitectures = ["12.0"]\n' + MT_SHAPE})
check("a table with a rule of 17..64 rows generates", mt_t is not None, err)
if plain_t and mt_t:
    reach = lambda t: [l for l in t.split("\n")
                       if l.startswith("inline constexpr unsigned kReach") and "Bf16Mt" not in l]
    check("its rule reaches no 1..16-row kernel", reach(mt_t) == reach(plain_t),
          [a for a, b in zip(reach(mt_t), reach(plain_t)) if a != b])
    mt_reach = lambda t: [int(v.rstrip("u"), 16) for v in
                          re.search(r"kReachBf16Mt\[7\] = \{(.*?)\};", t).group(1).split(", ")]
    cfg = int(re.search(r"X\((\d+), 4, 16, 2, 3\)", mt_t).group(1))
    defaults = mt_reach(plain_t)
    check("its bf16_mt configuration is reached on its architecture (sm_120) and in a build for any",
          mt_reach(mt_t) == [d | (1 << cfg if b >= 5 else 0) for b, d in enumerate(defaults)],
          [hex(v) for v in mt_reach(mt_t)])
    check("every architecture reaches the defaults' bf16_mt configurations", len(set(defaults[:6])) == 1 and
          defaults[0] != 0 and defaults[6] == defaults[0], [hex(v) for v in defaults])

# A drafting pass's K partition is K's alone (draft_chunks), in every table: a bf16_draft rule names the tile and the
# unroll, and serves a shape in its K's chunks (the shapes no table lists: in either partition, by K at run time).
DRAFT = """
sku = "a"
[[gemm]]
format = "bf16_draft"
n = 1536
k = 4608
rules = [{ rows = [17, 32], nt = 1, unroll = 4 }, { rows = [33, 64], nt = 2, unroll = 4 }]
[[gemm]]
format = "bf16_draft"
n = 2560
k = 1536
rules = [{ rows = [17, 64], nt = 2, unroll = 4 }]
[rules]
bf16_draft = [{ rows = [17, 48], nt = 1, unroll = 4 }, { rows = [49, 64], nt = 4, unroll = 4 }]
"""
text, err = generate({"a": DRAFT})
check("a table with drafting rules generates", text is not None, err)
if text:
    listed = text.split("#define FLYX_BF16_MT(X)")[1].split("#define FLYX_BF16_MT_COUNT")[0]
    mt = {tuple(int(v) for v in m.groups()[1:]): int(m.group(1))
          for m in re.finditer(r"X\((\d+), (\d+), (\d+), (\d+), (\d+)\) \\", listed)}
    check("one part compiles only a drafting pass's partitions",
          {ch for (_, ch, _, parts) in mt if parts == 1} == set(gen.DRAFT_CHUNKS), sorted(mt))
    check("the partition by K: 8 chunks, 16 from K = 4096",
          [gen.draft_chunks(k) for k in (96, 256, 1536, 2048, 4095, 4096, 4608, 9728)] == [8] * 5 + [16] * 3)
    check("the header says the same", f"kDraftLongK = {gen.DRAFT_LONG_K}, kDraftChunks = 8, kDraftChunksLong = 16"
          in text)
    rules = entries(text, "kRules")
    want = [f"{{0, 4, 1536, 4608, 17, 32, {mt[(1, 16, 4, 1)]}}}", f"{{0, 4, 1536, 4608, 33, 64, {mt[(2, 16, 4, 1)]}}}",
            f"{{0, 4, 2560, 1536, 17, 64, {mt[(2, 8, 4, 1)]}}}"]
    check("a shape's rules run its K's chunks (K 4608: 16, K 1536: 8)", rules == want, rules)
    family = [r for r in entries(text, "kBf16DraftRules") if r.startswith("{0,")]
    check("the shapes no table lists: each rule's tile and unroll in both partitions",
          family == [f"{{0, 17, 48, {mt[(1, 8, 4, 1)]}, {mt[(1, 16, 4, 1)]}}}",
                     f"{{0, 49, 64, {mt[(4, 8, 4, 1)]}, {mt[(4, 16, 4, 1)]}}}"], family)
with open(CANDIDATES, encoding="utf-8") as f:
    four = f.read().replace("[[nvfp4_skinny]]", "[[bf16_mt]]\nnt = 1\nchunks = 4\nunroll = 4\nparts = 1\n\n"
                                                "[[nvfp4_skinny]]", 1)
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "candidates.toml")
    open(path, "w", encoding="utf-8").write(four)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            gen.main(["--candidates", path, "--out", os.path.join(d, "out.inc")])
        err = None
    except SystemExit as e:
        err = str(e)
check("refused: a one-part configuration in a partition no K is given",
      err is not None and "parts 1 with chunks 8 or 16" in err, err)

refusals = {
    "host-copy edges wider than the host copies": (PLACEMENT.replace("edge_host = 2", "edge_host = 7"), "invalid strata_placement"),
    "two placements for one strata layout": (PLACEMENT + PLACEMENT, "duplicate strata_placement"),
    "nonboolean half scan": (PLACEMENT.replace("scan_halves = true", "scan_halves = 2"), "scan_halves takes"),
    "round defaults before an exception": (f"[rules]\nblocks_round = [{{ defaults = true }}, {override}]\n", "must end"),
    "a false default marker": ("[rules]\nblocks_round = [{ defaults = false }]\n", "must end"),
    "a grouped attention configuration that is not compiled": (FLASH.replace("hb = 2", "hb = 4"), "not compiled"),
    "two grouped attention winners for one shape": (FLASH + FLASH.replace("hb = 2", "hb = 6"), "duplicate flash_gqa"),
    "pdl that is not true or false": ('pdl = "yes"\n', "pdl takes true or false"),
    "a drafting rule that names its chunks": (
        "[rules]\nbf16_draft = [{ rows = [17, 64], nt = 1, chunks = 4, unroll = 4, parts = 1 }]\n",
        "names rows, nt and unroll alone"),
    "a drafting rule for shapes no table lists compiled in one partition only": (
        "[rules]\nbf16_draft = [{ rows = [17, 64], nt = 2, unroll = 8 }]\n", "both partitions"),
    "a drafting rule for a shape whose K's partition is not compiled in its tile and unroll": (
        '[[gemm]]\nformat = "bf16_draft"\nn = 1536\nk = 4608\nrules = [{ rows = [17, 64], nt = 2, unroll = 8 }]\n',
        "no compiled bf16_mt configuration"),
    "drafting rules that leave rows uncovered": ("[rules]\nbf16_draft = [{ rows = [17, 48], nt = 1, unroll = 4 }]\n",
                                                 "cover rows 17..64 once"),
    "an architecture that is no compute capability": ('architectures = ["sm_89"]\n', "compute capabilities like"),
    "a family that leaves rows uncovered": ("[rules]\nblocks_round = [{ rows = [1, 1], columns = 1, ahead = 2 }]\n",
                                            "takes 2 rows"),
    "a staged configuration that is not compiled": (
        "[rules]\nblocks_round = [{ rows = [2, 2], columns = 1, ahead = 0, staged = 1 }]\n", "is not compiled"),
    "half of a family": ("[rules]\nnvfp4_k128 = { span = 128, chunks = 4, nt = 1, warps = 4, unroll = 4 }\n",
                         "take nvfp4_k128, nvfp4_k64"),
    "a family that does not exist": ("[rules]\nblocks_rounds = []\n", "unknown rules"),
    "a configuration that is not compiled": ("[rules]\nblocks = [{ rows = [1, 16], nt = 3, nw = 1, kw = 1 }]\n",
                                             "compiled configuration"),
    "a K partition per shape broken": ("[rules]\nblocks_q8_runs = [{ runs = 1 }, { n_max = 5120, runs = 2 }]\n",
                                       "rising n_max"),
}
for what, (rules, says) in refusals.items():
    _, err = generate({"a": 'sku = "a"\n' + rules})
    check("refused: " + what, err is not None and says in err, err)
_, err = generate({"a": 'sku = "a"\ninherit = "b"\n', "b": 'sku = "b"\ninherit = "a"\n'})
check("refused: tables that inherit each other", err is not None and "goes round" in err, err)
_, err = generate({"a": 'sku = "a"\n', "b": 'sku = "a"\n'})
check("refused: two tables of one sku", err is not None and "two tables" in err, err)
_, err = generate({"a": 'sku = "a"\narchitectures = ["8.9"]\n', "b": 'sku = "b"\narchitectures = ["8.9"]\n'})
check("refused: two tables of one architecture", err is not None and "already serves" in err, err)

print(f"{len(failures)} failed" if failures else "all passed")
sys.exit(1 if failures else 0)
