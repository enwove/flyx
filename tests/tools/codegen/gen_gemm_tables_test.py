# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""tools/codegen/gen_gemm_tables.py: a GPU's table carries rules of its own for any family the defaults have, a
table that inherits another takes the families it lacks from it, and tables that break the rules are refused.

    python tests/tools/codegen/gen_gemm_tables_test.py SOURCE_DIR
"""
import contextlib
import glob
import importlib.util
import io
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from check import check, finish  # noqa: E402

ROOT =os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "../../.."))
spec = importlib.util.spec_from_file_location("gen_gemm_tables", os.path.join(ROOT, "tools/codegen/gen_gemm_tables.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
CANDIDATES = os.path.join(ROOT, "backends/cuda/variants/candidates.toml")
INF = 2**31 - 1

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


text, err = generate({"a": OWN, "b": INHERITS})
check(text is not None, "a table with rules of its own generates", err)
if text:
    skus = entries(text, "kSkus")
    round_bit, runs_bit = 1 << list(gen.FAMILIES).index("blocks_round"), 1 << list(gen.FAMILIES).index("blocks_q8_runs")
    check(skus[0] == f'{{"a", kDevices0, 1, -1, {round_bit | runs_bit}u, false, 0}}',
          "the GPU's table says which families it has", skus[0])
    check(skus[1] == '{"b", kDevices1, 1, 0, 0u, false, 0}',
          "the GPU that inherits has none of its own and names its parent", skus[1])
    rounds = entries(text, "kBlocksRound")
    own = [r for r in rounds if r.startswith("{0,")]
    check(own[:2] == [f"{{0, FLYX_DT_GGML_Q4_K, 7, 8, 0, {INF}, -1, 1}}", f"{{0, -1, 1, 1, 0, {INF}, 0, 0}}"],
          "the GPU's round rules, its type's exception first", own[:2])
    check(len(own) == 6 and not any(r.startswith("{1,") for r in rounds),
          "every rule of the GPU's own, none of another's")
    base, _ = generate({})
    check([r for r in rounds if r.startswith("{-1,")] == entries(base, "kBlocksRound"),
          "the defaults stay as they are, after the GPUs' rules")
    check(entries(text, "kBlocksQ8Runs")[:2] == [f"{{0, 5120, {INF}, 2}}", f"{{0, {INF}, {INF}, 1}}"],
          "the GPU's runs of K", entries(text, "kBlocksQ8Runs")[:2])
    check(all(r.startswith("{-1,") for r in entries(text, "kBlocksQ8") + entries(text, "kBlocks")),
          "the families it lacks come from the defaults alone")

override = '{ types = ["IQ3_S"], n_min = 5119, n_max = 5120, rows = [4, 4], columns = 1, ahead = 0 }'
text, err = generate({"a": f'sku = "a"\n[rules]\nblocks_round = [{override}, {{ defaults = true }}]\n'})
check(text is not None, "a measured round configuration can precede the default family", err)
if text:
    own = [r for r in entries(text, "kBlocksRound") if r.startswith("{0,")]
    defaults = [r.replace("{-1,", "{0,", 1) for r in entries(text, "kBlocksRound") if r.startswith("{-1,")]
    check(own[0].startswith("{0, FLYX_DT_GGML_IQ3_S, 4, 4, 5119, 5120,") and own[1:] == defaults,
          "the exact-width exception precedes all defaults in their original order")

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
check(text and entries(text, "kFlashGqa") == ["{0, 24, 4, 256, 128, 1024, 2}"],
      "a staged FP8 attention rule names one SKU and one shape", err)

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
check(text and entries(text, "kStrataPlacement") == ["{0, 16, 12, 2, 4, 256, 311296, 2, 1}"],
      "a host-copy placement names exactly the measured SKU and layout", err)

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
check(text is not None, "a table naming a staged configuration generates", err)
if text:
    compiled = re.search(r"X\((\d+), 2, 4, 0, 16, 1, 1\)", text)
    rule = [r for r in entries(text, "kBlocksRound") if r.startswith("{0, -1, 2, 2,")]
    check(bool(compiled and rule and rule[0].endswith(f", {compiled.group(1)}, 0}}")),
          "its rule runs that configuration", rule)

# The tensor cores with the streams split among blocks: kernel 3, no dp4a configuration.
text, err = generate({"a": STAGED.replace('rows = [9, 16], tensor_cores = "rows"', 'rows = [9, 16], tensor_cores = "split"')})
check(text is not None, "a table naming the split kernel generates", err)
if text:
    rule = [r for r in entries(text, "kBlocksRound") if r.startswith("{0, -1, 9, 16,")]
    check(bool(rule and rule[0].endswith(", -1, 3}")), "its rule runs kernel 3", rule)

# Programmatic dependent launch by GPU: a table's own, else the one it inherits, else off.
text, err = generate({"a": 'sku = "a"\npdl = true\n', "b": 'sku = "b"\ninherit = "a"\n',
                      "c": 'sku = "c"\ninherit = "a"\npdl = false\n', "d": 'sku = "d"\n'})
check(text is not None, "tables that say pdl generate", err)
if text:
    pdl = [s.rsplit(", ", 2)[1] for s in entries(text, "kSkus")]
    check(pdl == ["true", "true", "false", "false"],
          "pdl: the table's own, the inherited one, an inheriting table's own, none", pdl)

# A table that also serves the GPUs of its architectures without one of their own.
text, err = generate({"a": 'sku = "a"\narchitectures = ["8.6", "8.0"]\n', "b": 'sku = "b"\narchitectures = ["12.0"]\n',
                      "c": 'sku = "c"\n'})
check(text is not None, "tables that serve architectures generate", err)
if text:
    check(entries(text, "kArchTables") == ["{86, 0}", "{80, 0}", "{120, 1}"],
          "each compute capability names its table", entries(text, "kArchTables"))
    base, _ = generate({"c": 'sku = "c"\n'})
    check(entries(base, "kArchTables") == ["{-1, -1}"] and "kArchTableCount = 0" in base, "none: an empty list")

# A pruned release build compiles the kernels its tables reach (kReach*, by architecture): a rule of 17..64 rows names
# a bf16_mt configuration (k_gemm_skinny_mt, kReachBf16Mt), never a 1..16-row kernel's.
MT_SHAPE = ('[[gemm]]\nformat = "bf16_rows"\nn = 1536\nk = 4608\n'
            'rules = [{ rows = [17, 64], nt = 4, chunks = 16, unroll = 2, parts = 3 }]\n')
plain_t, _ = generate({"a": 'sku = "a"\narchitectures = ["12.0"]\n'})
mt_t, err = generate({"a": 'sku = "a"\narchitectures = ["12.0"]\n' + MT_SHAPE})
check(mt_t is not None, "a table with a rule of 17..64 rows generates", err)
if plain_t and mt_t:
    reach = lambda t: [l for l in t.split("\n")
                       if l.startswith("inline constexpr unsigned kReach") and "Bf16Mt" not in l]
    check(reach(mt_t) == reach(plain_t), "its rule reaches no 1..16-row kernel",
          [a for a, b in zip(reach(mt_t), reach(plain_t)) if a != b])
    mt_reach = lambda t: [int(v.rstrip("u"), 16) for v in
                          re.search(r"kReachBf16Mt\[7\] = \{(.*?)\};", t).group(1).split(", ")]
    cfg = int(re.search(r"X\((\d+), 4, 16, 2, 3\)", mt_t).group(1))
    defaults = mt_reach(plain_t)
    check(mt_reach(mt_t) == [d | (1 << cfg if b >= 5 else 0) for b, d in enumerate(defaults)],
          "its bf16_mt configuration is reached on its architecture (sm_120) and in a build for any",
          [hex(v) for v in mt_reach(mt_t)])
    check(len(set(defaults[:6])) == 1 and defaults[0] != 0 and defaults[6] == defaults[0],
          "every architecture reaches the defaults' bf16_mt configurations", [hex(v) for v in defaults])

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
check(text is not None, "a table with drafting rules generates", err)
if text:
    listed = text.split("#define FLYX_BF16_MT(X)")[1].split("#define FLYX_BF16_MT_COUNT")[0]
    mt = {tuple(int(v) for v in m.groups()[1:]): int(m.group(1))
          for m in re.finditer(r"X\((\d+), (\d+), (\d+), (\d+), (\d+)\) \\", listed)}
    check({ch for (_, ch, _, parts) in mt if parts == 1} == set(gen.DRAFT_CHUNKS),
          "one part compiles only a drafting pass's partitions", sorted(mt))
    check([gen.draft_chunks(k) for k in (96, 256, 1536, 2048, 4095, 4096, 4608, 9728)] == [8] * 5 + [16] * 3,
          "the partition by K: 8 chunks, 16 from K = 4096")
    check(f"kDraftLongK = {gen.DRAFT_LONG_K}, kDraftChunks = 8, kDraftChunksLong = 16" in text,
          "the header says the same")
    rules = entries(text, "kRules")
    want = [f"{{0, 4, 1536, 4608, 17, 32, {mt[(1, 16, 4, 1)]}}}", f"{{0, 4, 1536, 4608, 33, 64, {mt[(2, 16, 4, 1)]}}}",
            f"{{0, 4, 2560, 1536, 17, 64, {mt[(2, 8, 4, 1)]}}}"]
    check(rules == want, "a shape's rules run its K's chunks (K 4608: 16, K 1536: 8)", rules)
    family = [r for r in entries(text, "kBf16DraftRules") if r.startswith("{0,")]
    check(family == [f"{{0, 17, 48, {mt[(1, 8, 4, 1)]}, {mt[(1, 16, 4, 1)]}}}",
                     f"{{0, 49, 64, {mt[(4, 8, 4, 1)]}, {mt[(4, 16, 4, 1)]}}}"],
          "the shapes no table lists: each rule's tile and unroll in both partitions", family)
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
check(err is not None and "parts 1 with chunks 8 or 16" in err,
      "refused: a one-part configuration in a partition no K is given", err)

# MoE layers' fused gate|up GEMMs (moe_gate_up): a GPU's own rules by type and rows, the original tiles for every other
# type; a release build compiles what they reach for that type, and the original tiles for every type everywhere.
MOE = """
sku = "a"
[rules]
moe_gate_up = [
  { types = ["IQ2_S"], rows = [1, 3], columns = 4, units = 2 },
  { types = ["IQ2_S"], columns = 1, units = 2 },
  { columns = 0, units = 0 },
]
"""
text, err = generate({"a": MOE})
base, _ = generate({})
check(text is not None, "a table with MoE gate|up rules of its own generates", err)
if text and base:
    listed = text.split("#define FLYX_MOE_DP4A(X)")[1].split("#define FLYX_MOE_DP4A_COUNT")[0]
    moe = {(int(c), int(u)): int(i) for i, c, u in re.findall(r"X\((\d+), (\d+), (\d+)\)", listed)}
    original = moe.get((0, 0))
    check(original is not None and f"kMoeDp4aOriginal = {original};" in text,
          "the original tiles are a configuration, named as such", sorted(moe))
    own = [r for r in entries(text, "kMoeGateUp") if r.startswith("{0,")]
    check(own == [f"{{0, FLYX_DT_GGML_IQ2_S, 1, 3, {moe[(4, 2)]}}}",
                  f"{{0, FLYX_DT_GGML_IQ2_S, 1, {INF}, {moe[(1, 2)]}}}", f"{{0, -1, 1, {INF}, {original}}}"],
          "the GPU's MoE rules: its type's in order, then the original tiles for every type", own)
    moe_bit = 1 << list(gen.FAMILIES).index("moe_gate_up")
    check(entries(text, "kSkus")[0] == f'{{"a", kDevices0, 0, -1, {moe_bit}u, false, 0}}',
          "the GPU's table says it has the family", entries(text, "kSkus")[0])
    check(entries(base, "kMoeGateUp")[-1] == f"{{-1, -1, 1, {INF}, {original}}}",
          "the defaults end with the original tiles for every type", entries(base, "kMoeGateUp")[-1])

    def moe_reach(t):
        body = re.search(r"kReachMoeDp4a\[7\]\[2\]\[21\] = \{(.*?)\n\};", t, re.S).group(1)
        rows = [[int(v.rstrip("u"), 16) for v in m.split(", ")] for m in re.findall(r"\{([0-9a-fxu, ]+)\}", body)]
        assert len(rows) == 14
        return [[rows[2 * i], rows[2 * i + 1]] for i in range(7)]
    r, b = moe_reach(text), moe_reach(base)
    s = gen.GGML_TYPES.index("IQ2_S")
    # rows = [1, 3] hits both rows == 1 (band 0) and rows >= 2 (band 1); the fallback (1x2) is shadowed at rows == 1.
    add_r1 = 1 << moe[(4, 2)]
    add_r2 = (1 << moe[(4, 2)]) | (1 << moe[(1, 2)])
    check(all(r[i][0][s] == (b[i][0][s] | add_r1) and r[i][1][s] == (b[i][1][s] | add_r2) for i in range(7)),
          "a release build compiles the table's configurations per row band for its type on every architecture",
          [(hex(row[0][s]), hex(row[1][s])) for row in r])
    check(all(r[i][band][d] == b[i][band][d] for i in range(7) for band in (0, 1)
              for d in range(len(gen.GGML_TYPES)) if d != s),
          "... and nothing more for the other types")
    check(all(v & (1 << original) for row in b for band in row for v in band),
          "every type reaches the original tiles in both row bands in every build")
    allowed = sum(1 << i for i in moe.values())
    check(all(b[i][band][d] == 1 << original for i in range(7) for band in (0, 1)
              for d, t in enumerate(gen.GGML_TYPES) if t not in gen.MOE_DP4A_TYPES)
          and all(v & ~allowed == 0 for row in b for band in row for v in band),
          "the types without configurations of their own reach the original tiles alone")
    arch_t, arch_err = generate({"a": 'architectures = ["12.0"]\n' + MOE})
    check(arch_t is not None, "an architecture-scoped MoE gate|up table generates", arch_err)
    if arch_t:
        ar = moe_reach(arch_t)
        check(all(ar[i][0][s] == (b[i][0][s] | (add_r1 if i >= 5 else 0)) and
                  ar[i][1][s] == (b[i][1][s] | (add_r2 if i >= 5 else 0)) for i in range(7)),
              "architecture-scoped MoE rules add reachability only on sm_120 (bucket 5) and any (bucket 6)",
              [(hex(row[0][s]), hex(row[1][s])) for row in ar])
with open(CANDIDATES, encoding="utf-8") as f:
    no_original = f.read().replace("[[moe_dp4a]]\ncolumns = 0\nunits = 0\n", "", 1)
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "candidates.toml")
    open(path, "w", encoding="utf-8").write(no_original)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            gen.main(["--candidates", path, "--out", os.path.join(d, "out.inc")])
        err = None
    except SystemExit as e:
        err = str(e)
check(err is not None and "the original tiles" in err, "refused: MoE configurations without the original tiles", err)

MOE_REST = "{ columns = 0, units = 0 }"
# A split configuration in a shape's own rules: the scratch holds the chunks' sums of its columns, for the GPU and the
# one that inherits its table.
SPLIT_SHAPE = """
sku = "a"
devices = ["GPU A"]
[[gemm]]
format = "bf16"
n = 5120
k = 17408
rules = [{ rows = [1, 8], nt = 4, warps = 16, unroll = 4 }, { rows = [9, 16], nt = 4, warps = 4, unroll = 4, split = 4 }]
"""
text, err = generate({"a": SPLIT_SHAPE, "b": INHERITS})
check(text is not None, "a shape's split rule generates", err)
if text:
    cols = [e.rsplit(", ", 1)[1].rstrip("}") for e in entries(text, "kSkus")]
    check(cols == ["5120", "5120"], "the split columns: the table's widest split shape, inherited too", cols)
    base, _ = generate({})
    check(base is not None and entries(base, "kSkus") == ['{"none", nullptr, 0, -1, 0u, false, 0}'],
          "no table: no split columns", base and entries(base, "kSkus"))
TC_REST = "{ bm = 64, bn = 128, wn = 4, stages = 2 }"
# A GPU's own prefill rules: by the format and the rows, the defaults' tile counts for NVFP4.
TC_OWN = f"""
sku = "a"
devices = ["GPU A"]
[rules]
prefill_tc = [
  {{ format = "bf16", rows = [1, 128], bm = 64, bn = 32, wn = 2, stages = 6 }},
  {{ format = "nvfp4", sm_tiles_below = 0.5, bm = 64, bn = 32, wn = 2, stages = 2 }},
  {TC_REST},
]
"""
text, err = generate({"a": TC_OWN})
check(text is not None, "a table with prefill rules of its own generates", err)
if text:
    tc = [list(map(str.strip, e.strip("{}").split(","))) for e in entries(text, "kPrefillTc")]
    cfg = {tuple(map(int, m.groups()[1:])): int(m.group(1))
           for m in re.finditer(r"X\((\d+), (\d+), (\d+), (\d+), (\d+)\)", text)}
    own = [r for r in tc if r[0] == "0"]
    check(own == [["0", "0", "1", "128", "0.0", str(cfg[(64, 32, 2, 6)])],
                  ["0", "1", "1", str(INF), "0.5", str(cfg[(64, 32, 2, 2)])],
                  ["0", "-1", "1", str(INF), "0.0", str(cfg[(64, 128, 4, 2)])]],
          "the GPU's prefill rules in order, by format, rows and tiles per SM", own)
    check(any(r[0] == "-1" for r in tc), "the default prefill rules follow", tc)
refusals = {
    "a MoE rule for every type in a configuration the other types do not compile": (
        "[rules]\nmoe_gate_up = [{ columns = 2, units = 1 }]\n", "beside the original tiles"),
    "a MoE configuration for a type that compiles the original tiles alone": (
        f'[rules]\nmoe_gate_up = [{{ types = ["Q8_0"], columns = 2, units = 1 }}, {MOE_REST}]\n',
        "beside the original tiles"),
    "a MoE configuration that is not compiled": (
        f'[rules]\nmoe_gate_up = [{{ types = ["IQ2_S"], columns = 8, units = 1 }}, {MOE_REST}]\n',
        "compiled moe_dp4a configuration"),
    "MoE rules of a type that leave it rows to another list": (
        f'[rules]\nmoe_gate_up = [{{ types = ["IQ2_S"], rows = [1, 4], columns = 2, units = 1 }}, {MOE_REST}]\n',
        "end with a rule for the rest"),
    "MoE rules without the rule for every type": (
        '[rules]\nmoe_gate_up = [{ types = ["IQ2_S"], columns = 2, units = 1 }]\n', "end with a rule for the rest"),
    "a MoE rule naming a type after the rules for every type": (
        f'[rules]\nmoe_gate_up = [{MOE_REST}, {{ types = ["IQ2_S"], columns = 2, units = 1 }}]\n',
        "come before the others"),
    "a type in two lists of MoE rules": (
        f'[rules]\nmoe_gate_up = [{{ types = ["IQ2_S"], columns = 2, units = 1 }}, '
        f'{{ types = ["IQ2_S", "IQ2_XS"], columns = 1, units = 1 }}, {MOE_REST}]\n', "a type is named once"),
    "a MoE rule with a key it does not take": (
        "[rules]\nmoe_gate_up = [{ columns = 0, units = 0, n_max = 64 }]\n", "maybe rows, and columns and units"),
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
    "a split configuration for shapes no table lists": (
        "[rules]\nbf16 = [{ rows = [1, 16], nt = 4, warps = 4, unroll = 4, split = 4 }]\n", "of split 1"),
    "a split configuration for W12 weights": (
        '[[gemm]]\nformat = "w12"\nn = 5120\nk = 17408\nrules = [{ rows = [1, 16], nt = 4, warps = 4, unroll = 4, '
        'split = 4 }]\n', "serves bf16 weights"),
    "a split configuration that is not compiled": (
        '[[gemm]]\nformat = "bf16"\nn = 5120\nk = 17408\nrules = [{ rows = [1, 16], nt = 4, warps = 4, unroll = 4, '
        'split = 2 }]\n', "is not in candidates"),
    "a prefill configuration that is not compiled": (
        f"[rules]\nprefill_tc = [{{ bm = 64, bn = 16, wn = 1, stages = 3 }}, {TC_REST}]\n", "compiled configuration"),
    "prefill rules without a rule for the rest of a format": (
        '[rules]\nprefill_tc = [{ format = "bf16", bm = 64, bn = 128, wn = 4, stages = 2 }]\n',
        "with a rule for the rest"),
    "a prefill rule after the rules for the rest": (
        f'[rules]\nprefill_tc = [{TC_REST}, {{ format = "bf16", rows = [1, 64], bm = 64, bn = 32, wn = 2, stages = 6 }}]\n',
        "after the rules for the rest"),
    "a prefill rule of no format": (
        f'[rules]\nprefill_tc = [{{ format = "fp8", bm = 64, bn = 32, wn = 2, stages = 6 }}, {TC_REST}]\n',
        "takes a format"),
    "a K partition per shape broken": ("[rules]\nblocks_q8_runs = [{ runs = 1 }, { n_max = 5120, runs = 2 }]\n",
                                       "rising n_max"),
}
for what, (rules, says) in refusals.items():
    _, err = generate({"a": 'sku = "a"\n' + rules})
    check(err is not None and says in err, "refused: " + what, err)
_, err = generate({"a": 'sku = "a"\ninherit = "b"\n', "b": 'sku = "b"\ninherit = "a"\n'})
check(err is not None and "goes round" in err, "refused: tables that inherit each other", err)
_, err = generate({"a": 'sku = "a"\n', "b": 'sku = "a"\n'})
check(err is not None and "two tables" in err, "refused: two tables of one sku", err)
_, err = generate({"a": 'sku = "a"\narchitectures = ["8.9"]\n', "b": 'sku = "b"\narchitectures = ["8.9"]\n'})
check(err is not None and "already serves" in err, "refused: two tables of one architecture", err)
_, err = generate({"a": 'sku = "a"\ndevices = ["GPU A"]\n', "b": 'sku = "b"\ndevices = ["GPU B", "GPU A"]\n'})
check(err is not None and "already named" in err, "refused: two tables of one device name", err)

# The committed tables: they generate, each is variants/sm<cc>/tables/<sku>.toml, and each CUDA device name it gives
# picks it (kSkus, in the generator's order of the paths) and no other table.
committed = sorted(glob.glob(os.path.join(ROOT, "backends/cuda/variants/*/tables/*.toml")))
with tempfile.TemporaryDirectory() as d:
    out = os.path.join(d, "gemm_tables.inc")
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            gen.main(["--candidates", CANDIDATES, "--out", out] + committed)
        text, err = open(out, encoding="utf-8").read(), None
    except SystemExit as e:
        text, err = None, str(e)
check(text is not None, f"the {len(committed)} committed tables generate", err)
if text is not None:
    skus = [re.match(r'\{"([^"]+)", kDevices(\d+), (\d+),', e) for e in entries(text, "kSkus")]
    names = {}
    for path in committed:
        t = gen._load_toml(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        arch_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        check(t["sku"] == stem, f"{arch_dir}/tables/{stem}.toml: its sku is its name", t["sku"])
        check(re.fullmatch(r"sm\d+", arch_dir) is not None, f"{stem}: under an sm<cc> directory", arch_dir)
        check(bool(t.get("devices")) and all(n.startswith("NVIDIA ") for n in t["devices"]),
              f"{stem}: names a CUDA device", t.get("devices"))
        for n in t.get("devices", []):
            names.setdefault(n, []).append(t["sku"])
    for n, owners in sorted(names.items()):
        i = next((m for m in skus if m and m.group(1) == owners[0]), None)
        listed = i is not None and re.search(r"kDevices%s\[\] = \{[^}]*\"%s\"" % (i.group(2), re.escape(n)), text)
        check(len(owners) == 1 and bool(listed), f"{n!r} picks {owners[0]} alone", owners)

sys.exit(finish("gen_gemm_tables_test"))
