# Jinja engine from llama.cpp

The chat-template engine flyx renders model templates with. It is
llama.cpp's own Jinja implementation (introduced in llama.cpp PR #18462),
copied (with the differences below) from

    https://github.com/ggml-org/llama.cpp  commit f7591a6 (2026-11-07)

under the MIT license (`LICENSE`, "Copyright (c) 2023-2026 The ggml authors").

| here | llama.cpp |
|---|---|
| `jinja/` | `common/jinja/` (lexer, parser, runtime, values, input marking, template capabilities) |
| `json.h`, `json.cpp` | `common/json.h`, `common/json.cpp` (`common_json`, a thin layer over nlohmann/json) |
| `unicode.h`, `unicode.cpp` | `common/unicode.h`, `common/unicode.cpp` (UTF-8 helpers) |
| `LICENSE` | `LICENSE` |

nlohmann/json is ours (`third_party/nlohmann`). Differences from upstream:

* `ggml.h` is added: a shim that defines `GGML_ASSERT`, the one thing
  `json.cpp` takes from ggml.
* `jinja/string.cpp`: `strip()` without a character set (the `trim` filter,
  `.strip()`, `.lstrip()`, `.rstrip()`) removes the whitespace Python's
  `str.isspace()` accepts, Unicode spaces such as U+3000 included; upstream
  removes ASCII whitespace only.  transformers renders with Python's rules,
  so without this a message ending in an ideographic space renders
  differently (`tests/e2e/check_real_model.py`, the content-parts case).
  Marked `fastest:` in the source; a candidate to send upstream.
* `jinja/value.cpp`: `split()` and `rsplit()` mark every piece of a client
  string as client text; upstream marks only the last piece, so in
  `content.split('</think>')[0]` (Qwen3.5's template takes an assistant
  turn's thinking this way) special-token text would be read as real
  tokens. Found by the differential test of decision 0018
  (`tests/models/formats/qwen35_test.cpp`). Marked `fastest:` in the source;
  a candidate to send upstream.
* `jinja/value.cpp`, `jinja/value.h`: text made from a value that holds a
  client string is client text: `tojson` of it, and a list or a mapping
  turned into a string (`| string`, `~`).  Object keys from client JSON are
  client strings.  Upstream returns such text as template text, so a tool
  definition, a call's arguments or an argument's name could carry
  special-token text that is then read as real tokens.  Found while checking
  that the special tokens of a prompt are the ones the template wrote
  (`tests/core/render/chat_test.cpp`, the twin with the client's special-token
  text taken out).  Marked `fastest:` in the source; a candidate to send
  upstream.
* `jinja/value.cpp`, `jinja/value.h`: `tojson(sort_keys=true)` is implemented
  (sorting object keys lexicographically in UTF-8 order, matching Python's
  `json.dumps(..., sort_keys=True)`); upstream threw `not_implemented_exception`.
  Needed by Inkling-2-Mini's chat template. Marked `flyx:` in the source.

Everything else is byte-for-byte upstream.

To update: copy the same files from a newer llama.cpp commit, note the
commit here, and run the chat-template tests (`ctest -R chat`).

flyx uses it in `core/render/chat.cpp` (decision 0017).
