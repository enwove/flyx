# XGrammar (C++ core)

Grammar-constrained decoding: structural tags (tool calls, reasoning blocks),
JSON schema, EBNF; a matcher that accepts tokens, rolls back and fills the
bitmask of the tokens a grammar allows next. flyx uses it to keep tool
calls well formed (decision 0020, `core/grammar/`).

Copied (with the differences below) from

    https://github.com/mlc-ai/xgrammar  commit 8f9a127 (2026-11-06)

under the Apache License 2.0 (`LICENSE`, `NOTICE`).

| here | upstream |
|---|---|
| `include/xgrammar/` | `include/xgrammar/` |
| `cpp/` | `cpp/` without `cpp/tvm_ffi/` (the Python bindings) |
| `3rdparty/picojson/picojson.h` | `3rdparty/picojson/picojson.h` (BSD-2-Clause, in the header) |
| `3rdparty/dlpack/` | dmlc/dlpack commit bbd2f4d (the submodule's pin): `include/dlpack/dlpack.h`, `LICENSE` (Apache-2.0) |

Not copied: the Python package, TVM-FFI bindings, cpptrace (built with
`XGRAMMAR_ENABLE_CPPTRACE=0`), GPU kernels, web code, tests. The structural
tags for each model family are built by flyx's chat formats
(`models/formats/`), not by xgrammar's Python `get_model_structural_tag`.

Differences from upstream, each marked `flyx:` in the source and tested by
`tests/third_party/xgrammar/local_changes_test.cpp`:

* `cpp/tokenizer_info.cc`: a token's subtree in the sorted vocabulary ends
  at the first token that does not start with it. Upstream keeps every
  token that contains it anywhere, so with `ab` and `acab` next to each
  other in the sorted vocabulary (one that has no token `ac`) a grammar
  that rejects `ab` skips `acab` with it, and the mask leaves out a token
  `AcceptToken` takes. A candidate to send upstream.
* Atomic tokens: tokens that have text and match as tokens only, for a chat
  format whose reply markers are control tokens (`core/grammar/grammar.cpp`
  names them; Inkling-2's `<|message_model|>`, `<|end_message|>`). Upstream
  matches a token that has text along its bytes, so a control token given
  its text is also taken wherever a grammar takes those characters (in free
  text, inside a JSON string), and a grammar cannot tell the token from its
  spelling in ordinary tokens.
  * `include/xgrammar/tokenizer_info.h`, `cpp/tokenizer_info.cc`,
    `cpp/tokenizer_info_impl.h`: `TokenizerInfo` takes `atomic_token_ids`.
    They keep their decoded text, stay out of the sorted byte vocabulary
    (which makes neighbours of the kind the fix above is for) and are
    listed with the special ids; the metadata carries them
    (`atomic_token_ids`), so a serialized `TokenizerInfo` keeps them.
  * `cpp/structural_tag.cc`: a tag's `begin` and `end`, a `const_string`
    and a `tags_with_separator` separator are split at the atomic tokens
    their text holds (the leftmost, the longest there) into byte strings
    and token terminals. `any_text` does not add an enclosing end that
    holds an atomic token to its byte exclusions: only a format's own
    `excludes` keep that spelling out of the text. A trigger, the `begin`
    of a tag in `triggered_tags` and a `dispatch` key that hold an atomic
    token's text are refused, because dispatch scans bytes.
  * `cpp/earley_parser.{h,cc}`, `cpp/grammar_matcher.cc`: an atomic id
    advances explicit token edges only (never its bytes, never an
    exclude-token wildcard); `FillNextTokenBitmask` sets its bit by trying
    the same edges on a row it pops again; one that is also a stop id is
    taken as a delimiter only where that completes the grammar.

Everything else is byte-for-byte upstream.

To update: copy the same directories from a newer commit, note it here,
make the differences above again where the new version still needs them,
and run `ctest -R "grammar|inkling2|golden|cuda_ops"` and the HTTP checks
on a real Inkling-2 model (`tests/apps/flyx-server/check_server.py`: tool
calls and JSON answers).
