# Chat templates the tests render

Real templates, as the model repositories ship them (Apache-2.0: Alibaba
Cloud / the Qwen team, OpenBMB, Google), so the template tests run without
downloading a model:

| file | from | chat format (models/formats) |
|---|---|---|
| `qwen3.5.jinja` | Qwen/Qwen3.5-4B, `chat_template.jinja` (sha256 a4aee8af...) | `qwen35` (its own renderer; XML tool calls) |
| `qwen3.jinja` | Qwen/Qwen3-4B, `tokenizer_config.json` → `chat_template` | `hermes` (JSON tool calls) |
| `minicpm5.jinja` | openbmb/MiniCPM5-1B, `chat_template.jinja` (sha256 7451a05c...) | `plain` (its `<function name=...>` calls are not read: tools refused) |
| `gemma4.jinja` | google/gemma-4-31B-it, `chat_template.jinja` (sha256 ae53464b...) | `gemma4` (thinking in its channel; `<\|tool_call>call:NAME{...}<tool_call\|>`) |
| `gemma4_early.jinja` | nvidia/Gemma-4-31B-IT-NVFP4, `chat_template.jinja` (sha256 94899c0f...: Google's before its 2026-07-09 revision) | `gemma4` |

`tests/e2e/make_tiny.py` renders the conversations of `chat_cases.py` with
each of them through transformers (`apply_chat_template`), and `chat_test`
renders the same with the engine.
