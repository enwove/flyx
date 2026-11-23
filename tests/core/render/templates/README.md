# Chat templates the tests render

Real templates, as the model repositories ship them (Apache-2.0: Alibaba
Cloud / the Qwen team, OpenBMB), so the template tests run without downloading
a model:

| file | from | chat format (models/formats) |
|---|---|---|
| `qwen3.5.jinja` | Qwen/Qwen3.5-4B, `chat_template.jinja` (sha256 a4aee8af...) | `qwen35` (its own renderer; XML tool calls) |
| `qwen3.jinja` | Qwen/Qwen3-4B, `tokenizer_config.json` → `chat_template` | `hermes` (JSON tool calls) |
| `minicpm5.jinja` | openbmb/MiniCPM5-1B, `chat_template.jinja` (sha256 7451a05c...) | `plain` (its `<function name=...>` calls are not read: tools refused) |

`tests/e2e/make_tiny.py` renders the conversations of `chat_cases.py` with
each of them through transformers (`apply_chat_template`), and `chat_test`
renders the same with the engine.
