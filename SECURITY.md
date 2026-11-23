# Security

flyx serves an HTTP API (`flyx serve`) and reads model files it is given, so both are places to look for problems:
a request that crashes or hangs the server, reads or writes outside what it should, or a model file whose header
makes the reader go outside the file.

**Please report a security problem privately**, through the repository's private vulnerability reporting (on
GitHub: the Security tab, "Report a vulnerability"), not in a public issue. Include the version or commit, how to
reproduce it, and what it lets an attacker do. We will acknowledge the report, work on a fix, and credit you in its
release notes unless you prefer otherwise.

What flyx assumes, so you can judge what is in scope:

- `flyx serve` listens on `127.0.0.1` by default. Bound to another address, or reached through a reverse proxy or
  a tunnel, it should run with `--api-key`; the model panel's endpoints (`POST /v1/load` and the Hugging Face
  listing), the request log and the server log answer only this machine or the key's holder. A request a proxy
  passed on (it carries `Forwarded`, `X-Forwarded-*`, `X-Real-IP` or `Via`) does not count as this machine's.
- Without a key, a request a browser sends from another site is refused (`Sec-Fetch-Site`, `Origin` and `Host` are
  checked, which also stops DNS rebinding); only a link to the dashboard page itself may be followed from
  elsewhere. A page may call the inference API from a browser when its origin is this machine's (`localhost`, any
  port: the default) or one `--cors` names; `--cors '*'` without a key lets any page a browser opens use the model
  (Chrome and Edge still ask the user before a public page may reach a loopback address).
  The model panel, the request log, the logs and the telemetry are never another origin's.
- A request is held to 64 MiB, 64 levels of JSON nesting and an 8 KiB target before a handler sees it
  (`server_hardening` tests these).
- A model file is data, never code: the `.flyx` and GGUF readers check the offsets and sizes a header gives
  against the file (the `.flyx` reader's refusals are tested by `flyx_reader`). Chat templates run in a Jinja
  engine without file or network access.
- Models fetched by name come from Hugging Face over HTTPS and are checked against the SHA-256 the Hub gives.
