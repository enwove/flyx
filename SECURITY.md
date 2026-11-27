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
  a tunnel, it should run with `--api-key`. By default, management endpoints (`POST /v1/load`, `/v1/hub/*`,
  `/v1/settings/network`), the request log and the server log answer only this machine or the key's holder.
  On a trusted LAN without a key, `--no-auth` or the saved `remote_manage` setting (effective only while bound to a
  non-loopback address) opts direct, unforwarded LAN clients into management, logs and requests; `POST /v1/hub/config`
  still requires this machine or `--api-key`, and `POST /v1/hub/dir`, `POST /v1/device/close` and
  `POST /v1/agents/launch` always require loopback. A request a proxy passed on (it carries `Forwarded`,
  any `X-Forwarded-*` header, `X-Real-IP`, `Via`, `CF-Connecting-IP` or `True-Client-IP`) never counts as loopback or
  unforwarded LAN, and always requires `--api-key` for management. An opaque TCP tunnel (such as SSH port forwarding)
  or a reverse proxy that strips forwarding headers cannot be distinguished from a direct client by HTTP headers alone:
  always set `--api-key` when exposing `flyx serve` through a proxy or tunnel.
- Without a key, a request a browser sends from another site is refused (`Sec-Fetch-Site` and `Origin` are checked; on
  loopback listeners `Host` must also be a loopback address to stop DNS rebinding, and on keyless LAN management routes
  `Host` must match the socket's local address/port, a known listener address, or the configured `--host` / system
  hostname; non-browser LAN inference clients behind Docker port mappings or custom hostnames remain unaffected). Only
  a link to the dashboard page itself may be followed from elsewhere. A page may call the inference API from a browser
  when its origin is this machine's (`localhost`, any port: the default) or one `--cors` names; `--cors '*'` without a
  key lets any page a browser opens use the model (Chrome and Edge still ask the user before a public page may reach a
  loopback address). The model panel, the request log, the logs and the telemetry are never another origin's.
- A request is held to 64 MiB, 64 levels of JSON nesting and an 8 KiB target before a handler sees it
  (`server_hardening` tests these).
- A model file is data, never code: the `.flyx` and GGUF readers check the offsets and sizes a header gives
  against the file (the `.flyx` reader's refusals are tested by `flyx_reader`). Chat templates run in a Jinja
  engine without file or network access.
- Models fetched by name come from Hugging Face (or a configured mirror) and are checked against the SHA-256 the Hub
  gives. A configured `HF_TOKEN` is sent only over `https://` to the default Hugging Face endpoint or an `HF_ENDPOINT`
  environment override, never over `http://` or to a runtime-saved mirror. Remote management requests cannot probe
  Windows UNC or device paths (`\\server\share`, `\\?\`, `\\.\`), mapped network drives (`DRIVE_REMOTE`), or reparse
  points / symlinks targeting them outside an operator-configured `--models-dir`.
