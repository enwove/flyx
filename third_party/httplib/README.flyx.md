# cpp-httplib 0.20.0

Local changes over upstream 0.20.0:

1. `detail::parse_header` skips URL decoding for the `Authorization` and `X-API-Key` HTTP headers,
   so an API key containing a percent escape reaches the server as supplied by the client. The
   release smoke test sends both headers.
2. `detail::read_headers` caps the number of headers per request at `CPPHTTPLIB_HEADER_MAX_COUNT`
   (default 128), preventing unbounded header-map growth on unauthenticated connections.
3. `detail::read_content` holds a chunked request body, and one sent without a `Content-Length`, to
   `payload_max_length` (HTTP 413); upstream 0.20.0 held only a body with a `Content-Length` to it, so
   `Transfer-Encoding: chunked` let any peer fill memory before a handler (or the API key check) ran.
4. `stream_line_reader::getline` refuses a line longer than `CPPHTTPLIB_LINE_MAX_LENGTH` (default 64 KiB:
   above the request-line and header limits, so those still answer 414 and 400); upstream buffered a line
   without an end until memory ran out.
5. `Server::process_request` answers a request target over `CPPHTTPLIB_REQUEST_URI_MAX_LENGTH` with 414 at
   once and closes the connection; upstream read the headers a second time first, which waited for the read
   timeout and held a worker thread for as long.

Changes 3 to 5 are covered by `tests/apps/flyx-server/hardening.py` (ctest `server_hardening`).

[Upstream release](https://github.com/yhirose/cpp-httplib/releases/tag/v0.20.0)
