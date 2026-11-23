# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Check Responses replacement when different Codex threads submit matching histories."""
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from check_server import Client
from client_conversations import post, queued_pair, running_pair, serve_cases


def check(url):
    client = Client(url, None)
    model = "test model"
    long_prompt = "0 Write integers from 1 to 2000."
    blocker = {"model": model, "prompt": long_prompt, "stream": True, "temperature": 0, "max_tokens": 8000}
    opening = [{"role": "user", "content": "Hello"}]
    followup = [{"role": "assistant", "content": "OK"}, {"role": "user", "content": "Continue"}]
    resumed = opening + followup

    def response(thread, items, affinity="shared", limit=8):
        headers = {"session-id": affinity}
        if thread:
            headers["thread-id"] = thread
        body = {"model": model, "input": items, "max_output_tokens": limit, "temperature": 0,
                "prompt_cache_key": affinity}
        return post(client, "/v1/responses", body, headers)

    def messages():
        body = {"model": model, "messages": opening, "max_tokens": 8, "temperature": 0}
        return post(client, "/v1/messages", body, {"x-claude-code-session-id": "s"})

    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for first, before, second, after, expected in (
            ("a", opening, "b", opening, 200), ("a", opening, "b", resumed, 200),
            ("b", resumed, "a", opening, 200), ("a", opening, "a", opening, 500),
            ("a", opening, "a", resumed, 500), ("a", resumed, "a", opening, 200),
            ("", opening, "b", opening, 500), ("a", opening, "", opening, 500), ("", opening, "", opening, 500),
        ):
            # Thread identity can differ under shared cache affinity, or stay the same when affinity changes.
            affinity = "shared" if first and second and first != second else "other"
            (old, reply), (new, _), queued = queued_pair(client, pool, blocker, partial(response, first, before),
                                                         partial(response, second, after, affinity))
            if (old, new) != (expected, 200) or (expected == 200 and queued != 2):
                failures.append(("queued", first, len(before), second, len(after), old, new, queued))
            elif expected == 500:
                assert reply["error"]["type"] == "server_error", reply["error"]["type"]

        # Both protocols accept opaque strings, including the encoding used by the other protocol.
        codex = partial(response, "messages:1:s", opening)
        anonymous = partial(response, "", opening)
        for first, second, expected in ((messages, codex, 200), (codex, messages, 200),
                                        (messages, anonymous, 409), (anonymous, messages, 500)):
            (old, _), (new, _), queued = queued_pair(client, pool, blocker, first, second)
            if (old, new) != (expected, 200) or (expected == 200 and queued != 2):
                failures.append(("cross API", old, new, queued))

        long = [{"role": "user", "content": long_prompt}]
        for thread, items in (("b", long), ("b", long + followup), ("a", long + followup)):
            (old, reply), (new, _), queued = running_pair(client, pool, "responses",
                partial(response, "a", long, limit=1024), partial(response, thread, items))
            assert (old, new) == (200, 200), (old, new)
            tokens = reply["usage"]["output_tokens"]
            if thread == "a":
                assert reply["status"] == "incomplete" and reply["incomplete_details"] is None and 0 < tokens < 1024
            elif queued != 1 or tokens != 1024 or reply["incomplete_details"] != {"reason": "max_output_tokens"}:
                failures.append(("running", thread, len(items), queued, tokens, reply["incomplete_details"]))
    assert not failures, failures


if __name__ == "__main__":
    serve_cases(check)
