# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""Fill a dedicated flyx server's 50-entry request log and measure conditional reads."""
import argparse
from collections import Counter
import json
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from http_api import api_request, get


def poll(url, count):
    statuses = Counter()
    total_bytes = 0
    tag = None
    start = time.process_time()
    for _ in range(count):
        request = api_request(url, "/v1/requests?limit=50")
        if tag:
            request.add_header("If-None-Match", tag)
        try:
            response = urlopen(request, timeout=30)
        except HTTPError as exc:
            if exc.code != 304:
                raise
            response = exc
        with response:
            body = response.read()
            statuses[response.status] += 1
            total_bytes += len(body)
            if response.status == 304:
                if body or not tag:
                    raise ValueError("unexpected conditional response")
            elif len(json.loads(body)["requests"]) != 50:
                raise ValueError("expected 50 completed requests in the log")
            tag = response.headers.get("ETag")
    return {"statuses": dict(statuses), "body_bytes": total_bytes,
            "client_cpu_ms": round((time.process_time() - start) * 1000, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    model = get(args.url, "/v1/models")["data"][0]["id"]
    for row in range(50):
        message = {"role": "user", "content": f"Row {row}", "annotation": "Z" * 32768}
        body = {"model": model, "max_tokens": 0, "messages": [message]}
        with urlopen(api_request(args.url, "/v1/chat/completions", body), timeout=60) as response:
            response.read()
    for _ in range(3):
        print(json.dumps(poll(args.url, 24)), flush=True)


if __name__ == "__main__":
    main()
