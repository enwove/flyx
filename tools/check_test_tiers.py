# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
"""The quick tier stays quick: no test labelled quick takes longer than a limit (docs/testing.md, section 2).

    python tools/check_test_tiers.py JUNIT.xml [JUNIT.xml ...] [--limit SECONDS] [--build DIR] [--warn-only]

Reads ctest's JUnit reports (ctest --output-junit), which carry each test's labels (property cmake_labels) and time,
and names every test labelled quick that took longer than --limit seconds (default 20, as timed on a CI runner).
flyx_finish_test_labels() infers a test's tier from how it is registered and cannot see how long it runs: a slow test
inferred tiny lands in quick, and `ctest --preset quick` gets slow for everyone.  The times are the run's own: a test
run beside seven others (-j 8) on a runner of four cores takes longer than alone.

--build DIR names the build tree the reports came from: the tests that set up a fixture there (make_assets) are then
listed apart and not held to the limit, since their time is what they make, once per change of its inputs.
--warn-only reports and exits 0 (nightly, until the limit has held a while); without it a test over the limit fails.
Under GitHub Actions each test over the limit is also a warning annotation, and the summary goes to the job's page.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


def read_reports(paths: list[str]) -> dict[str, tuple[float, list[str]]]:
    """Every test run (not skipped, not disabled): its time and labels; a test in several reports keeps its longest."""
    tests: dict[str, tuple[float, list[str]]] = {}
    labelled = False
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            labels: list[str] = []
            for prop in case.iter("property"):
                if prop.get("name") == "cmake_labels":
                    labels = [label for label in prop.get("value", "").split(";") if label]
                    labelled = True
            if case.get("status") in ("notrun", "disabled") or case.find("skipped") is not None:
                continue
            name, seconds = case.get("name", "?"), float(case.get("time") or 0)
            if name not in tests or tests[name][0] < seconds:
                tests[name] = (seconds, labels)
    if tests and not labelled:
        sys.exit(f"{', '.join(paths)}: no test carries its labels (cmake_labels): not a report of ctest --output-junit")
    return tests


def fixture_setups(build: str) -> set[str]:
    """The tests that set up a fixture in this build tree (FIXTURES_SETUP), as ctest lists them."""
    listing = subprocess.run(["ctest", "--test-dir", build, "--show-only=json-v1"], check=True, capture_output=True,
                             text=True).stdout
    setups = set()
    for test in json.loads(listing).get("tests", []):
        if any(p.get("name") == "FIXTURES_SETUP" for p in test.get("properties", [])):
            setups.add(test["name"])
    return setups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", nargs="+", help="ctest --output-junit reports")
    parser.add_argument("--limit", type=float, default=20.0, help="seconds a quick test may take (default 20)")
    parser.add_argument("--build", help="the build tree the reports came from: its fixture setups are listed apart")
    parser.add_argument("--warn-only", action="store_true", help="report tests over the limit, exit 0")
    args = parser.parse_args()

    tests = read_reports(args.reports)
    setups = fixture_setups(args.build) if args.build else set()
    quick = {n: s for n, (s, labels) in tests.items() if "quick" in labels and n not in setups}
    made = {n: s for n, (s, labels) in tests.items() if "quick" in labels and n in setups}
    over = sorted(((s, n) for n, s in quick.items() if s > args.limit), reverse=True)
    slowest = sorted(((s, n) for n, s in quick.items()), reverse=True)[:5]

    lines = [f"quick tier: {len(quick)} tests run, {sum(quick.values()):.1f} s in all; the slowest: "
             + (", ".join(f"{n} {s:.1f} s" for s, n in slowest) or "none")]
    if made:
        lines.append("fixture setups (their time is what they make, not held to the limit): "
                     + ", ".join(f"{n} {s:.1f} s" for n, s in sorted(made.items())))
    if over:
        lines.append(f"over {args.limit:g} s, too slow for quick (give it another tier, or make it faster):")
        lines += [f"  {n}: {s:.1f} s (labels {';'.join(tests[n][1])})" for s, n in over]
    else:
        lines.append(f"no quick test over {args.limit:g} s")
    print("\n".join(lines))

    if os.environ.get("GITHUB_ACTIONS") == "true":
        for s, n in over:
            print(f"::warning title=quick test over {args.limit:g} s::{n} took {s:.1f} s (tools/check_test_tiers.py)")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write("### Quick tier times (tools/check_test_tiers.py)\n\n```\n" + "\n".join(lines) + "\n```\n")
    return 1 if over and not args.warn_only else 0


if __name__ == "__main__":
    sys.exit(main())
