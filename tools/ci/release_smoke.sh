#!/usr/bin/env bash
# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# A release's Linux packages on a clean distribution without a GPU (.github/workflows/release-smoke.yml runs this as
# root in a container of each distribution it names):
#
#   bash tools/ci/release_smoke.sh PACKAGE_DIR WHEEL MODEL RESULTS
#
# PACKAGE_DIR is the unpacked tarball (flyx-<version>-linux-x86_64-cuda13/), WHEEL the release's wheel, MODEL the
# tiny test model (tiny.flyx: tools/ci/tiny_model.py).  Every check runs whatever the one before it gave:
#   - the libraries each binary needs are there (ldd: none missing, no GLIBC symbol version newer than this libc);
#   - flyx --version, flyx --help, flyx-server --help (the link is the server);
#   - a run on the CUDA backend is refused with a reason, not a crash (no driver here);
#   - the CLI falls back to the float64 backend and gives a first token of the tiny model (timed);
#   - with this distribution's Python installed: the tarball's Python binding and its server
#     (packaging/smoke_model.py), then the wheel installed by pip and generating (packaging/smoke_binding.py).
# RESULTS gets a line a check: check<TAB>ok|FAIL|expected|skipped<TAB>detail.  Exits 1 when a check failed.
set -uo pipefail
P=$(readlink -f "$1")
WHEEL=$(readlink -f "$2")
MODEL=$(readlink -f "$3")
RESULTS=$4
HERE=$(cd "$(dirname "$0")/../.." && pwd)
: > "$RESULTS"
failed=0
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

result() {  # check status detail
  printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$RESULTS"
  if [[ $2 == FAIL ]]; then failed=1; fi
}
one_line() { tr '\n\t' '  ' | sed -e "s|$P/||g" -e 's/  */ /g' -e 's/^ //' -e 's/ $//' | cut -c1-400; }
crashed() { (( $1 >= 128 )) || grep -qiE 'segmentation fault|core dumped|aborted|terminate called' <<< "$2"; }
ms_since() { echo $(( ($(date +%s%N) - $1) / 1000000 )); }

# shellcheck disable=SC1091
. /etc/os-release
libc=$(getconf GNU_LIBC_VERSION 2>/dev/null || /lib64/libc.so.6 2>/dev/null | head -n 1)
result system ok "${PRETTY_NAME:-$ID}, ${libc:-libc version unknown}"

# ldd is a script around the loader; a distribution without it still has the loader's --list.
list_libs() { if command -v ldd > /dev/null; then ldd "$1"; else /lib64/ld-linux-x86-64.so.2 --list "$1"; fi; }
for f in bin/flyx lib/libflyx.so lib/flyx/libflyx-cpu_ref.so lib/flyx/libflyx-cuda.so; do
  out=$(list_libs "$P/$f" 2>&1)
  missing=$(grep -E 'not found|not a dynamic' <<< "$out" | one_line)
  if [[ -n $missing ]]; then result "libraries of $f" FAIL "$missing"
  else result "libraries of $f" ok "$(awk '{print $1}' <<< "$out" | one_line)"; fi
done

out=$("$P/bin/flyx" --version 2>&1); rc=$?
starts=$(( rc == 0 ))
if (( starts )); then result "flyx --version" ok "$(one_line <<< "$out")"
else result "flyx --version" FAIL "exit $rc: $(one_line <<< "$out")"; fi
out=$("$P/bin/flyx" --help 2>&1); rc=$?
if (( rc == 0 )); then result "flyx --help" ok "$(head -n 1 <<< "$out" | one_line)"
else result "flyx --help" FAIL "exit $rc: $(head -n 3 <<< "$out" | one_line)"; fi
out=$("$P/bin/flyx-server" --help 2>&1); rc=$?
if [[ $out == *"usage: flyx serve"* ]]; then result "flyx-server --help" ok "is flyx serve (exit $rc)"
else result "flyx-server --help" FAIL "exit $rc: $(head -n 3 <<< "$out" | one_line)"; fi

# No GPU here: the CUDA backend, asked for, must say why it cannot run (the engine's "no usable backend" and the
# backend's reason).
out=$("$P/bin/flyx" run "$MODEL" -p hello -n 1 --ctx 128 --backend cuda 2>&1); rc=$?
if (( ! starts )); then result "no GPU: --backend cuda" FAIL "flyx does not start here (flyx --version)"
elif crashed $rc "$out"; then result "no GPU: --backend cuda" FAIL "crashed (exit $rc): $(one_line <<< "$out")"
elif (( rc == 0 )); then result "no GPU: --backend cuda" FAIL "ran on CUDA without a GPU: $(one_line <<< "$out")"
elif [[ $out != *"no usable backend"* ]]; then result "no GPU: --backend cuda" FAIL "exit $rc: $(one_line <<< "$out")"
else result "no GPU: --backend cuda" expected "exit $rc: $(one_line <<< "$out")"; fi

# The default backend list (cuda, then cpu_ref): it says it falls back, then generates.  Timed from the process's
# start to its exit after one token: loading the model, the prompt, the token.
start=$(date +%s%N)
out=$("$P/bin/flyx" run "$MODEL" -p hello -n 1 --ctx 128 2> "$T/err"); rc=$?
ms=$(ms_since "$start")
err=$(cat "$T/err")
if (( ! starts )); then result "first token (default backend)" FAIL "flyx does not start here (flyx --version)"
elif crashed $rc "$err"; then result "first token (default backend)" FAIL "crashed (exit $rc): $(one_line <<< "$err")"
elif (( rc != 0 )); then result "first token (default backend)" FAIL "exit $rc: $(one_line <<< "$err")"
else result "first token (default backend)" ok "${ms} ms, text $(printf '%q' "$out"); stderr: $(one_line <<< "$err")"; fi

# Python: whatever this distribution installs as python3 (with venv), as a user of it would.
install_python() {
  if command -v apt-get > /dev/null; then
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-venv > /dev/null
  elif command -v dnf > /dev/null; then dnf install -y -q python3 > /dev/null
  elif command -v pacman > /dev/null; then pacman -Sy --noconfirm --needed python > /dev/null
  else return 1; fi
}
if ! install_python > "$T/py" 2>&1 || ! command -v python3 > /dev/null; then
  result "python3" FAIL "could not install it: $(tail -n 3 "$T/py" | one_line)"
else
  result "python3" ok "$(python3 --version 2>&1)"
  if out=$(env -i PATH=/usr/bin:/bin HOME="$T" python3 "$HERE/packaging/smoke_model.py" "$P/bin/flyx" "$MODEL" \
             --backend cpu_ref 2>&1); then
    result "tarball's Python binding and server" ok "$(grep -E '^  ok' <<< "$out" | one_line)"
  else result "tarball's Python binding and server" FAIL "$(tail -n 4 <<< "$out" | one_line)"; fi

  if ! out=$(python3 -m venv "$T/venv" 2>&1); then result "wheel: pip install" FAIL "no venv: $(one_line <<< "$out")"
  elif ! out=$("$T/venv/bin/python" -m pip install --disable-pip-version-check -q "$WHEEL" 2>&1); then
    result "wheel: pip install" FAIL "$(grep -E 'ERROR|error' <<< "$out" | tail -n 2 | one_line)"
  else
    result "wheel: pip install" ok "$("$T/venv/bin/python" -m pip --version | cut -d' ' -f1-2) installed $(basename "$WHEEL")"
    site=$("$T/venv/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['platlib'])")
    if out=$("$T/venv/bin/python" -I "$HERE/packaging/smoke_binding.py" "$site" "$MODEL" 2>&1); then
      result "wheel: generate" ok "$(one_line <<< "$out")"
    else result "wheel: generate" FAIL "$(tail -n 4 <<< "$out" | one_line)"; fi
  fi
fi
exit $failed
