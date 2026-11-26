# CI

Every workflow runs on GitHub's hosted runners, none of which has a GPU: CI compiles the CUDA backend but runs no
test of it. The GPU suite runs on a maintainer's machine (`tools/ci/gpu.sh`, below).

| Workflow | Check | When | What |
|---|---|---|---|
| [`ci.yml`](../../.github/workflows/ci.yml) | `cpu` | every pull request, every push to `main` | `tools/ci/cpu.sh`: the `cpu` preset's build and every test but DV's (`ctest --preset cpu`), the static checks |
| [`cpu-full.yml`](../../.github/workflows/cpu-full.yml) | `cpu-dv` | a pull request that changes DV's code (AGENTS.md gate 3's paths) | `tools/ci/cpu.sh cpu-full -L '^dv$'`: DV's tests (`FLYX_TEST_DV`) |
| [`cuda-compile.yml`](../../.github/workflows/cuda-compile.yml) | `cuda-compile` | a pull request that changes what the CUDA build compiles or how (its `paths`: `backends/cuda`, `ops`, `bench/microbench`, `tests/backends/cuda`, `core/interfaces`, `include`, `tools/codegen`, CMake files) | `tools/ci/cuda-compile.sh`: the `dev` preset with CUDA 13.0's `nvcc` from PyPI, `FLYX_CUDA_ARCHS=120-real`, compiled, nothing run |
| [`nightly.yml`](../../.github/workflows/nightly.yml) | `cpu-full`, `cuda-compile`, `windows-cpu` | every night (18:17 UTC) on `main`; by hand (Actions, Nightly, Run workflow) | every test of `cpu-full`, DV's included; the CUDA build with `dev-full`; on Windows the `cpu-windows` build, its quick tier, then the rest |
| [`release-smoke.yml`](../../.github/workflows/release-smoke.yml) | `checksums`, `linux`, `windows` | a release published; by hand with its tag | the release's assets as people download them, on machines without a GPU (below) |

A pull request's `cpu-dv` and `cuda-compile` checks appear only when it touches their paths. A change to code DV
shares with the rest (the round graphs, the kernels) runs no DV test on its pull request: gate 3 asks for DV's
tests only when DV's own files change, and the nightly `cpu-full` is where DV breaking shows. When DV's code gets a
build switch of its own (off in the release package), the nightly `cuda-compile` turns it on (a TODO in
`tools/ci/cuda-compile.sh` and `nightly.yml`).

## Scripts

| Script | Runs | Output in `logs/ci/` (gitignored) |
|---|---|---|
| `tools/ci/cpu.sh [cpu\|cpu-full] [ctest arguments]` | configure the preset, build (`cpu-full` is `cpu`'s build tree with DV's tests registered), `ctest --preset` it | `<preset>.log`, `ctest.xml` (JUnit) |
| `tools/ci/cuda-compile.sh [dev\|dev-full]` | configure the preset into `$BUILD` (default `~/flyx-build-cuda-compile`) with `$CUDACXX` and `FLYX_CUDA_ARCHS` (default `120-real`), build, check that the CUDA backend, `cuda_op_test` and `cuda_bench` were built | `cuda-compile.log` |
| `tools/ci/gpu.sh` | a maintainer's machine with an NVIDIA GPU and the converted models under `$MODELS` (`tools/quickstart.sh` provisions both): the `dev` preset's CUDA build, `ctest --preset dev`, the real-model check against `transformers`, the HTTP API suite, `flyx bench`. **No workflow runs it**: there is no GPU runner | `gpu.log`, `bench.txt` |
| `tools/check_test_tiers.py JUNIT.xml... [--build DIR] [--warn-only]` | names every test labelled `quick` that took longer than 20 s in the reports (fixture setups such as `make_assets` apart); the nightly runs warn | |
| `tools/ci/release_smoke.sh`, `tools/ci/release_smoke_windows.ps1`, `tools/ci/tiny_model.py`, `tools/ci/release_smoke_report.py` | the release smoke test's pieces (below) | |

Every script can be run by hand on a matching machine. The one-architecture CUDA compile leaves out device code under
another architecture's `#if`: only the release package's build (`80`, `89`, `90`, `100`, `120`) compiles all of it.

## Caches

The hosted workflows compile through ccache (`CMakeLists.txt` picks it up when installed, `nvcc`'s compiles too)
and keep its directory per commit: `ccache-<os>-<commit>` for the CPU jobs (the `cpu` and `cpu-full` builds are the
same compiles), `ccache-cuda-<os>-<commit>` for the CUDA one. A run restores the newest cache saved under its prefix
(its pull request's last push, else `main`'s last run: the nightly's for the CUDA build) and compiles only what
changed. A run on `main` is never cancelled once it has started; a pull request's newer push cancels its older run.

The tests' fixtures (`tests/e2e/make_tiny.py`) are made again only when what they are made from changed (the Python
reference, the converter, the chat templates, the versions of the packages that build them). Each group, core's and
DV's, has its own digest (`python tests/e2e/make_tiny.py --digest --group core|dv`) and stamp in the assets directory
(`.inputs-core`, `.inputs-dv`). The `cpu` job caches the assets as `fixtures-core-<os>-<core digest>`; `cpu-full`
as `fixtures-all-<os>-<core digest>-<dv digest>`, restoring either kind, and saves only when both stamps match. A
stamp, not the cache's name, decides on every run what is made again, so a cache made from other inputs is made
over, never trusted.

## Release smoke test

`release-smoke.yml` downloads a release's assets with `gh release download` and checks:

- every asset against `SHA256SUMS.txt` (`sha256sum -c --strict`), and no asset the file leaves out;
- the Linux tarball and wheel in containers of `ubuntu:22.04`, `ubuntu:24.04`, `ubuntu:26.04`, `debian:12`,
  `debian:13`, `fedora:latest` and `archlinux:latest`, as pulled (`tools/ci/release_smoke.sh`, run as root): the
  libraries each binary needs and the GLIBC version they ask for (`ldd`), `flyx --version`, `flyx --help`,
  `flyx-server --help`, a run on the CUDA backend refused with a reason and not a crash, a first token of the tiny
  model on the float64 backend the CLI falls back to (timed), then, with the distribution's own Python installed, the
  tarball's Python binding and its server (`packaging/smoke_model.py`) and the wheel installed by `pip` and
  generating (`packaging/smoke_binding.py`);
- the Windows executable and the zip's `flyx.exe` on a Windows runner (`tools/ci/release_smoke_windows.ps1`): the
  DLLs they import (no Visual C++ runtime, no CUDA or driver DLL: `dumpbin /dependents`), the same CLI checks, the
  server (and the zip's Python binding), then the wheel installed by `pip`.

The tiny model (`tiny.flyx`) is made by the release's own sources checked out at its tag (`tools/ci/tiny_model.py`:
`make_tiny.py`'s first model, seconds), so it is a file of the format that release reads. Each platform's results
are a table on its job's page; the `report` job puts them on one. A release published by a workflow with
`GITHUB_TOKEN` starts no other workflow: run this one by hand then.

## Requirements

CMake ≥ 3.24, Ninja, Python 3.12+ with the packages pinned in [`tests/requirements.txt`](../../tests/requirements.txt)
(`pip install -r tests/requirements.txt`; the hosted workflows install exactly these). For `cuda-compile.sh`, CUDA 13
(`nvcc`: an installed toolkit, or PyPI's `nvidia-cuda-nvcc==13.0.88` with `nvidia-nvvm`, `nvidia-cuda-runtime`,
`nvidia-cuda-cccl` and `nvidia-cuda-crt` as `cuda-compile.yml` and `tools/quickstart.sh` install them). For
`gpu.sh`, an NVIDIA GPU (driver 580 or newer) and the converted models.
