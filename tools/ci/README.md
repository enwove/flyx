# CI

The CPU tier runs in GitHub Actions via [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). Both scripts can be run directly on matching hosts.

| Tier | Script | Environment | Coverage |
|---|---|---|---|
| `cpu` | `tools/ci/cpu.sh` | Linux, no GPU (hosted runner on push/PR) | `cpu` preset build, float64 reference vs `transformers`, tokenizer, `cpu_ref` golden tests, `layering`, `numerics`, `docs` checks |
| `gpu` | `tools/ci/gpu.sh` | Linux + NVIDIA GPU + test models (self-hosted runner) | `dev` preset CUDA build, full `ctest` (op tests, fused/unfused golden tests), real-model check vs `transformers`, HTTP API suite, `flyx bench` |

Both scripts fail fast and store logs in `logs/ci/` (gitignored). The CPU tier writes ctest's JUnit report to `logs/ci/ctest.xml`, which the hosted workflow uploads as an artifact on every run; the GPU tier writes benchmark output to `logs/ci/bench.txt`.

The hosted workflow compiles through ccache (`CMakeLists.txt` picks it up when installed) and keeps its cache per commit, so a run compiles only what changed since the last one saved. A run on `main` is never cancelled once it has started; a pull request's newer push cancels its older run.

The tests' fixtures (`make_assets`: `tests/e2e/make_tiny.py`) are most of the CPU tier's time when they have to be made. They are made again only when what they are made from changed (the Python reference, the converter, the chat templates, the versions of the packages that build them: the digest in `<build>/assets/.inputs`), and the hosted workflow keeps that directory in a cache named by the same digest (`python tests/e2e/make_tiny.py --digest`), so a change that touches none of those restores them instead.

Requirements: CMake ≥ 3.24, Ninja, Python 3.12+ with the packages pinned in [`tests/requirements.txt`](../../tests/requirements.txt) (`pip install -r tests/requirements.txt`; the hosted workflow installs exactly these); for `gpu`, CUDA 13 (`nvcc`) and converted models under `$MODELS` (provisioned by `tools/quickstart.sh`).
