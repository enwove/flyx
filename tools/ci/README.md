# CI

The CPU tier runs in GitHub Actions via [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml). Both scripts can be run directly on matching hosts.

| Tier | Script | Environment | Coverage |
|---|---|---|---|
| `cpu` | `tools/ci/cpu.sh` | Linux, no GPU (hosted runner on push/PR) | `cpu` preset build, float64 reference vs `transformers`, tokenizer, `cpu_ref` golden tests, `layering`, `numerics`, `docs` checks |
| `gpu` | `tools/ci/gpu.sh` | Linux + NVIDIA GPU + test models (self-hosted runner) | `dev` preset CUDA build, full `ctest` (op tests, fused/unfused golden tests), real-model check vs `transformers`, HTTP API suite, `flyx bench` |

Both scripts fail fast and store logs in `logs/ci/` (gitignored). The GPU tier writes benchmark output to `logs/ci/bench.txt`.

Requirements: CMake ≥ 3.24, Ninja, Python 3.11+ with `torch`, `transformers`, `tokenizers`, `safetensors`, `gguf`; for `gpu`, CUDA 13 (`nvcc`) and converted models under `$MODELS` (provisioned by `tools/quickstart.sh`).
