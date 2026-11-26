# Contributing to flyx

Read [AGENTS.md](AGENTS.md) first—it defines the directory layering boundaries, the five correctness gates, and the licensing rules that every change must satisfy.

## Architecture and Build Entrypoints

- Architecture and source layout: [docs/design/architecture.md](docs/design/architecture.md)
- Kernel precision contract: [docs/design/numerics.md](docs/design/numerics.md)
- Build presets and toolchains: [docs/building.md](docs/building.md)
- Test suites and real-model checks: [docs/testing.md](docs/testing.md)

```sh
# CUDA development build + full test suite
cmake --preset dev && cmake --build --preset dev
pip install -r tests/requirements.txt
ctest --preset quick                     # while editing: unit, tiny and smoke tests (docs/testing.md section 2)
ctest --preset dev --output-on-failure   # every test but DV's (DV is not in production)

# A change to DV's code: the same tree with DV's tests registered (AGENTS.md gate 3)
cmake --preset dev-full && ctest --preset full --output-on-failure

# CPU-only build (no GPU required)
cmake --preset cpu && cmake --build --preset cpu
ctest --preset cpu --output-on-failure   # cpu-full (after cmake --preset cpu-full): DV's tests too
```

Changes touching CUDA kernels or model execution paths should also pass the real-model and server checks (`tools/quickstart.sh` steps `check` and `serve`).

## Pull Request Checklist

- **One topic per commit**, tests passing, commit message explains what changed and why.
- **File header:** every new source file starts with `Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0`.
- **Layering:** directory dependencies follow [AGENTS.md](AGENTS.md) (`python tools/check_layering.py`).
- **CUDA kernels:** include a float64 reference (`ops/<op>/ref.h`), op test (`tests/backends/cuda/op_test_<family>.cu`), and row in [docs/design/numerics.md](docs/design/numerics.md) (`python tools/check_numerics.py`).
- **Docs:** keep [llms.txt](llms.txt) and public docs in sync with CLI/API changes (`python tools/check_docs.py`).
- **Benchmarks:** include model, quantization, GPU, driver, and exact command in the PR description.
- **Third-party code:** credit algorithmic ideas in [docs/prior-art.md](docs/prior-art.md); vendor whole libraries under `third_party/` with license, `NOTICE` entry, and local-change notes in `README.flyx.md`.

## Reporting Bugs and Security Issues

- **Bugs:** include the exact command, model file/quantization, GPU, driver version, and stderr output. Any token mismatch between plain and speculative decoding is a bug.
- **Security:** report vulnerabilities privately per [SECURITY.md](SECURITY.md).

## License

Apache License 2.0 ([LICENSE](LICENSE)). Contributions are accepted under the same license.
