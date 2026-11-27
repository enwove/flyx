## Summary
Describe what changed and why.

## Checklist
- [ ] Read [AGENTS.md](../AGENTS.md) and verified layering rules (`python tools/check_layering.py`)
- [ ] Updated [docs/design/numerics.md](../docs/design/numerics.md) if adding or modifying CUDA kernels (`python tools/check_numerics.py`)
- [ ] Verified public docs and links (`python tools/check_docs.py`)
- [ ] Every new source file starts with `Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0`
- [ ] Tests pass (`ctest --preset dev` or `ctest --preset cpu`; a change to DV's code: `ctest --preset full` or `cpu-full` too, AGENTS.md gate 3)
- [ ] Performance claims include hardware, driver, model, quantization, and exact command
