# Building and Packaging

Build prerequisites, CMake presets, CMake options, and release packaging for Linux and Windows.

---

## 1. Prerequisites

| Target | Required Tools |
|---|---|
| **Linux / WSL2 (CUDA)** | CUDA 13 (`nvcc` on `PATH`), CMake ≥ 3.24, Ninja, `g++` ≥ 11, Python 3.11+ (tuning table codegen & tests), `libcurl` (not linked: loaded at run time for HF Hub fetch, which the hub tests exercise) |
| **Windows (native MSVC + CUDA)** | Visual Studio 2022 C++ Build Tools, CMake ≥ 3.24, Ninja, Python 3.11+, CUDA 13 (`nvcc`, or auto-installed from PyPI wheels by `packaging\package_windows.ps1`) |
| **CPU-only CLI (`-DFLYX_CUDA=OFF -DFLYX_TESTS=OFF`)** | CMake ≥ 3.24, Ninja, C++20 compiler (no Python required) |
| **Runtime (release packages)** | Linux: `libc` + NVIDIA driver ≥ 580, and the system's `libcurl` to fetch models by name; Windows: NVIDIA driver ≥ 580 (CUDA and C++ runtimes linked statically). Python wheel: Python ≥ 3.9, no PyTorch required. |

---

## 2. CMake Presets

Defined in [CMakePresets.json](../CMakePresets.json).

| Preset | Platform | Build Dir | Purpose |
|---|---|---|---|
| `dev` | Linux / WSL2 | `~/flyx-build` | Release + tests, auto-detects local GPU arch (`nvidia-smi`; falls back to `sm_120`), keeps tuning candidates & lineinfo |
| `debug` | Linux / WSL2 | `~/flyx-build-debug` | `RelWithDebInfo` (line info) + tests |
| `cpu` | Linux / WSL2 | `~/flyx-build-cpu` | CPU-only (`cpu_ref` float64 backend) + tests |
| `dev-windows` | Windows (MSVC) | `%USERPROFILE%\flyx-build` | Release + tests, local GPU arch |
| `cpu-windows` | Windows (MSVC) | `%USERPROFILE%\flyx-build-cpu` | CPU-only + tests |
| `release-linux` | Linux / WSL2 | `~/flyx-build-release` | Static runtime, multi-arch (`sm_80,89,90,100,120` + `compute_120` PTX), compressed fatbins, pruned candidates |
| `release-windows` | Windows (MSVC) | `%USERPROFILE%\flyx-build-release` | Windows desktop/workstation multi-arch (`sm_80,89,120` + `compute_120` PTX; omits datacenter-only `sm_90,100`), `FLYX_TESTS=OFF` |

```sh
# Linux / WSL2 development build + tests
cmake --preset dev && cmake --build --preset dev
ctest --preset dev --output-on-failure
```

Build outputs in `<build-dir>/bin`: `flyx` (CLI + `flyx serve`; `flyx-server` symlink on Linux), `libflyx.so` / `flyx.dll` (C API), `libflyx-cuda.so` / `libflyx-cuda.dll`, `libflyx-cpu_ref.so` / `libflyx-cpu_ref.dll`, and test/benchmark executables.

---

## 3. Key CMake Options

| Option | Default (`dev` / `release`) | Effect |
|---|---|---|
| `FLYX_CUDA` | `ON` | Build CUDA backend (`OFF` for CPU-only). |
| `FLYX_CUDA_ARCHS` | local GPU / Linux: `80;89;90;100;120;120-virtual`, Win: `80;89;120;120-virtual` | Semicolon-separated SM architectures (≥ 8.0). |
| `FLYX_TESTS` | `ON` (`dev`) / `OFF` (`release-windows`) | Build unit, op, golden, and server tests (plus `flyx-trace` when CUPTI is present). |
| `FLYX_STATIC_RUNTIME` | `OFF` / `ON` | Statically link C++/CUDA runtimes for portable release binaries. |
| `FLYX_STANDALONE_EXE` | `OFF` | Also build a single-file `flyx-standalone` executable with `cuda` and `cpu_ref` backends linked in. |
| `FLYX_CUDA_LINEINFO` | `ON` / `OFF` | Emit `-lineinfo` for profilers. |
| `FLYX_CUDA_COMPRESS` | empty (`nvcc`'s default) / `size` | `nvcc --compress-mode` of the CUDA fatbins. |
| `FLYX_TUNE_CANDIDATES` | `ON` / `OFF` | Compile all tuning candidates (`OFF` prunes to active GPU table entries). |
| `FLYX_COMPILER_CACHE` | `ON` (Linux) / `OFF` (Win) | Auto-use `sccache` (C++ & `nvcc`) or `ccache` (C++) if on `PATH`. |

---

## 4. Release Packaging

Version comes from `project(flyx VERSION ...)` in [CMakeLists.txt](../CMakeLists.txt).

### Linux Tarball and Python Wheel

```sh
. "${VENV:-$HOME/.venvs/flyx}/bin/activate"
python -m pip install 'setuptools>=77' wheel auditwheel patchelf
JOBS=6 bash packaging/package.sh    # limit JOBS on low-RAM WSL2 hosts
```

Produces `dist/flyx-<version>-linux-x86_64-cuda13.tar.gz` and a `manylinux` Python wheel in `dist/` after installing and smoke-testing both (`packaging/smoke.sh`).

### Windows ZIP, Standalone EXE, and Python Wheel

```powershell
powershell -ExecutionPolicy Bypass -File packaging\package_windows.ps1 -Assets DIR [-Exe]
```

- Installs CUDA 13.0 PyPI wheels into `~\.venvs\flyx-win` automatically when that environment has no `nvcc` yet, and builds with it (an installed CUDA toolkit is not used).
- Produces:
  - `dist\flyx-<version>-windows-x86_64-cuda13.zip` (`bin\flyx.exe`, `lib\flyx.dll`, `lib\flyx.lib`, `lib\flyx\`, `include\`, `python\`; self-contained with zero runtime DLL dependencies beyond Windows + NVIDIA driver; note Windows uses `flyx.exe serve`, with no `flyx-server.exe`).
  - `dist\flyx-<version>-py3-none-win_amd64.whl` (self-contained ctypes Python wheel with the same DLLs bundled inside, no PyTorch required).
  - With `-Exe`: `dist\flyx-<version>-windows-x86_64-cuda13.exe` (single-file portable executable with both `cuda` and `cpu_ref` backends linked in, zero external DLLs; double-clicking in Explorer starts `flyx serve` and opens the web dashboard).
- Pass `-BuildDir DIR` for a custom build directory; `-Assets DIR` points to `tiny.flyx` from a test build to smoke-test both the unpacked ZIP (`cpu_ref`, `cuda`, HTTP server) and the wheel in an isolated virtual environment.
- Windows executables embed `core/os/utf8.manifest` (UTF-8 active code page); CUDA 13.1+ MSVC `/Zc:preprocessor` flags are handled automatically by `CMakeLists.txt`.

Native Windows development build (in PowerShell, dot-source `tools\windows\vsenv.ps1` to configure VS 2022, Ninja, CMake, `nvcc`, and `~\.venvs\flyx` / `~\.venvs\flyx-win`):

```powershell
. .\tools\windows\vsenv.ps1
cmake --preset dev-windows; cmake --build --preset dev-windows
ctest --preset dev-windows --output-on-failure
```

### Smoke-Testing an Unpacked Release

```sh
# Python from an unpacked archive
PYTHONPATH=<archive>/python python -c "import flyx; print(flyx.__all__)"

# Backend smoke check against a tiny model
python packaging/smoke_model.py <archive>/bin/flyx <tiny.flyx> --backend cuda
python packaging/smoke_model.py <archive>/bin/flyx <tiny.flyx> --backend cpu_ref
```
