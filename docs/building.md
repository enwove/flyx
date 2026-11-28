# Building and Packaging

Build prerequisites, CMake presets, CMake options, and release packaging for Linux and Windows.

---

## 1. Prerequisites

| Target | Required Tools |
|---|---|
| **Linux / WSL2 (CUDA)** | CUDA 13 (`nvcc` on `PATH`), CMake ≥ 3.24, Ninja, `g++` ≥ 11, Python 3.11+ for table generation, Python 3.12+ for the pinned test requirements, `libcurl` (not linked: loaded at run time for HF Hub fetch, which the hub tests exercise) |
| **Windows (native MSVC + CUDA)** | Visual Studio 2022 or newer with C++ Build Tools, CMake ≥ 3.24, Ninja, Python 3.11+ (Python 3.12+ for tests), CUDA 13 (`nvcc`, or auto-installed from PyPI wheels by `packaging\package_windows.ps1`) |
| **CPU-only CLI (`-DFLYX_CUDA=OFF -DFLYX_TESTS=OFF`)** | CMake ≥ 3.24, Ninja, C++20 compiler (no Python required) |
| **Runtime (release packages)** | Linux x86-64 requires glibc ≥ 2.35, the NVIDIA driver ≥ 580 for CUDA, and the system's `libcurl` to fetch models by name. Windows requires the NVIDIA driver ≥ 580 for CUDA (CUDA and C++ runtimes linked statically). The wheel requires Python ≥ 3.9. |

**CUDA 13.0.88 on Linux.** On glibc 2.41 or newer (Ubuntu 26.04 and recent Fedora and Arch releases), `nvcc` fails because the declarations of `rsqrt` and `rsqrtf` in `crt/math_functions.h` have different exception specifications from glibc's. Use Ubuntu 24.04, build in the Ubuntu 22.04 CUDA 13 container with [packaging/package_linux_docker.sh](../packaging/package_linux_docker.sh), or use a newer CUDA toolkit that supports your host distribution.

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
| `release-windows` | Windows (MSVC) | `%USERPROFILE%\flyx-build-release` | Windows desktop/workstation multi-arch (`sm_80,89,120` plus the specialized `120a-real` W4A4 target, zero PTX; omits datacenter-only `sm_90,100`), `FLYX_TESTS=OFF` |

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
| `FLYX_CUDA_ARCHS` | local GPU / Linux: `80-real;89-real;90-real;100-real;120-real;120-virtual`, Win: `80-real;89-real;120-real` (no PTX: a GPU of a later architecture than sm_120 needs the Linux package or a build of its own) | Semicolon-separated SM architectures (≥ 8.0). |
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

Build releases against **Ubuntu 22.04 (glibc 2.35)**. The same x86-64 packages target Debian 12 (glibc 2.36), Ubuntu 24.04, and Fedora and Arch systems with glibc 2.35 or newer.

On a Linux release host with Docker, NVIDIA Container Toolkit and a supported GPU, run

```sh
JOBS=6 bash packaging/package_linux_docker.sh
```

The helper uses the pinned CUDA 13.0.2 development image for Ubuntu 22.04. An optional positional argument selects another CUDA development image. `JOBS` limits parallel compiles. The container runs the release tests, then smoke-tests the unpacked tarball and installed wheel before placing both in `dist/`.

For a native build on the same baseline, activate a Python 3.12+ environment, install the Linux build prerequisites above and GNU binutils, then run

```sh
python -m pip install -r tests/requirements.txt wheel auditwheel patchelf
JOBS=6 bash packaging/package.sh
```

`packaging/smoke.sh` checks the dynamic symbols of every packaged ELF before starting the CLI. After auditwheel repair, `packaging/package.sh` checks both the wheel's manylinux platform tags and its bundled ELF files. Both scripts pass `MAX_GLIBC` (default `2.35`) to the checker. To inspect an extracted tarball and a wheel independently, run

```sh
python3 packaging/check_glibc.py --max-glibc 2.35 /path/to/extracted-package dist/*.whl
```

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

Native Windows development build (in PowerShell, dot-source `tools\windows\vsenv.ps1` to configure Visual Studio 2022 or newer, Ninja, CMake, `nvcc`, and `~\.venvs\flyx` / `~\.venvs\flyx-win`):

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

## 5. Docker

[Dockerfile.server](../packaging/Dockerfile.server) builds the engine and its HTTP server from this checkout into one image; [docker-compose.yml](../packaging/docker-compose.yml) runs it with the GPU reservation, the volumes and the environment. The project publishes no image of its own — the build is the release package's configuration, [CMakePresets.json](../CMakePresets.json)'s `release-linux` cache variables passed directly (static CUDA and C++ runtimes, compressed fatbins, tuning candidates pruned to what the GPU tables select) with `-DFLYX_TESTS=OFF`, so the runtime stage needs no CUDA toolkit, only what the NVIDIA Container Toolkit mounts in from the host driver and the system's `libcurl`.

Building needs nothing but Docker; running the result needs the **NVIDIA Container Toolkit** and a driver ≥ 580 (the CPU-only image needs neither). The builder is the pinned Ubuntu 22.04 CUDA 13 image [packaging/Dockerfile.linux](../packaging/Dockerfile.linux) names, which [packaging/package_linux_docker.sh](../packaging/package_linux_docker.sh) builds its release tarballs in, and the runtime stage is the same Ubuntu 22.04: the glibc the binary asks for is fixed by the image, so the host needs no particular glibc of its own — the 2.35 baseline is what the *tarballs* demand of a host that runs them directly. The image pins its runtime base by digest as well, and the build ends by checking both: the CUDA plugin is installed, and the newest glibc symbol the binary references is at or below 2.35.

```sh
docker build -f packaging/Dockerfile.server -t flyx:local .                                       # every release architecture
docker build -f packaging/Dockerfile.server -t flyx:local --build-arg FLYX_CUDA_ARCHS=89-real .    # one GPU: a fifth of the compiles
docker build -f packaging/Dockerfile.server -t flyx:local --build-arg FLYX_CUDA=OFF .              # the float64 reference backend

docker run --gpus all --init -p 127.0.0.1:8080:8080 -v flyx-data:/data flyx:local                  # the log prints the URL to open
docker run --gpus all --init -p 8080:8080 -e FLYX_API_KEY=secret -v flyx-data:/data flyx:local     # published to the network
```

`--init` matters for a container that will hold tens of gigabytes: PID 1 ignores SIGTERM while a model is downloading, before `flyx` installs its own handler, and `docker stop` then waits out the full grace period. The published port is the boundary — see the two consequences below — so a `-p 8080:8080` is a server on every interface of the machine, and `-p 127.0.0.1:8080:8080` is one you reach over an SSH tunnel.

Measured on a 16-job host: `86-real;89-real` builds in 12 minutes into a 218 MB image, 52 MB of it `/opt/flyx` (`flyx` and `flyx-server`); the release's six architectures cost about three times that. A CUDA server image that links the toolkit dynamically and ships it runs to gigabytes for the same server, which is what static linking buys here.

With Compose — the file lives in `packaging/`, so its build context is the repository (`context: ..`) and it reads `packaging/.env`, which is git-ignored and excluded from the build context by [packaging/Dockerfile.server.dockerignore](../packaging/Dockerfile.server.dockerignore):

```sh
cp packaging/.env.example packaging/.env            # optional: model, API key, GPU, architectures
docker compose -f packaging/docker-compose.yml up -d --build
docker compose -f packaging/docker-compose.yml logs -f flyx
curl http://127.0.0.1:8080/health
```

The compose file publishes `127.0.0.1:8080` and sets `init: true`, so the two `docker run` lines above are what you write by hand to change one of those choices.

`packaging/docker-entrypoint.sh` starts `flyx serve` from the environment, so a container needs no arguments; arguments given to `docker run` replace that line and keep the same binding (a port published by Docker reaches nothing bound to the container's own loopback). It binds `0.0.0.0` always and it supplies a key when nothing names one, which is what serving through a published port takes. Two consequences of that serving are worth knowing before the first `403`:

- a request that arrives through the port carries the bridge's gateway address rather than localhost, so the endpoints that fetch and load models answer `loading models is for this machine (use --api-key KEY, or --no-auth on a trusted LAN)` even from the machine itself;
- a browser names the machine in its `Host` header, and inside a container that name is none of the addresses the server trusts, so the dashboard's own reads are refused and the page reports `GET /v1/telemetry answered HTTP 403` under a `No model loaded` panel.

One key answers both. `FLYX_API_KEY` names it; with nothing set, the entrypoint generates one, keeps it in the volume as `docker-api-key` so a restart keeps the address a browser has bookmarked, and prints what to open — `flyx-server: the dashboard at http://127.0.0.1:8080/#key=…`. Open that address once: the dashboard takes the key from the fragment, remembers it, and sends `Authorization: Bearer` on every call after that. `--no-auth` (in `FLYX_ARGS` or in the container's arguments) declines the generated key, and the dashboard's remote-management toggle ([server.md](server.md)) opens management to the LAN; neither is an answer under bridge networking, because both still check `Host` against the container's own addresses — they are the keyless answer for `network_mode: host`, where the container has the machine's addresses and a LAN page's `Host` matches one of them.

| Variable | Effect |
|---|---|
| `FLYX_MODEL` | Served at startup: a Hugging Face name (`org/repo:QUANT`, fetched into the cache) or a file under `/models` by its full path. Unset starts unloaded; the dashboard then loads one. |
| `FLYX_ARGS` | More `flyx serve` flags, split on whitespace, no shell evaluation ([server.md](server.md) lists them). A value with a space in it belongs in the container's arguments instead. |
| `FLYX_PORT` | The port the server listens on and Compose publishes. Default 8080. |
| `FLYX_HOST` | The address the server binds *inside* the container. Default `0.0.0.0`, which a port mapping needs; it does not limit what the host publishes. |
| `FLYX_PUBLISH_ADDR` | Compose only: the host address the port is published on. Default `127.0.0.1`; `0.0.0.0` serves the network. |
| `FLYX_API_KEY` | Names the key the API and the dashboard use. Unset (and no `--no-auth`), the entrypoint generates one, keeps it in the volume as `docker-api-key`, and prints the URL with it. |
| `GPU_DEVICE` | Compose only: the card to serve, its number as `nvidia-smi` shows it, reserved by `device_ids`. Default `0`. |
| `CUDA_VISIBLE_DEVICES` | With `count: all` (every card visible to the container) this picks the card; with `device_ids` the container sees one card and the variable must name it `0`, so the compose file leaves it alone. |
| `HF_TOKEN`, `HF_ENDPOINT` | Gated repositories; a mirror's endpoint. |
| `MODELS_DIR` | A folder of your own `.gguf` files, bound read-only at `/models` (uncomment its volume line). |
| `CUDA_IMAGE`, `RUNTIME_IMAGE`, `FLYX_CUDA_ARCHS`, `FLYX_JOBS`, `FLYX_IMAGE_TAG` | The build: base images, architectures (80, 86, 89, 90, 100, 120), parallel jobs, the image's tag. |

CUDA's own device numbering orders devices by speed, so a card chosen as `1` today names a different card the day a second one is added. The compose file sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, which pins it to `nvidia-smi`'s order, and reserves the card by Compose's `device_ids`, which takes `nvidia-smi`'s number directly. A service that reserves by `device_ids` sees exactly one card, and there `CUDA_VISIBLE_DEVICES` must name it as `0` — the number from `nvidia-smi` fails the load with `no CUDA-capable device is detected`.

`memlock` is raised in the compose file as a knob, not a requirement. A long context's K/V in host RAM (`--kv-ram`, [models.md](models.md)) is allocated through CUDA — `cudaHostAlloc`, `cuMemCreate` — and no part of flyx reads `RLIMIT_MEMLOCK`. Measured: a server with `--ctx 65536 --kv-ram 4`, whose plan puts 3.69 GB of K/V in host memory, inside a container given `--ulimit memlock=67108864:67108864`, loaded in 17 s and served an 11,215-token prompt — fifty times the locked-memory limit it ran under. Docker's own default is 64 MiB and the NVIDIA Container Toolkit raises it, so on a GPU host the question rarely comes up; rootless Docker and Podman can refuse to raise hard limits, and a platform that rejects the `ulimits` block loses nothing.

