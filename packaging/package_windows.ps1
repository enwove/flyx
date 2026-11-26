# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
#
# Build the Windows release package and check it on this machine (Windows PowerShell 5.1 or 7):
#
#   powershell -ExecutionPolicy Bypass -File packaging\package_windows.ps1 [-Assets DIR] [-BuildDir DIR] [-Jobs N] [-Exe]
#
#   dist\flyx-<version>-windows-x86_64-cuda13.zip
#       bin\flyx.exe (the command line; `flyx serve` the HTTP API), lib\flyx.dll (C API, flyx.lib to link it),
#       lib\flyx\ backends, include\, python\; needs only Windows and the NVIDIA driver (>= 580)
#   dist\flyx-<version>-py3-none-win_amd64.whl
#       the ctypes binding with the same DLLs inside (no torch)
#   dist\flyx-<version>-windows-x86_64-cuda13.exe  (with -Exe)
#       single-file standalone executable with both backends linked in; double-clicking opens the dashboard
#
# Needs Visual Studio 2022 or newer Build Tools (C++), CMake >= 3.24, Ninja and Python 3.11+.  The CUDA 13.0 compiler is
# taken from a virtual environment ($env:FLYX_WIN_VENV, default ~\.venvs\flyx-win) and installed into it from PyPI's
# wheels when it is not there.  The smoke test unpacks the zip into an empty directory, checks the DLLs it needs
# (Windows' own), and with -Assets (a directory holding tiny.flyx, from a test build: <build>/assets) runs the CLI on
# both backends and the server, and the wheel in a virtual environment of its own on both backends.
param([string]$Assets = "", [int]$Jobs = 0, [string]$BuildDir = "$env:USERPROFILE\flyx-build-release", [switch]$Exe)
$ErrorActionPreference = "Stop"
$Build = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($BuildDir)
if ($Assets) { $Assets = (Resolve-Path -LiteralPath $Assets -ErrorAction Stop).Path }
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Version = (Select-String -Path CMakeLists.txt -Pattern 'project\(flyx VERSION ([0-9.]+)').Matches[0].Groups[1].Value
$Name = "flyx-$Version-windows-x86_64-cuda13"
$Dist = (New-Item -ItemType Directory -Force (Join-Path $Root "dist")).FullName

# MSVC: the environment vcvars64.bat makes, taken into this process
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "no Visual Studio with the C++ tools (vswhere found none)" }
cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" > nul && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}

# CUDA 13.0's compiler from PyPI, in its own virtual environment
$Venv = if ($env:FLYX_WIN_VENV) { $env:FLYX_WIN_VENV } else { "$env:USERPROFILE\.venvs\flyx-win" }
$Cu = "$Venv\Lib\site-packages\nvidia\cu13"
if (-not (Test-Path "$Cu\bin\nvcc.exe")) {
  if (-not (Test-Path "$Venv\Scripts\python.exe")) {
    python -m venv $Venv
    if ($LASTEXITCODE) { throw "creating the CUDA build environment failed" }
  }
  & "$Venv\Scripts\python.exe" -m pip install -q "nvidia-cuda-nvcc==13.0.88" "nvidia-nvvm==13.0.88" `
    "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-cccl==13.0.85" "nvidia-cuda-crt==13.0.88"
  if ($LASTEXITCODE) { throw "installing the CUDA build dependencies failed" }
}
$env:PATH = "$Cu\bin;$env:PATH"

Write-Host "== configure ($Build)"
$StandaloneOpt = if ($Exe) { "ON" } else { "OFF" }
cmake --preset release-windows -B $Build "-DCMAKE_CUDA_COMPILER=$Cu\bin\nvcc.exe" "-DPython3_EXECUTABLE=$Venv\Scripts\python.exe" "-DFLYX_STANDALONE_EXE=$StandaloneOpt" | Out-Null
if ($LASTEXITCODE) { throw "configure failed" }
Write-Host "== build"
if ($Jobs -gt 0) { cmake --build $Build -j $Jobs } else { cmake --build $Build }
if ($LASTEXITCODE) { throw "build failed" }
# Install and verify a fresh candidate on the same volume as the published ZIP.
$Work = (New-Item -ItemType Directory -Path (Join-Path $Dist ("package-" + [Guid]::NewGuid()))).FullName
try {
  $Pkg = Join-Path $Work $Name
  Write-Host "== install -> $Pkg"
  cmake --install $Build --prefix $Pkg | Out-Null
  if ($LASTEXITCODE) { throw "install failed" }

  Write-Host "== zip"
  $Zip = Join-Path $Work "$Name.zip"
  Compress-Archive -LiteralPath $Pkg -DestinationPath $Zip
  $Test = Join-Path $Work "unpacked"
  Expand-Archive -LiteralPath $Zip -DestinationPath $Test
  $P = Join-Path $Test $Name

  Write-Host "== smoke test"
  foreach ($required in 'bin\flyx.exe', 'lib\flyx.dll', 'lib\flyx.lib', 'lib\flyx\libflyx-cpu_ref.dll',
                        'lib\flyx\libflyx-cuda.dll', 'include\flyx\flyx.h', 'python\flyx\__init__.py', 'LICENSE', 'NOTICE') {
    if (-not (Test-Path -LiteralPath (Join-Path $P $required) -PathType Leaf)) { throw "package is missing $required" }
  }
  # CUDA and the C/C++ runtimes are linked in. The backend loads the driver and NVML when it starts; DXGI tells it
  # what the process holds of the GPU (backend ABI 37); USERENV supplies CreateEnvironmentBlock (os_windows.cpp).
  $system = '^(KERNEL32|USER32|ADVAPI32|USERENV|SHELL32|WS2_32|WINHTTP|ole32|OLEAUT32|bcrypt|CRYPT32|SHLWAPI|GDI32|IPHLPAPI|DXGI|api-ms-win-.*)\.dll$'
  foreach ($f in Get-ChildItem -LiteralPath $P -Recurse -File | Where-Object { $_.Extension -in '.exe', '.dll' }) {
    $deps = dumpbin /nologo /dependents $f.FullName | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } | ForEach-Object { $_.Trim() }
    if ($LASTEXITCODE -or -not $deps) { throw "could not inspect imports of $($f.Name)" }
    $other = $deps | Where-Object { $_ -notmatch $system -and $_ -ne 'flyx.dll' }
    if ($other) { throw "$($f.Name) needs: $($other -join ', ')" }
  }
  Write-Host "  ok   only Windows' own DLLs are needed"
  if ($Assets) {
    & "$Venv\Scripts\python.exe" "$PSScriptRoot\smoke_model.py" "$P\bin\flyx.exe" "$Assets\tiny.flyx"
    if ($LASTEXITCODE) { throw "model smoke test failed" }
  } else {
    Write-Host "  --   no -Assets: the CLI and the server not run"
  }

  Write-Host "== wheel"
  # The ctypes binding with the same DLLs inside, as the Linux package's (flyx\_lib\flyx.dll, its backends in
  # flyx\_lib\flyx\, where the engine looks beside the C API's library), with the package's notices.
  $W = Join-Path $Work "wheel"
  New-Item -ItemType Directory -Force "$W\flyx\_lib\flyx" | Out-Null
  Copy-Item "$Root\python\flyx\*.py" "$W\flyx\"
  Copy-Item -LiteralPath "$Pkg\lib\flyx.dll" "$W\flyx\_lib\"
  Copy-Item "$Pkg\lib\flyx\*.dll" "$W\flyx\_lib\flyx\"
  Copy-Item -LiteralPath "$Pkg\LICENSE", "$Pkg\NOTICE", "$Root\README.md" $W
  Copy-Item -Recurse -LiteralPath "$Pkg\third_party" $W
  @"
[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "flyx"
version = "$Version"
description = "Local LLM inference engine: lowest single-user latency on one consumer GPU"
readme = "README.md"
license = "Apache-2.0"
license-files = ["LICENSE", "NOTICE", "third_party/**/LICENSE*", "third_party/**/NOTICE",
                 "third_party/**/README.*.md", "third_party/**/picojson.h"]
requires-python = ">=3.9"
keywords = ["llm", "inference", "cuda", "gguf", "speculative-decoding"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/enwove/flyx"
Repository = "https://github.com/enwove/flyx"
Issues = "https://github.com/enwove/flyx/issues"

[tool.setuptools]
packages = ["flyx"]
include-package-data = true

[tool.setuptools.package-data]
flyx = ["_lib/*.dll", "_lib/flyx/*.dll"]
"@ | Out-File -Encoding ascii "$W\pyproject.toml"
  $Py = "$Venv\Scripts\python.exe"
  & $Py -m pip install -q "setuptools>=77" wheel
  if ($LASTEXITCODE) { throw "installing the wheel's build dependencies failed" }
  & $Py -m pip wheel --no-deps --no-build-isolation -q -w "$W\out" $W
  if ($LASTEXITCODE) { throw "wheel failed" }
  $Whl = (Get-ChildItem -LiteralPath "$W\out" -Filter *.whl)[0].FullName
  # The DLLs are platform specific, the Python code is not: py3-none-win_amd64 in the file's name and in its WHEEL
  # metadata, as auditwheel tags the Linux wheel; the file tested below is the one published.
  $Tagged = & $Py -m wheel tags --platform-tag win_amd64 --remove $Whl | Select-Object -Last 1
  if ($LASTEXITCODE) { throw "tagging the wheel failed" }
  $Whl = Join-Path "$W\out" "$Tagged"
  if (-not (Test-Path -LiteralPath $Whl) -or "$Tagged" -notmatch '-py3-none-win_amd64\.whl$') {
    throw "tagging the wheel: $Tagged"
  }
  if ($Assets) {
    # The wheel in a virtual environment of its own: the binding it brings, with its own DLLs, and the same tokens
    # from both backends (the float64 one alone where the CUDA backend is not usable here, as the CLI's check allows).
    $V = Join-Path $Work "venv"
    & $Py -m venv $V
    if ($LASTEXITCODE) { throw "creating the wheel's test environment failed" }
    & "$V\Scripts\python.exe" -m pip install -q --no-index --no-deps $Whl
    if ($LASTEXITCODE) { throw "installing the wheel failed" }
    & "$V\Scripts\python.exe" -I "$PSScriptRoot\smoke_binding.py" "$V\Lib\site-packages" "$Assets\tiny.flyx"
    if ($LASTEXITCODE) { throw "wheel smoke test failed" }
    $check = "import flyx, sys; t = {}`nfor b in ('cpu_ref', 'cuda'):`n" +
             "    try: e = flyx.Engine(sys.argv[1], backend=b, max_ctx=512)`n" +
             "    except flyx.FlyxError: continue`n" +
             "    t[b] = e.session().generate(text='hello', max_new_tokens=8); e.close()`n" +
             "print('no backend' if 'cpu_ref' not in t else 'cpu_ref only' if len(t) == 1 else " +
             "'both backends, the same tokens' if t['cpu_ref'] == t['cuda'] else 'both backends, DIFFERENT tokens')"
    $out = & "$V\Scripts\python.exe" -I -c $check "$Assets\tiny.flyx" | Select-Object -Last 1
    if ("$out" -match 'the same tokens') { Write-Host "  ok   wheel: $out" }
    elseif ("$out" -match 'cpu_ref only') { Write-Host "  ok   wheel: float64 backend (CUDA backend not usable here)" }
    else { throw "wheel: $out" }
  } else {
    Write-Host "  --   no -Assets: the wheel not run"
  }

  $Standalone = $null
  if ($Exe) {
    Write-Host "== standalone exe"
    $BuiltExe = Join-Path $Build "bin\flyx-standalone.exe"
    if (-not (Test-Path -LiteralPath $BuiltExe -PathType Leaf)) { throw "missing standalone executable: $BuiltExe" }
    $StandaloneDir = (New-Item -ItemType Directory -Force (Join-Path $Work "standalone")).FullName
    $Standalone = Join-Path $StandaloneDir "flyx.exe"
    Copy-Item -LiteralPath $BuiltExe -Destination $Standalone
    $deps = dumpbin /nologo /dependents $Standalone | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } | ForEach-Object { $_.Trim() }
    if ($LASTEXITCODE -or -not $deps) { throw "could not inspect imports of standalone flyx.exe" }
    $other = $deps | Where-Object { $_ -notmatch $system }
    if ($other) { throw "standalone flyx.exe needs: $($other -join ', ')" }
    Write-Host "  ok   standalone exe needs only Windows' own DLLs"
    if ($Assets) {
      & "$Venv\Scripts\python.exe" "$PSScriptRoot\smoke_model.py" --no-binding $Standalone "$Assets\tiny.flyx"
      if ($LASTEXITCODE) { throw "standalone exe smoke test failed" }
    } else {
      Write-Host "  --   no -Assets: standalone exe not run"
    }
  }

  $Published = Join-Path $Dist "$Name.zip"
  if ([IO.File]::Exists($Published)) { [IO.File]::Replace($Zip, $Published, [NullString]::Value) }
  else { [IO.File]::Move($Zip, $Published) }
  Write-Host ("   $Published ({0:N1} MB)" -f ((Get-Item -LiteralPath $Published).Length / 1MB))
  $PublishedWhl = Join-Path $Dist (Split-Path -Leaf $Whl)
  if ([IO.File]::Exists($PublishedWhl)) { [IO.File]::Replace($Whl, $PublishedWhl, [NullString]::Value) }
  else { [IO.File]::Move($Whl, $PublishedWhl) }
  Write-Host ("   $PublishedWhl ({0:N1} MB)" -f ((Get-Item -LiteralPath $PublishedWhl).Length / 1MB))
  if ($Standalone) {
    $PublishedExe = Join-Path $Dist "$Name.exe"
    if ([IO.File]::Exists($PublishedExe)) { [IO.File]::Replace($Standalone, $PublishedExe, [NullString]::Value) }
    else { [IO.File]::Move($Standalone, $PublishedExe) }
    Write-Host ("   $PublishedExe ({0:N1} MB)" -f ((Get-Item -LiteralPath $PublishedExe).Length / 1MB))
  }
} finally {
  if ([IO.Path]::GetDirectoryName($Work) -ne $Dist) { throw "unexpected package directory: $Work" }
  Remove-Item -LiteralPath $Work -Recurse -Force
}
