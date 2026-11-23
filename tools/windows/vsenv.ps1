# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# The build environment on Windows, in the current PowerShell session:
#
#   . tools\windows\vsenv.ps1        # the leading dot matters: it changes this session
#   cmake --preset dev-windows; cmake --build --preset dev-windows; ctest --preset dev-windows
#
# Imports the x64 MSVC environment of Visual Studio 2022, finds a CUDA 13 compiler (an installed toolkit, else the one
# packaging\package_windows.ps1 installs from PyPI into ~\.venvs\flyx-win), puts the project's Python ahead of the
# Microsoft Store alias, and says what is still missing.  The tests' assets need torch, transformers, tokenizers,
# safetensors and gguf in that Python (docs/testing.md).
#
# Only building needs this; the release package needs nothing but the NVIDIA driver.

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $vswhere)) {
  throw ('Visual Studio 2022 Build Tools not found: https://aka.ms/vs/17/release/vs_buildtools.exe ' +
         '("Desktop development with C++")')
}
$vs = & $vswhere -latest -products * -version '[17.0,18.0)' `
  -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) {
  throw 'Visual Studio 2022 has no C++ x64 tools: add the "Desktop development with C++" workload'
}
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
cmd /s /c "`"$vcvars`" >nul 2>&1 && set" | ForEach-Object {
  if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}

# A terminal opened before the CUDA installer ran does not see its variables: take them from the machine's
# environment, else the newest v13 under the default place, else PyPI's compiler as the release build takes it.
if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
  $cuda = [Environment]::GetEnvironmentVariable('CUDA_PATH', 'Machine')
  if (-not $cuda) {
    $cuda = Get-ChildItem 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA' -Directory -Filter 'v13.*' `
      -ErrorAction SilentlyContinue | Sort-Object { [version]($_.Name.Substring(1)) } | Select-Object -Last 1 `
      | ForEach-Object { $_.FullName }
  }
  $pypi = Join-Path $(if ($env:FLYX_WIN_VENV) { $env:FLYX_WIN_VENV } else { "$env:USERPROFILE\.venvs\flyx-win" }) `
    'Lib\site-packages\nvidia\cu13'
  if ($cuda -and (Test-Path "$cuda\bin\nvcc.exe")) {
    $env:CUDA_PATH = $cuda
    $env:PATH = "$cuda\bin;$cuda\bin\x64;$env:PATH"
  } elseif (Test-Path "$pypi\bin\nvcc.exe") {
    $env:PATH = "$pypi\bin;$env:PATH"
  }
}

# Python: the project's virtual environment if there is one (~\.venvs\flyx or ~\.venvs\flyx-win;
# FLYX_VENV names another), else what the py launcher knows; the Store's python.exe in WindowsApps only opens the Store.
$venv = if ($env:FLYX_VENV) { $env:FLYX_VENV } else { Join-Path $env:USERPROFILE '.venvs\flyx' }
if (-not $env:FLYX_VENV -and -not (Test-Path "$venv\Scripts\python.exe")) {
  $win = Join-Path $env:USERPROFILE '.venvs\flyx-win'
  if (Test-Path "$win\Scripts\python.exe") { $venv = $win }
}
if (Test-Path "$venv\Scripts\python.exe") {
  $env:VIRTUAL_ENV = $venv
  $env:PATH = "$venv\Scripts;$env:PATH"
} elseif (Get-Command py -ErrorAction SilentlyContinue) {
  $py = & py -3 -c 'import sys; print(sys.executable)' 2>$null
  if ($py) {
    $dir = Split-Path $py
    $env:PATH = "$dir;$dir\Scripts;$env:PATH"
  }
}

Write-Host "flyx build environment (Visual Studio: $vs)"
$missing = @()
foreach ($tool in 'cl', 'cmake', 'ninja', 'nvcc', 'python') {
  $c = Get-Command $tool -ErrorAction SilentlyContinue
  if ($c -and $c.Source -notlike '*\WindowsApps\*') {
    Write-Host ('  {0,-7} {1}' -f $tool, $c.Source)
  } else {
    Write-Host ('  {0,-7} missing' -f $tool)
    $missing += $tool
  }
}
if ($missing -notcontains 'nvcc') {
  $release = (& nvcc --version | Select-String 'release').ToString().Trim()
  Write-Host "  nvcc:   $release"
} else {
  Write-Host '  without nvcc the build has the float64 backend only (packaging\package_windows.ps1 installs one)'
}
if ($missing -notcontains 'python') {
  Write-Host ('  python: ' + (& python -c 'import sys; print(sys.version.split()[0])'))
  $assets = "'torch', 'transformers', 'tokenizers', 'safetensors', 'gguf'"
  $absent = & python -c "import importlib.util as u; print(' '.join(m for m in ($assets) if not u.find_spec(m)))"
  if ($absent) { Write-Host "  the tests' assets also need: pip install $absent" }
}
