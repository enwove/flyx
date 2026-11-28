# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
# A release's Windows packages on a machine without a GPU (.github/workflows/release-smoke.yml runs this on a hosted
# runner):
#
#   pwsh tools/ci/release_smoke_windows.ps1 -Exe EXE -Zip ZIP -Wheel WHEEL -Model tiny.flyx -Results results.tsv `
#     -Tag v0.2.2
#
# EXE is the single-file executable, ZIP the package (bin\flyx.exe, lib\flyx.dll and the backends, python\), WHEEL the
# Windows wheel, MODEL the tiny test model (tools/ci/tiny_model.py), TAG the release's tag.  The three files' names
# say the tag's version; then every check runs whatever the one before it gave, for the executable and for the
# package's flyx.exe alike:
#   - the DLLs it imports are Windows' own (no Visual C++ runtime, no CUDA DLL, no driver DLL: dumpbin /dependents);
#   - flyx --version says the tag's version; flyx --help;
#   - a run on the CUDA backend is refused with a reason, not a crash (no driver here);
#   - a run on the default backend: from 0.2.2 on it is refused too, naming --backend cpu_ref (a CUDA package never
#     falls back to the float64 reference on its own); before 0.2.2 it fell back and gave a token;
#   - --backend cpu_ref gives a first token of the tiny model (timed);
#   - its server answers a completion (packaging/smoke_model.py; the package's Python binding too);
# then the wheel installed by pip into a new virtual environment generates (packaging/smoke_binding.py).
# RESULTS gets a line a check: check<TAB>ok|FAIL|expected|skipped<TAB>detail.  Exits 1 when a check failed.
param(
  [Parameter(Mandatory)] [string] $Exe,
  [Parameter(Mandatory)] [string] $Zip,
  [Parameter(Mandatory)] [string] $Wheel,
  [Parameter(Mandatory)] [string] $Model,
  [Parameter(Mandatory)] [string] $Results,
  [Parameter(Mandatory)] [string] $Tag
)
$ErrorActionPreference = 'Stop'
$Here = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Exe, $Zip, $Wheel, $Model = ($Exe, $Zip, $Wheel, $Model) | ForEach-Object { (Resolve-Path $_).Path }
$Work = Join-Path ([IO.Path]::GetTempPath()) ("flyx-smoke-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force $Work | Out-Null
Set-Content -Path $Results -Value $null -Encoding utf8NoBOM
$script:failed = $false
$Version = $Tag -replace '^v', ''
# From 0.2.2 on a CUDA package whose CUDA backend cannot run says so instead of falling back to cpu_ref.
$RefusesCudaByDefault = [version]$Version -ge [version]'0.2.2'

function Write-Result([string] $Check, [string] $Status, [string] $Detail) {
  $line = ($Detail -replace '\s+', ' ').Trim()
  if ($line.Length -gt 400) { $line = $line.Substring(0, 400) }
  Add-Content -Path $Results -Value "$Check`t$Status`t$line" -Encoding utf8NoBOM
  Write-Host "$Check`t$Status`t$line"
  if ($Status -eq 'FAIL') { $script:failed = $true }
}

# A program's exit code, output and time, its arguments passed as they are (no shell between).
function Invoke-Program([string] $File, [string[]] $Arguments) {
  $psi = [Diagnostics.ProcessStartInfo]::new($File)
  foreach ($a in $Arguments) { $psi.ArgumentList.Add($a) }
  $psi.RedirectStandardOutput = $true
  $psi.RedirectStandardError = $true
  $psi.UseShellExecute = $false
  $psi.WorkingDirectory = $Work
  $clock = [Diagnostics.Stopwatch]::StartNew()
  $p = [Diagnostics.Process]::Start($psi)
  $out = $p.StandardOutput.ReadToEndAsync()
  $err = $p.StandardError.ReadToEndAsync()
  if (-not $p.WaitForExit(120000)) { $p.Kill($true) }
  $p.WaitForExit()
  [pscustomobject]@{ Code = $p.ExitCode; Out = $out.Result; Err = $err.Result; Ms = $clock.ElapsedMilliseconds }
}

# An NTSTATUS (0xC0000005 access violation, 0xC0000135 a DLL not found, 0xC0000409 a failed check) or abort()'s 3.
function Test-Crashed($Run) { $Run.Code -lt 0 -or $Run.Code -eq 3 }

$os = Get-CimInstance Win32_OperatingSystem
$gpus = (Get-CimInstance Win32_VideoController | ForEach-Object Name) -join ', '
Write-Result 'system' 'ok' "$($os.Caption) $($os.Version), display adapters: $gpus"

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$dumpbin = $null
if (Test-Path $vswhere) {
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if ($vs) {
    $dumpbin = Get-ChildItem (Join-Path $vs 'VC\Tools\MSVC\*\bin\Hostx64\x64\dumpbin.exe') -ErrorAction SilentlyContinue |
      Select-Object -Last 1 -ExpandProperty FullName
  }
}

function Test-Cli([string] $Name, [string] $Flyx, [string[]] $SmokeArgs) {
  if ($dumpbin) {
    $deps = & $dumpbin /nologo /dependents $Flyx | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } | ForEach-Object { $_.Trim() }
    $bad = $deps | Where-Object { $_ -match '^(vcruntime|msvcp|concrt|nvcuda|cudart|nvml)' }
    if ($bad) { Write-Result "$Name`: DLLs it imports" 'FAIL' "needs $($bad -join ', ') (all: $($deps -join ', '))" }
    else { Write-Result "$Name`: DLLs it imports" 'ok' ($deps -join ', ') }
  } else { Write-Result "$Name`: DLLs it imports" 'skipped' 'no dumpbin on this machine' }

  $r = Invoke-Program $Flyx @('--version')
  $starts = $r.Code -eq 0
  $says = $r.Out.Trim()
  if (-not $starts) { Write-Result "$Name`: flyx --version" 'FAIL' "exit $($r.Code): $($r.Out)$($r.Err)" }
  elseif ($says -ne "flyx $Version") { Write-Result "$Name`: flyx --version" 'FAIL' "says '$says', the tag $Tag" }
  else { Write-Result "$Name`: flyx --version" 'ok' $says }
  $r = Invoke-Program $Flyx @('--help')
  Write-Result "$Name`: flyx --help" ($r.Code -eq 0 ? 'ok' : 'FAIL') "exit $($r.Code): $((($r.Out + $r.Err) -split "`n")[0])"

  $r = Invoke-Program $Flyx @('run', $Model, '-p', 'hello', '-n', '1', '--ctx', '128', '--backend', 'cuda')
  $said = "exit $($r.Code): $($r.Out)$($r.Err)"
  if (-not $starts) { Write-Result "$Name`: no GPU, --backend cuda" 'FAIL' 'flyx does not start here (flyx --version)' }
  elseif (Test-Crashed $r) { Write-Result "$Name`: no GPU, --backend cuda" 'FAIL' "crashed, $said" }
  elseif ($r.Code -eq 0) { Write-Result "$Name`: no GPU, --backend cuda" 'FAIL' "ran on CUDA without a GPU, $said" }
  elseif (($r.Out + $r.Err) -notmatch 'no usable backend') { Write-Result "$Name`: no GPU, --backend cuda" 'FAIL' $said }
  else { Write-Result "$Name`: no GPU, --backend cuda" 'expected' $said }

  # The default backend (cuda, the backend a CUDA package ships): from 0.2.2 on, one that cannot run here is refused
  # with the reason and the way to the float64 reference, and the CLI never falls back to it on its own (a real model
  # would run a minute a prompt there).  Before 0.2.2 the CLI fell back to it and generated.
  $r = Invoke-Program $Flyx @('run', $Model, '-p', 'hello', '-n', '1', '--ctx', '128')
  $said = "exit $($r.Code): $($r.Out)$($r.Err)"
  if (-not $starts) { Write-Result "$Name`: no GPU, default backend" 'FAIL' 'flyx does not start here' }
  elseif (Test-Crashed $r) { Write-Result "$Name`: no GPU, default backend" 'FAIL' "crashed, $said" }
  elseif (-not $RefusesCudaByDefault) {
    Write-Result "$Name`: no GPU, default backend" ($r.Code -eq 0 ? 'ok' : 'FAIL') "fell back to cpu_ref (before 0.2.2), $said"
  }
  elseif ($r.Code -eq 0) { Write-Result "$Name`: no GPU, default backend" 'FAIL' "generated without a GPU, $said" }
  elseif (-not ($r.Err.Contains('cuda backend is not usable here') -and $r.Err.Contains('--backend cpu_ref'))) {
    Write-Result "$Name`: no GPU, default backend" 'FAIL' $said
  } else { Write-Result "$Name`: no GPU, default backend" 'expected' $said }

  # The float64 backend named: a first token of the tiny model, timed from the process's start to its exit after one
  # token (loading the model, the prompt, the token).
  $r = Invoke-Program $Flyx @('run', $Model, '-p', 'hello', '-n', '1', '--ctx', '128', '--backend', 'cpu_ref')
  if (-not $starts) { Write-Result "$Name`: first token (--backend cpu_ref)" 'FAIL' 'flyx does not start here' }
  elseif ((Test-Crashed $r) -or $r.Code -ne 0) {
    Write-Result "$Name`: first token (--backend cpu_ref)" 'FAIL' "exit $($r.Code): $($r.Err)"
  } elseif (-not $r.Err.Contains('| 1 tokens in ')) {  # its closing statistics: one token made
    Write-Result "$Name`: first token (--backend cpu_ref)" 'FAIL' "no token; stderr: $($r.Err)"
  } else {
    Write-Result "$Name`: first token (--backend cpu_ref)" 'ok' "$($r.Ms) ms, text '$($r.Out.Trim())'; stderr: $($r.Err)"
  }

  $out = & python (Join-Path $Here 'packaging\smoke_model.py') $Flyx $Model --backend cpu_ref @SmokeArgs 2>&1 | Out-String
  if ($LASTEXITCODE -eq 0) {
    Write-Result "$Name`: server$($SmokeArgs ? '' : ' and Python binding')" 'ok' (($out -split "`n" | Where-Object { $_ -match '^\s+ok' }) -join '; ')
  } else {
    Write-Result "$Name`: server$($SmokeArgs ? '' : ' and Python binding')" 'FAIL' (($out -split "`n" | Select-Object -Last 4) -join ' ')
  }
}

$names = ($Exe, $Zip, $Wheel | ForEach-Object { Split-Path -Leaf $_ }) -join ', '
if (@($Exe, $Zip, $Wheel | Where-Object { -not (Split-Path -Leaf $_).StartsWith("flyx-$Version-") }).Count) {
  Write-Result 'package names' 'FAIL' "$names, the tag $Tag"
} else { Write-Result 'package names' 'ok' $names }

Test-Cli 'exe' $Exe @('--no-binding')

$unpacked = Join-Path $Work 'zip'
try {
  Expand-Archive -Path $Zip -DestinationPath $unpacked
  $flyx = Get-ChildItem $unpacked -Recurse -Filter flyx.exe | Select-Object -First 1 -ExpandProperty FullName
  if (-not $flyx) { throw 'no flyx.exe in it' }
  Write-Result 'zip: unpack' 'ok' "Expand-Archive: $($flyx.Substring($unpacked.Length + 1))"
  Test-Cli 'zip' $flyx @()
} catch {
  Write-Result 'zip: unpack' 'FAIL' "$_"
}

$venv = Join-Path $Work 'venv'
& python -m venv $venv
$py = Join-Path $venv 'Scripts\python.exe'
$out = & $py -m pip install --disable-pip-version-check -q $Wheel 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
  Write-Result 'wheel: pip install' 'FAIL' (($out -split "`n" | Where-Object { $_ -match 'error' } | Select-Object -Last 2) -join ' ')
} else {
  Write-Result 'wheel: pip install' 'ok' "$(& $py -m pip --version) installed $(Split-Path -Leaf $Wheel)"
  $site = & $py -c "import sysconfig; print(sysconfig.get_paths()['platlib'])"
  $out = & $py -I (Join-Path $Here 'packaging\smoke_binding.py') $site $Model 2>&1 | Out-String
  Write-Result 'wheel: generate' ($LASTEXITCODE -eq 0 ? 'ok' : 'FAIL') $out
}

Remove-Item -Recurse -Force $Work -ErrorAction SilentlyContinue
exit ($script:failed ? 1 : 0)
