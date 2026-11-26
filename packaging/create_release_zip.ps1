# Copyright 2026 The flyx Authors. SPDX-License-Identifier: Apache-2.0
param([Parameter(Mandatory)][string]$Package, [Parameter(Mandatory)][string]$Destination)
$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.IO.Compression.FileSystem
$root = Get-Item -LiteralPath $Package
$zip = [IO.Compression.ZipFile]::Open($Destination, 'Create')
try {
  foreach ($item in Get-ChildItem -LiteralPath $root.FullName -Force -Recurse) {
    $entry = $root.Name + '/' + $item.FullName.Substring($root.FullName.Length + 1).Replace('\', '/')
    if ($item.PSIsContainer) {
      $null = $zip.CreateEntry($entry + '/')
    } else {
      $null = [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $item.FullName, $entry)
    }
  }
} finally {
  $zip.Dispose()
}
