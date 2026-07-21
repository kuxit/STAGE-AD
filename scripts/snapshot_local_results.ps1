[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceRoot,

    [string]$BackupRoot,

    [int]$Seed = 2026
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($BackupRoot)) {
    $BackupRoot = Join-Path $ProjectRoot 'results\backups'
}

$SourceRoot = [IO.Path]::GetFullPath($SourceRoot)
$BackupRoot = [IO.Path]::GetFullPath($BackupRoot)
$UnitsRoot = Join-Path $SourceRoot 'units'
if (-not (Test-Path -LiteralPath $UnitsRoot -PathType Container)) {
    throw "Source does not contain a units directory: $UnitsRoot"
}

$metricNames = @('VUS-PR', 'VUS-ROC', 'R-based-F1', 'AUC-PR', 'AUC-ROC', 'Standard-F1')

function Test-FiniteNumber {
    param($Value)
    if ($null -eq $Value) {
        return $false
    }
    $number = 0.0
    $ok = [double]::TryParse(
        [string]$Value,
        [Globalization.NumberStyles]::Float,
        [Globalization.CultureInfo]::InvariantCulture,
        [ref]$number
    )
    return $ok -and -not [double]::IsNaN($number) -and -not [double]::IsInfinity($number)
}

$validated = New-Object System.Collections.Generic.List[object]
$problems = New-Object System.Collections.Generic.List[string]
$unitFiles = @(Get-ChildItem -LiteralPath $UnitsRoot -Recurse -File -Filter '*.json')
if ($unitFiles.Count -eq 0) {
    throw "No unit JSON files found under $UnitsRoot"
}

foreach ($unitFile in $unitFiles) {
    try {
        $unit = Get-Content -LiteralPath $unitFile.FullName -Raw | ConvertFrom-Json
        $ok = ([int]$unit.seed -eq $Seed) -and ($null -eq $unit.error)
        foreach ($metricName in $metricNames) {
            $metricProperty = $unit.metrics.PSObject.Properties[$metricName]
            if ($null -eq $metricProperty -or -not (Test-FiniteNumber $metricProperty.Value)) {
                $ok = $false
            }
        }
        if (-not $ok) {
            $problems.Add("Invalid unit: $($unitFile.FullName)")
            continue
        }
        $relativePath = $unitFile.FullName.Substring($UnitsRoot.Length).TrimStart([char[]]'\/')
        $validated.Add([pscustomobject]@{
            Source       = $unitFile.FullName
            RelativePath = $relativePath
            Method       = [string]$unit.method
            Track        = [string]$unit.track
            Dataset      = [string]$unit.dataset
            File         = [string]$unit.file
        })
    }
    catch {
        $problems.Add("Unreadable unit: $($unitFile.FullName)")
    }
}

$duplicates = @($validated | Group-Object Method, Track, Dataset, File | Where-Object Count -ne 1)
if ($duplicates.Count -gt 0) {
    $problems.Add("Duplicate unit identities: $($duplicates.Count)")
}
if ($problems.Count -gt 0) {
    throw "Snapshot aborted before writing. $($problems -join '; ')"
}

$sourceName = (Split-Path -Leaf $SourceRoot) -replace '[^A-Za-z0-9._-]', '_'
$timestamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ')
$snapshotName = "${sourceName}_${timestamp}"
$snapshotRoot = Join-Path $BackupRoot $snapshotName
$zipPath = "$snapshotRoot.zip"
if ((Test-Path -LiteralPath $snapshotRoot) -or (Test-Path -LiteralPath $zipPath)) {
    throw "Refusing to overwrite an existing snapshot: $snapshotRoot"
}

New-Item -ItemType Directory -Path $snapshotRoot -Force | Out-Null
$destinationUnits = Join-Path $snapshotRoot 'units'

foreach ($item in $validated) {
    $destination = Join-Path $destinationUnits $item.RelativePath
    $destinationParent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    Copy-Item -LiteralPath $item.Source -Destination $destination
}

$metadataNames = @('run_manifest.json', 'protocol.json')
foreach ($metadataName in $metadataNames) {
    $metadataPath = Join-Path $SourceRoot $metadataName
    if (Test-Path -LiteralPath $metadataPath -PathType Leaf) {
        Copy-Item -LiteralPath $metadataPath -Destination (Join-Path $snapshotRoot $metadataName)
    }
}
foreach ($pattern in @('*.csv', '*.md')) {
    foreach ($metadataFile in @(Get-ChildItem -LiteralPath $SourceRoot -File -Filter $pattern)) {
        Copy-Item -LiteralPath $metadataFile.FullName -Destination (Join-Path $snapshotRoot $metadataFile.Name)
    }
}

$inventory = New-Object System.Collections.Generic.List[object]
foreach ($item in $validated) {
    $copiedPath = Join-Path $destinationUnits $item.RelativePath
    $inventory.Add([pscustomobject]@{
        path   = ('units/' + ($item.RelativePath -replace '\\', '/'))
        sha256 = (Get-FileHash -LiteralPath $copiedPath -Algorithm SHA256).Hash.ToLowerInvariant()
        method = $item.Method
        track  = $item.Track
        dataset = $item.Dataset
        file   = $item.File
    })
}

$backupManifest = [ordered]@{
    schema_version = 'stage-metrics-backup-v1'
    created_utc = (Get-Date).ToUniversalTime().ToString('o')
    source_root = $SourceRoot
    seed = $Seed
    valid_unit_count = $validated.Count
    invalid_unit_count = 0
    runtime_eligible_for_paper = $false
    note = 'Accuracy-metric backup. Shared-resource runtime values are not valid for the paper efficiency table.'
    units = $inventory
}
$manifestPath = Join-Path $snapshotRoot 'BACKUP_MANIFEST.json'
$backupManifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

Compress-Archive -Path (Join-Path $snapshotRoot '*') -DestinationPath $zipPath -CompressionLevel Optimal
$zipHash = (Get-FileHash -LiteralPath $zipPath -Algorithm SHA256).Hash.ToLowerInvariant()

[pscustomobject]@{
    SnapshotDirectory = $snapshotRoot
    ZipArchive = $zipPath
    ZipSha256 = $zipHash
    ValidUnits = $validated.Count
    InvalidUnits = 0
} | Format-List
