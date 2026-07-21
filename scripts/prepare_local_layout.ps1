[CmdletBinding()]
param(
    [string]$DataSource,
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
if ([string]::IsNullOrWhiteSpace($DataSource)) {
    $workspaceRoot = Split-Path -Parent $ProjectRoot
    $DataSource = Join-Path $workspaceRoot 'phasead_pipeline\data'
}
$DataSource = [IO.Path]::GetFullPath($DataSource)
$dataDestination = Join-Path $ProjectRoot 'data'

foreach ($required in @(
    'File_List\TSB-AD-U-Eva.csv',
    'File_List\TSB-AD-M-Eva.csv',
    'TSB-AD-U',
    'TSB-AD-M'
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $DataSource $required))) {
        throw "Data source is incomplete; missing $required under $DataSource"
    }
}

if (Test-Path -LiteralPath $dataDestination) {
    Write-Host "READY data path already exists: $dataDestination"
}
else {
    New-Item -ItemType Junction -Path $dataDestination -Target $DataSource | Out-Null
    Write-Host "READY local data junction: $dataDestination -> $DataSource"
}

& (Join-Path $PSScriptRoot 'bootstrap_sources.ps1') -ProjectRoot $ProjectRoot
