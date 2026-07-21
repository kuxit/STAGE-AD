[CmdletBinding()]
param(
    [string]$Python = 'C:\Users\wyyxx\.conda\envs\00gwk\python.exe',
    [string]$Tracks = 'U,M',
    [string]$Methods,
    [string]$Gpu = '0',
    [string]$Result
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$arguments = @(
    (Join-Path $PSScriptRoot 'smoke_local.py'),
    '--project', $ProjectRoot,
    '--python', $Python,
    '--tracks', $Tracks,
    '--gpu', $Gpu
)
if (-not [string]::IsNullOrWhiteSpace($Methods)) {
    $arguments += @('--methods', $Methods)
}
if (-not [string]::IsNullOrWhiteSpace($Result)) {
    $arguments += @('--result', $Result)
}
& $Python @arguments
exit $LASTEXITCODE
