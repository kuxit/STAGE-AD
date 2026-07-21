[CmdletBinding()]
param(
    [string]$Python = 'C:\Users\wyyxx\.conda\envs\00gwk\python.exe',
    [string]$Gpu = '0',
    [int]$DpadEpochs = 3,
    [int]$DneEpochs = 3,
    [string]$Result
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$arguments = @(
    (Join-Path $PSScriptRoot 'smoke_aaai_recent.py'),
    '--project', $ProjectRoot,
    '--python', $Python,
    '--gpu', $Gpu,
    '--dpad-epochs', $DpadEpochs,
    '--dne-epochs', $DneEpochs
)
if (-not [string]::IsNullOrWhiteSpace($Result)) {
    $arguments += @('--result', $Result)
}
& $Python @arguments
exit $LASTEXITCODE
