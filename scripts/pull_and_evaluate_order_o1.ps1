param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteHost,

    [Parameter(Mandatory = $true)]
    [int]$RemotePort,

    [string]$RemoteUser = "root",
    [string]$IdentityFile = "C:\Users\wyyxx\.ssh\codex_stage_server_b",
    [string]$RemoteResultRoot = "/root/autodl-tmp/results/STAGE_majority_order_o1/seed2026",
    [string]$LocalResultRoot = "F:\python_project\AAAI\我们的论文\experiments\stage_majority_order_o1\seed2026",
    [string]$Python = "C:\Users\wyyxx\.conda\envs\00gwk\python.exe",
    [int]$EvaluatorWorkers = 8,
    [switch]$KeepCache
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Protocol = Join-Path $ProjectRoot "configs\stage_majority_order_o1.json"
$MetricsRoot = "F:\python_project\AAAI\STAGE-AD\external"

if (-not (Test-Path -LiteralPath $IdentityFile -PathType Leaf)) {
    throw "SSH identity file does not exist: $IdentityFile"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
New-Item -ItemType Directory -Force -Path $LocalResultRoot | Out-Null

$Remote = "$RemoteUser@$RemoteHost"
$CommonScp = @(
    "-P", "$RemotePort",
    "-i", $IdentityFile,
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new"
)

& scp.exe @CommonScp `
    "${Remote}:$RemoteResultRoot/stage_vuspr_search_plan.json" `
    $LocalResultRoot
if ($LASTEXITCODE -ne 0) {
    throw "Failed to pull the frozen O1 plan"
}

$RemoteCacheProbe = "test -d '$RemoteResultRoot/score_cache'"
& ssh.exe -p $RemotePort -i $IdentityFile `
    -o BatchMode=yes -o StrictHostKeyChecking=accept-new `
    $Remote $RemoteCacheProbe
if ($LASTEXITCODE -eq 0) {
    & scp.exe @CommonScp -r `
        "${Remote}:$RemoteResultRoot/score_cache" `
        $LocalResultRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to pull O1 score caches"
    }
}

$EvaluateArguments = @(
    (Join-Path $PSScriptRoot "stage_vuspr_search.py"),
    "--repo", $ProjectRoot,
    "--protocol", $Protocol,
    "--result-root", $LocalResultRoot,
    "--metrics-root", $MetricsRoot,
    "evaluate-caches",
    "--workers", "$EvaluatorWorkers"
)
if ($KeepCache) {
    $EvaluateArguments += "--keep-cache"
}

& $Python @EvaluateArguments
if ($LASTEXITCODE -ne 0) {
    throw "Local official-metric evaluation failed"
}
