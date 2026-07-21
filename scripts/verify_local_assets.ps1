[CmdletBinding()]
param(
    [string]$WorkspaceRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $WorkspaceRoot = Split-Path -Parent $ProjectRoot
}
$WorkspaceRoot = [IO.Path]::GetFullPath($WorkspaceRoot)
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)

$checks = New-Object System.Collections.Generic.List[object]
$failed = $false

function Add-Check {
    param(
        [string]$Name,
        [bool]$Ok,
        [string]$Detail
    )
    $script:checks.Add([pscustomobject]@{
        Check  = $Name
        Status = $(if ($Ok) { 'PASS' } else { 'FAIL' })
        Detail = $Detail
    })
    if (-not $Ok) {
        $script:failed = $true
    }
}

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

try {
    $checksumFile = Join-Path $ProjectRoot 'checksums\frozen-code.sha256'
    if (-not (Test-Path -LiteralPath $checksumFile -PathType Leaf)) {
        Add-Check 'Frozen checksums' $false "Missing $checksumFile"
    }
    else {
        $hashProblems = New-Object System.Collections.Generic.List[string]
        $hashCount = 0
        foreach ($line in Get-Content -LiteralPath $checksumFile) {
            if ([string]::IsNullOrWhiteSpace($line) -or $line.TrimStart().StartsWith('#')) {
                continue
            }
            if ($line -notmatch '^([0-9a-fA-F]{64})\s+(.+)$') {
                $hashProblems.Add("Malformed checksum line: $line")
                continue
            }
            $expected = $Matches[1].ToLowerInvariant()
            $relative = $Matches[2].Trim()
            $path = [IO.Path]::GetFullPath((Join-Path $ProjectRoot $relative))
            $hashCount++
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                $hashProblems.Add("Missing: $relative")
                continue
            }
            $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected) {
                $hashProblems.Add("Hash mismatch: $relative")
            }
        }
        Add-Check 'Frozen checksums' ($hashProblems.Count -eq 0 -and $hashCount -eq 12) $(
            if ($hashProblems.Count -eq 0) { "$hashCount files match" }
            else { $hashProblems -join '; ' }
        )
    }

    $protocolPath = Join-Path $ProjectRoot 'protocol.json'
    $protocol = Get-Content -LiteralPath $protocolPath -Raw | ConvertFrom-Json
    $methodCount = @($protocol.non_deep_baselines).Count + @($protocol.deep_baselines).Count + 1
    $protocolOk = ([int]$protocol.seed -eq 2027) -and ($methodCount -eq 15) -and
        (-not (@($protocol.non_deep_baselines) -contains 'KNN'))
    Add-Check 'Frozen protocol' $protocolOk "seed=$($protocol.seed); methods=$methodCount"

    $dataRoot = Join-Path $ProjectRoot 'data'
    if (-not (Test-Path -LiteralPath $dataRoot -PathType Container)) {
        $dataRoot = Join-Path $WorkspaceRoot 'phasead_pipeline\data'
    }
    $selectedFiles = New-Object System.Collections.Generic.List[object]
    $trackExpected = @{ U = 122; M = 71 }
    $missingData = New-Object System.Collections.Generic.List[string]
    $trackDetails = New-Object System.Collections.Generic.List[string]

    foreach ($track in @('U', 'M')) {
        $listPath = Join-Path $dataRoot "File_List\TSB-AD-$track-Eva.csv"
        if (-not (Test-Path -LiteralPath $listPath -PathType Leaf)) {
            $missingData.Add("Missing Eval list: $listPath")
            continue
        }
        $datasets = @($protocol.tracks.$track)
        $datasetPattern = ($datasets | ForEach-Object { [regex]::Escape([string]$_) }) -join '|'
        $rows = @(Import-Csv -LiteralPath $listPath | Where-Object {
            ([string]$_.file_name) -match "_($datasetPattern)_"
        })
        $trackDetails.Add("$track=$($rows.Count)")
        if ($rows.Count -ne $trackExpected[$track]) {
            $missingData.Add("Unexpected $track selection count: $($rows.Count)")
        }
        foreach ($row in $rows) {
            $fileName = [string]$row.file_name
            $dataPath = Join-Path $dataRoot "TSB-AD-$track\$fileName"
            if (-not (Test-Path -LiteralPath $dataPath -PathType Leaf)) {
                $missingData.Add("Missing data: TSB-AD-$track/$fileName")
            }
            $selectedFiles.Add([pscustomobject]@{ Track = $track; File = $fileName })
        }
    }
    $uniqueSelected = @($selectedFiles | Group-Object Track, File)
    $unitCount = $uniqueSelected.Count * $methodCount
    $dataOk = ($missingData.Count -eq 0) -and ($uniqueSelected.Count -eq 193) -and ($unitCount -eq 2895)
    Add-Check 'Official Eval data' $dataOk $(
        if ($missingData.Count -eq 0) {
            "$($trackDetails -join ', '); series=$($uniqueSelected.Count); units=$unitCount"
        }
        else {
            $missingData -join '; '
        }
    )

    $dependencyLockPath = Join-Path $ProjectRoot 'dependencies.lock.json'
    $dependencyLock = Get-Content -LiteralPath $dependencyLockPath -Raw | ConvertFrom-Json
    $dependencyProblems = New-Object System.Collections.Generic.List[string]
    foreach ($source in @($dependencyLock.sources)) {
        $sourceRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot ([string]$source.destination)))
        $markerPath = Join-Path $sourceRoot '.duoba-source.json'
        if (-not (Test-Path -LiteralPath $sourceRoot -PathType Container)) {
            $dependencyProblems.Add("Missing source: $($source.name)")
            continue
        }
        if (Test-Path -LiteralPath $markerPath -PathType Leaf) {
            $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
            if ([string]$marker.commit -ne [string]$source.commit) {
                $dependencyProblems.Add("Commit marker mismatch: $($source.name)")
            }
        }
        foreach ($relative in @($source.required_files)) {
            $path = Join-Path $sourceRoot ([string]$relative -replace '/', '\')
            $hashProperty = $source.required_sha256.PSObject.Properties[[string]$relative]
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                $dependencyProblems.Add("Missing $($source.name)/$relative")
                continue
            }
            if ($null -eq $hashProperty) {
                $dependencyProblems.Add("Unlocked $($source.name)/$relative")
                continue
            }
            $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne ([string]$hashProperty.Value).ToLowerInvariant()) {
                $dependencyProblems.Add("Hash mismatch $($source.name)/$relative")
            }
        }
    }
    Add-Check 'Pinned upstream sources' ($dependencyProblems.Count -eq 0) $(
        if ($dependencyProblems.Count -eq 0) { "$(@($dependencyLock.sources).Count) source trees match commit locks and required-file hashes" }
        else { $dependencyProblems -join '; ' }
    )

    $recoveryRoot = Join-Path $ProjectRoot 'multiserver\server_b_recovery_396_20260720T1441Z'
    $unitRoot = Join-Path $recoveryRoot 'units'
    $metricNames = @('VUS-PR', 'VUS-ROC', 'R-based-F1', 'AUC-PR', 'AUC-ROC', 'Standard-F1')
    $recoveryProblems = New-Object System.Collections.Generic.List[string]
    $validUnits = New-Object System.Collections.Generic.List[object]
    $unitFiles = @(Get-ChildItem -LiteralPath $unitRoot -Recurse -File -Filter '*.json')
    foreach ($unitFile in $unitFiles) {
        try {
            $unit = Get-Content -LiteralPath $unitFile.FullName -Raw | ConvertFrom-Json
            $ok = ([int]$unit.seed -eq 2027) -and ($null -eq $unit.error)
            foreach ($metricName in $metricNames) {
                $metricProperty = $unit.metrics.PSObject.Properties[$metricName]
                if ($null -eq $metricProperty -or -not (Test-FiniteNumber $metricProperty.Value)) {
                    $ok = $false
                }
            }
            if ($ok) {
                $validUnits.Add($unit)
            }
            else {
                $recoveryProblems.Add("Invalid unit: $($unitFile.FullName)")
            }
        }
        catch {
            $recoveryProblems.Add("Unreadable unit: $($unitFile.FullName)")
        }
    }

    $duplicates = @($validUnits | Group-Object method, track, dataset, file | Where-Object Count -ne 1)
    if ($duplicates.Count -gt 0) {
        $recoveryProblems.Add("Duplicate unit identities: $($duplicates.Count)")
    }
    $methodCounts = @{}
    foreach ($group in ($validUnits | Group-Object method)) {
        $methodCounts[$group.Name] = $group.Count
    }
    foreach ($method in @($protocol.deep_baselines)) {
        $expectedCount = $(if ($method -eq 'GBOC') { 9 } else { 43 })
        $actualCount = $(if ($methodCounts.ContainsKey($method)) { [int]$methodCounts[$method] } else { 0 })
        if ($actualCount -ne $expectedCount) {
            $recoveryProblems.Add("$method count is $actualCount, expected $expectedCount")
        }
    }
    $recoveryOk = ($unitFiles.Count -eq 396) -and ($validUnits.Count -eq 396) -and ($recoveryProblems.Count -eq 0)
    Add-Check 'Recovered Server B metrics' $recoveryOk $(
        if ($recoveryProblems.Count -eq 0) { '396 valid; 0 invalid; 0 duplicate identities' }
        else { $recoveryProblems -join '; ' }
    )

    $planPath = Join-Path $ProjectRoot 'multiserver\server_b_split_plan.json'
    $plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
    $expectedPlanHash = '543c884bfc761f5fa767392b5bc29c9b7c3db6f56a07d2abd641cd3f14c90178'
    $planOk = ([string]$plan.plan_sha256).ToLowerInvariant() -eq $expectedPlanHash
    Add-Check 'Historical split plan identity' $planOk "embedded plan_sha256=$($plan.plan_sha256)"
}
catch {
    Add-Check 'Verifier execution' $false $_.Exception.Message
}

$checks | Format-Table -AutoSize -Wrap
if ($failed) {
    Write-Error 'Local asset verification failed. Do not launch the active experiment.'
    exit 1
}

Write-Host 'PASS: local active assets, official Eval selection, and 396 recovered metrics are internally consistent.' -ForegroundColor Green
exit 0
