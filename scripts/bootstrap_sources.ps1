[CmdletBinding()]
param(
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Split-Path -Parent $PSScriptRoot
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
$lockPath = Join-Path $ProjectRoot 'dependencies.lock.json'
$lock = Get-Content -LiteralPath $lockPath -Raw | ConvertFrom-Json
$tempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())

function Assert-RequiredFiles {
    param($Source, [string]$Destination)
    foreach ($relative in @($Source.required_files)) {
        $path = Join-Path $Destination ([string]$relative -replace '/', '\')
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "$($Source.name) is incomplete; missing $relative"
        }
        $hashProperty = $Source.required_sha256.PSObject.Properties[[string]$relative]
        if ($null -eq $hashProperty) {
            throw "$($Source.name) lock is missing the SHA256 for $relative"
        }
        $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne ([string]$hashProperty.Value).ToLowerInvariant()) {
            throw "$($Source.name) hash mismatch for $relative"
        }
    }
}

foreach ($source in @($lock.sources)) {
    $destination = [IO.Path]::GetFullPath((Join-Path $ProjectRoot ([string]$source.destination)))
    $markerPath = Join-Path $destination '.stage-source.json'
    if (Test-Path -LiteralPath $destination -PathType Container) {
        Assert-RequiredFiles $source $destination
        if (Test-Path -LiteralPath $markerPath -PathType Leaf) {
            $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
            if ([string]$marker.commit -ne [string]$source.commit) {
                throw "$($source.name) exists at the wrong commit marker: $($marker.commit)"
            }
        }
        Write-Host "READY $($source.name): $destination"
        continue
    }

    $temporary = Join-Path $tempBase ("stage-source-{0}-{1}" -f $PID, [guid]::NewGuid().ToString('N'))
    $temporary = [IO.Path]::GetFullPath($temporary)
    if (-not $temporary.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Unsafe temporary path: $temporary"
    }
    New-Item -ItemType Directory -Path $temporary | Out-Null
    try {
        $archive = Join-Path $temporary 'source.zip'
        $expanded = Join-Path $temporary 'expanded'
        Write-Host "FETCH $($source.name) commit $($source.commit)"
        Invoke-WebRequest -Uri ([string]$source.archive_url) -OutFile $archive -UseBasicParsing
        Expand-Archive -LiteralPath $archive -DestinationPath $expanded
        $roots = @(Get-ChildItem -LiteralPath $expanded -Directory)
        if ($roots.Count -ne 1) {
            throw "$($source.name) archive has $($roots.Count) top-level directories"
        }
        Assert-RequiredFiles $source $roots[0].FullName
        $parent = Split-Path -Parent $destination
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
        Move-Item -LiteralPath $roots[0].FullName -Destination $destination

        $requiredHashes = [ordered]@{}
        foreach ($relative in @($source.required_files)) {
            $path = Join-Path $destination ([string]$relative -replace '/', '\')
            $requiredHashes[[string]$relative] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
        [ordered]@{
            schema_version = 'stage-source-marker-v1'
            name = [string]$source.name
            repository = [string]$source.repository
            commit = [string]$source.commit
            fetched_utc = (Get-Date).ToUniversalTime().ToString('o')
            required_file_sha256 = $requiredHashes
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $markerPath -Encoding UTF8
        Write-Host "READY $($source.name): $destination"
    }
    finally {
        if (Test-Path -LiteralPath $temporary) {
            $resolved = [IO.Path]::GetFullPath($temporary)
            if (-not $resolved.StartsWith($tempBase, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to remove unsafe temporary path: $resolved"
            }
            Remove-Item -LiteralPath $resolved -Recurse -Force
        }
    }
}
