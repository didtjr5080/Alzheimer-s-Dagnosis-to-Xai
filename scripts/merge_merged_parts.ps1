# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/merge_merged_parts.ps1 [-Root E:\Develop\CLIPtoXAI] [-Execute]
# Exit codes: 0 = success/dry-run success, 2 = content conflict, 3 = unsafe existing target or missing input, 4 = copy failure
param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$Execute
)

$ErrorActionPreference = "Stop"
$partNames = @(
    "merged_project-20260825T072246Z-1-001",
    "merged_project-20260825T072246Z-1-002",
    "merged_project-20260825T072246Z-1-003"
)
$target = Join-Path $Root "merged_project"
$artifactDir = Join-Path $Root "artifacts\merge_validation"
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $artifactDir "merge_$timestamp.txt"

function Write-Log([string]$Text) {
    $Text | Tee-Object -FilePath $logPath -Append
}

function Get-RelativePath([string]$Base, [string]$Path) {
    $baseUri = [Uri]((Resolve-Path $Base).Path.TrimEnd('\') + '\')
    $pathUri = [Uri](Resolve-Path $Path).Path
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('/', '\')
}

Write-Log "Root: $Root"
Write-Log "Target: $target"
Write-Log "Mode: $(if ($Execute) { 'EXECUTE' } else { 'DRY-RUN' })"

$partRoots = @()
foreach ($partName in $partNames) {
    $partRoot = Join-Path $Root "$partName\merged_project"
    if (-not (Test-Path -LiteralPath $partRoot -PathType Container)) {
        Write-Log "Missing part root: $partRoot"
        exit 3
    }
    $partRoots += [pscustomobject]@{ Part = $partName; Root = (Resolve-Path $partRoot).Path }
}

if ((Test-Path -LiteralPath $target -PathType Container) -and (Get-ChildItem -LiteralPath $target -Force | Select-Object -First 1)) {
    Write-Log "Target exists and is not empty. Refusing to copy: $target"
    exit 3
}

$seen = @{}
$wouldCopy = 0
$wouldSkip = 0
$copied = 0
$skipped = 0
foreach ($part in $partRoots) {
    Write-Log "Processing $($part.Part)"
    $files = Get-ChildItem -LiteralPath $part.Root -File -Recurse -Force
    foreach ($file in $files) {
        $rel = Get-RelativePath $part.Root $file.FullName
        $dest = Join-Path $target $rel
        if ($seen.ContainsKey($rel)) {
            $existing = $seen[$rel]
            if (-not $existing.Hash) {
                $existing.Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $existing.Path).Hash
            }
            $newHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $file.FullName).Hash
            if ($existing.Hash -ne $newHash) {
                Write-Log "Content conflict at $rel"
                Write-Log "  Existing: $($existing.Path) $($existing.Hash)"
                Write-Log "  New:      $($file.FullName) $newHash"
                exit 2
            }
            $wouldSkip++
            if ($Execute) { $skipped++ }
            continue
        }
        $seen[$rel] = [pscustomobject]@{ Path = $file.FullName; Hash = $null }
        $wouldCopy++
        if ($Execute) {
            $destDir = Split-Path -Parent $dest
            New-Item -ItemType Directory -Force -Path $destDir | Out-Null
            Copy-Item -LiteralPath $file.FullName -Destination $dest -Force
            (Get-Item -LiteralPath $dest).LastWriteTimeUtc = $file.LastWriteTimeUtc
            $copied++
        }
    }
    Write-Log "Finished $($part.Part): files=$($files.Count)"
}

if (-not $Execute) {
    Write-Log "Dry-run complete. Unique files to copy: $wouldCopy, duplicate identical files to skip: $wouldSkip"
    Write-Log "Run again with -Execute to create the merged project."
    exit 0
}

Write-Log "Merge complete. Copied: $copied, duplicate identical skipped: $skipped"
exit 0
