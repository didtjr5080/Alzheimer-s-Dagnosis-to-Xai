# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/inspect_merged_parts.ps1 [-Root E:\Develop\CLIPtoXAI]
# Exit codes: 0 = no blocking collisions, 2 = blocking content conflicts, 3 = missing inputs
param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
$partNames = @(
    "merged_project-20260825T072246Z-1-001",
    "merged_project-20260825T072246Z-1-002",
    "merged_project-20260825T072246Z-1-003"
)
$artifactDir = Join-Path $Root "artifacts\merge_validation"
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $artifactDir "inspect_$timestamp.txt"
$relCollisionPath = Join-Path $artifactDir "relative_path_collisions_$timestamp.csv"
$basenameCollisionPath = Join-Path $artifactDir "basename_collisions_$timestamp.csv"
$summaryPath = Join-Path $artifactDir "part_summary_$timestamp.csv"

function Write-Log([string]$Text) {
    $Text | Tee-Object -FilePath $logPath -Append
}

function Get-RelativePath([string]$Base, [string]$Path) {
    $baseUri = [Uri]((Resolve-Path $Base).Path.TrimEnd('\') + '\')
    $pathUri = [Uri](Resolve-Path $Path).Path
    return [Uri]::UnescapeDataString($baseUri.MakeRelativeUri($pathUri).ToString()).Replace('/', '\')
}

Write-Log "Root: $Root"
Write-Log "Artifact directory: $artifactDir"

$missing = @()
$partRoots = @()
foreach ($partName in $partNames) {
    $partRoot = Join-Path $Root "$partName\merged_project"
    if (-not (Test-Path -LiteralPath $partRoot -PathType Container)) {
        $missing += $partRoot
    } else {
        $partRoots += [pscustomobject]@{ Part = $partName; Root = (Resolve-Path $partRoot).Path }
    }
}
if ($missing.Count -gt 0) {
    Write-Log "Missing part roots:"
    $missing | ForEach-Object { Write-Log "  $_" }
    exit 3
}

$allRows = New-Object System.Collections.Generic.List[object]
$summaryRows = New-Object System.Collections.Generic.List[object]
foreach ($part in $partRoots) {
    Write-Log "Scanning $($part.Part)"
    $files = Get-ChildItem -LiteralPath $part.Root -File -Recurse -Force
    $totalBytes = 0L
    $extCounts = @{}
    foreach ($file in $files) {
        $totalBytes += $file.Length
        $ext = if ($file.Extension) { $file.Extension.ToLowerInvariant() } else { "[no_ext]" }
        $extCounts[$ext] = 1 + [int]($extCounts[$ext])
        $allRows.Add([pscustomobject]@{
            Part = $part.Part
            FullPath = $file.FullName
            RelativePath = Get-RelativePath $part.Root $file.FullName
            FileName = $file.Name
            Length = $file.Length
            LastWriteTimeUtc = $file.LastWriteTimeUtc.ToString("o")
        })
    }
    $summaryRows.Add([pscustomobject]@{
        Part = $part.Part
        FileCount = $files.Count
        TotalBytes = $totalBytes
        ExtensionCounts = (($extCounts.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join "; ")
    })
}

$summaryRows | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $summaryPath
Write-Log "Part summary written: $summaryPath"

$blocking = New-Object System.Collections.Generic.List[object]
$relGroups = $allRows | Group-Object RelativePath | Where-Object { $_.Count -gt 1 }
$relReport = New-Object System.Collections.Generic.List[object]
foreach ($group in $relGroups) {
    $hashes = @{}
    foreach ($row in $group.Group) {
        $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $row.FullPath).Hash
        $hashes[$hash] = $true
        $relReport.Add([pscustomobject]@{
            RelativePath = $row.RelativePath
            Part = $row.Part
            FullPath = $row.FullPath
            Length = $row.Length
            LastWriteTimeUtc = $row.LastWriteTimeUtc
            Sha256 = $hash
            IdenticalContent = $null
        })
    }
    $same = ($hashes.Keys.Count -eq 1)
    foreach ($entry in $relReport | Where-Object { $_.RelativePath -eq $group.Name }) {
        $entry.IdenticalContent = $same
    }
    if (-not $same) {
        $blocking.Add([pscustomobject]@{ RelativePath = $group.Name; Variants = $hashes.Keys.Count })
    }
}
$relReport | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $relCollisionPath
Write-Log "Relative path collisions: $($relGroups.Count), blocking content conflicts: $($blocking.Count)"
Write-Log "Relative collision report: $relCollisionPath"

$basenameGroups = $allRows | Group-Object FileName | Where-Object { $_.Count -gt 1 }
$basenameReport = foreach ($group in $basenameGroups) {
    foreach ($row in $group.Group) {
        [pscustomobject]@{
            FileName = $row.FileName
            Part = $row.Part
            RelativePath = $row.RelativePath
            FullPath = $row.FullPath
            Length = $row.Length
        }
    }
}
$basenameReport | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $basenameCollisionPath
Write-Log "Same filename groups across paths: $($basenameGroups.Count)"
Write-Log "Basename collision report: $basenameCollisionPath"

if ($blocking.Count -gt 0) {
    Write-Log "Blocking conflicts found. Do not merge until reviewed."
    $blocking | Format-Table -AutoSize | Out-String | Tee-Object -FilePath $logPath -Append
    exit 2
}
Write-Log "No blocking content conflicts found."
exit 0
