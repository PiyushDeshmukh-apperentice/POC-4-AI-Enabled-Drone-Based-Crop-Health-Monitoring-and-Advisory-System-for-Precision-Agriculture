#Requires -Version 7.0
# =============================================================================
# 00_download.ps1  —  Phase 0: Parallel Resumable Downloader
# =============================================================================
# Safely resumable: uses aws s3api head-object for S3 size verification.
# Parallel:         ForEach-Object -Parallel with configurable throttle limit.
# Logging:          Timestamped log file under .\logs\.
# Stray cleanup:    Deletes files not matching the 5-variant naming pattern.
#
# Usage:
#   .\pipeline\00_download.ps1
#   .\pipeline\00_download.ps1 -UriFile .\first_1000_uris.txt -DestRoot .\test_1000_images -Throttle 8
#
# Safely rerunnable — already-complete files are skipped.
# =============================================================================

param(
    [string]$UriFile    = ".\first_1000_uris.txt",
    [string]$DestRoot   = ".\test_1000_images",
    [string]$S3Prefix   = "s3://ieee-dataport/data/1426491/100367/",
    [int]   $Throttle   = 8
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
$LogDir = ".\logs"
if (-Not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$Timestamp  = (Get-Date -Format "yyyyMMdd_HHmmss")
$LogFile    = Join-Path $LogDir "download_${Timestamp}.log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    Add-Content -Path $LogFile -Value $line
    $color = switch ($Level) {
        "OK"      { "Green"    }
        "SKIP"    { "DarkGray" }
        "WARN"    { "Yellow"   }
        "ERROR"   { "Red"      }
        "STRAY"   { "Magenta"  }
        default   { "Cyan"     }
    }
    Write-Host $line -ForegroundColor $color
}

# ---------------------------------------------------------------------------
# Validate inputs
# ---------------------------------------------------------------------------
if (-Not (Test-Path $UriFile)) {
    Write-Log "URI file not found: $UriFile" "ERROR"
    exit 1
}

$resolved = Resolve-Path -LiteralPath $DestRoot -ErrorAction SilentlyContinue
if ($resolved) {
    $DestRoot = $resolved.Path
} else {
    Write-Log "DestRoot not found yet - it will be created: $DestRoot" "WARN"
    New-Item -ItemType Directory -Path $DestRoot -Force | Out-Null
    $DestRoot = (Resolve-Path -LiteralPath $DestRoot).Path
}

Write-Log "=== IMPaCT-UAV-MsRGB Downloader starting ===" "INFO"
Write-Log "URI file  : $UriFile"   "INFO"
Write-Log "Dest root : $DestRoot"  "INFO"
Write-Log "Throttle  : $Throttle"  "INFO"
Write-Log "Log file  : $LogFile"   "INFO"

# ---------------------------------------------------------------------------
# Step 1 — Stray file cleanup
# ---------------------------------------------------------------------------
Write-Log "--- Step 1: Scanning for stray files ---" "INFO"

$ValidPattern = '^\d{14}_\d{4}_(D\.JPG|MS_G\.TIF|MS_R\.TIF|MS_RE\.TIF|MS_NIR\.TIF)$'
$strayCount   = 0

if (Test-Path $DestRoot) {
    Get-ChildItem -Recurse -File -LiteralPath $DestRoot | Where-Object {
        $_.Name -notmatch $ValidPattern
    } | ForEach-Object {
        Write-Log "STRAY  $($_.FullName)  [$($_.Length) bytes]" "STRAY"
        Remove-Item -LiteralPath $_.FullName -Force
        $strayCount++
    }
}

Write-Log "Stray files removed: $strayCount" "INFO"

# ---------------------------------------------------------------------------
# Step 2 — Load URI list and build work queue
# ---------------------------------------------------------------------------
Write-Log "--- Step 2: Building work queue ---" "INFO"

$uris = Get-Content $UriFile | Where-Object { -Not [string]::IsNullOrWhiteSpace($_) }
$total = $uris.Count
Write-Log "Total URIs in manifest: $total" "INFO"

# Build a list of work items (hashtables) for parallel processing
$workItems = [System.Collections.Generic.List[hashtable]]::new()

foreach ($uri in $uris) {
    $relativePath = $uri -replace [regex]::Escape($S3Prefix), ""
    $localPath    = Join-Path $DestRoot $relativePath
    $workItems.Add(@{
        Uri          = $uri
        RelativePath = $relativePath
        LocalPath    = $localPath
        S3Bucket     = ($uri -replace '^s3://([^/]+)/.*', '$1')
        S3Key        = ($uri -replace '^s3://[^/]+/', '')
    })
}

# ---------------------------------------------------------------------------
# Step 3 — Parallel download with S3 size verification
# ---------------------------------------------------------------------------
Write-Log "--- Step 3: Parallel download (throttle=$Throttle) ---" "INFO"

# Thread-safe counters via ConcurrentBag + Interlocked
$results = [System.Collections.Concurrent.ConcurrentBag[string]]::new()

$workItems | ForEach-Object -Parallel {
    $item      = $_
    $logFile   = $using:LogFile
    $destRoot  = $using:DestRoot

    function ParallelLog([string]$msg, [string]$lvl) {
        $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$lvl] $msg"
        # File-safe append (PS7 Add-Content is thread-safe for append)
        Add-Content -Path $using:LogFile -Value $line
    }

    $localPath = $item.LocalPath
    $uri       = $item.Uri
    $relPath   = $item.RelativePath
    $bucket    = $item.S3Bucket
    $key       = $item.S3Key

    # Ensure local directory exists
    $localDir = Split-Path $localPath -Parent
    if (-Not (Test-Path $localDir)) {
        New-Item -ItemType Directory -Path $localDir -Force | Out-Null
    }

    # ------------------------------------------------------------------
    # Get S3 object size via head-object
    # ------------------------------------------------------------------
    $s3Size = $null
    try {
        $headJson = aws s3api head-object --bucket $bucket --key $key 2>&1
        if ($LASTEXITCODE -eq 0) {
            $headObj = $headJson | ConvertFrom-Json -ErrorAction SilentlyContinue
            $s3Size  = [long]$headObj.ContentLength
        }
    }
    catch { }

    # ------------------------------------------------------------------
    # Decide action
    # ------------------------------------------------------------------
    $needsDownload = $true

    if (Test-Path $localPath) {
        $localSize = (Get-Item -LiteralPath $localPath).Length

        if ($s3Size -ne $null -and $localSize -eq $s3Size -and $localSize -gt 0) {
            # File is complete
            ($using:results).Add("SKIP")
            ParallelLog "SKIP  $relPath  [local=$localSize == s3=$s3Size]" "SKIP"
            $needsDownload = $false
        }
        elseif ($localSize -eq 0) {
            ParallelLog "SIZE_MISMATCH  $relPath  [local=0, deleting]" "WARN"
            Remove-Item -LiteralPath $localPath -Force
        }
        else {
            $s3Disp = if ($s3Size -ne $null) { $s3Size } else { "unknown" }
            ParallelLog "SIZE_MISMATCH  $relPath  [local=$localSize, s3=$s3Disp, re-downloading]" "WARN"
            Remove-Item -LiteralPath $localPath -Force
        }
    }

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------
    if ($needsDownload) {
        ParallelLog "GET   $relPath" "INFO"
        aws s3 cp $uri $localPath --only-show-errors 2>&1 | Out-Null

        if ($LASTEXITCODE -eq 0) {
            ($using:results).Add("OK")
            ParallelLog "OK    $relPath" "OK"
        }
        else {
            ($using:results).Add("FAIL")
            ParallelLog "FAIL  $relPath" "ERROR"
            if (Test-Path $localPath) { Remove-Item -LiteralPath $localPath -Force }
        }
    }

} -ThrottleLimit $Throttle

# ---------------------------------------------------------------------------
# Step 4 — Summary
# ---------------------------------------------------------------------------
$resultList = $results.ToArray()
$okCount    = ($resultList | Where-Object { $_ -eq "OK"   }).Count
$skipCount  = ($resultList | Where-Object { $_ -eq "SKIP" }).Count
$failCount  = ($resultList | Where-Object { $_ -eq "FAIL" }).Count

Write-Log "=== Download Summary ===" "INFO"
Write-Log "Total in manifest : $total"          "INFO"
Write-Log "Newly downloaded  : $okCount"        "OK"
Write-Log "Skipped (complete): $skipCount"      "SKIP"
Write-Log "Failed            : $failCount"      $(if ($failCount -gt 0) {"ERROR"} else {"INFO"})
Write-Log "Strays removed    : $strayCount"     "INFO"

if ($failCount -gt 0) {
    Write-Log "Some files failed. Run this script again to retry them." "WARN"
    exit 1
}
else {
    Write-Log "All files verified complete." "OK"
    exit 0
}
