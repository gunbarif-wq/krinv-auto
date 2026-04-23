$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$runName = "live50_3m_12m_excl30_real_fetch"
$sourceSummaryPath = Join-Path $repo "data\chart_retrain\live50_3m_6m_excl30_real_fetch\summary.json"
$summaryPath = Join-Path $repo "data\chart_retrain\$runName\summary.json"
$stdoutPath = Join-Path $repo "logs\fetch_12m_real.stdout.txt"
$runnerLogPath = Join-Path $repo "logs\fetch_12m_real.runner.txt"
$dataDir = Join-Path $repo "data\chart_retrain\$runName"

function Read-JsonSafely([string]$path) {
    if (!(Test-Path $path)) { return $null }
    try { return Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null }
}

function Get-Total {
    $json = Read-JsonSafely $sourceSummaryPath
    if ($null -eq $json) { return 50 }
    return @($json.selected_symbols).Count
}

function Get-Completed {
    $json = Read-JsonSafely $summaryPath
    if ($null -eq $json) { return 0 }
    try { return @($json.per_symbol.PSObject.Properties).Count } catch { return 0 }
}

function Get-LastLine([string]$path, [int]$n = 1) {
    if (!(Test-Path $path)) { return "" }
    try { return (Get-Content $path -Tail $n | Select-Object -Last 1) } catch { return "" }
}

while ($true) {
    $total = Get-Total
    $done = Get-Completed
    $pct = 0.0
    if ($total -gt 0) { $pct = [Math]::Round(100.0 * $done / $total, 1) }

    $dirSize = 0
    if (Test-Path $dataDir) {
        $dirSize = (Get-ChildItem -Path $dataDir -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    }
    $dirSizeMb = [Math]::Round(($dirSize / 1MB), 1)

    $runner = Get-LastLine $runnerLogPath 1
    $stdout = Get-LastLine $stdoutPath 1

    Clear-Host
    Write-Host ("Fetch Progress: {0}/{1} ({2}%)" -f $done, $total, $pct)
    Write-Host ("Data Dir Size: {0} MB  ({1})" -f $dirSizeMb, $dataDir)
    if ($runner) { Write-Host ("Runner: {0}" -f $runner) }
    if ($stdout) { Write-Host ("Stdout: {0}" -f $stdout) }
    Write-Host ("Time: {0}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Start-Sleep -Seconds 5
}

