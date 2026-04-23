$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$python = (Get-Command python).Source
$runName = "live50_3m_12m_excl30_real_fetch"
$sourceSummaryPath = Join-Path $repo "data\chart_retrain\live50_3m_6m_excl30_real_fetch\summary.json"
$summaryPath = Join-Path $repo "data\chart_retrain\$runName\summary.json"
$stdoutPath = Join-Path $repo "logs\fetch_12m_real.stdout.txt"
$stderrPath = Join-Path $repo "logs\fetch_12m_real.stderr.txt"
$runnerLogPath = Join-Path $repo "logs\fetch_12m_real.runner.txt"

Set-Location $repo
if (!(Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

function Read-JsonSafely([string]$path) {
    if (!(Test-Path $path)) {
        return $null
    }
    try {
        return Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-SourceTotal {
    $json = Read-JsonSafely $sourceSummaryPath
    if ($null -eq $json) {
        return 50
    }
    return @($json.selected_symbols).Count
}

function Get-ProgressState {
    $total = Get-SourceTotal
    $json = Read-JsonSafely $summaryPath
    if ($null -eq $json) {
        return [pscustomobject]@{
            Total = $total
            Completed = 0
        }
    }
    $completed = @($json.per_symbol.PSObject.Properties).Count
    return [pscustomobject]@{
        Total = $total
        Completed = $completed
    }
}

while ($true) {
    $state = Get-ProgressState
    if ($state.Completed -ge $state.Total -and $state.Total -gt 0) {
        Add-Content $runnerLogPath ("[runner] completed {0}/{1} at {2}" -f $state.Completed, $state.Total, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        break
    }

    Add-Content $runnerLogPath ("[runner] launch {0}/{1} at {2}" -f $state.Completed, $state.Total, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    & $python -u user_only_strategy\retrain_chart_classifier_live.py `
        --base-url https://openapi.koreainvestment.com:9443 `
        --quote-base-url https://openapi.koreainvestment.com:9443 `
        --use-selected-summary data\chart_retrain\live50_3m_6m_excl30_real_fetch\summary.json `
        --bar-minutes 3 `
        --fetch-bdays 252 `
        --exclude-recent-bdays 30 `
        --fetch-only `
        --retry-count 2 `
        --max-day-errors-per-symbol 12 `
        --max-consecutive-day-errors 5 `
        --resume `
        --model-name $runName `
        1>> $stdoutPath `
        2>> $stderrPath

    Start-Sleep -Seconds 10
}
