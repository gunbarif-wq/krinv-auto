$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$python = (Get-Command python).Source
$runName = "auto150_3m_1y_excl50_real_fetch"
$excludeSummaryPath = Join-Path $repo "data\chart_retrain\live50_3m_6m_excl30_real_fetch\summary.json"
$selectedSummaryPath = Join-Path $repo "data\chart_retrain\$runName\selected_summary.json"
$summaryPath = Join-Path $repo "data\chart_retrain\$runName\summary.json"
$stdoutPath = Join-Path $repo "logs\fetch_150_1y_real.stdout.txt"
$stderrPath = Join-Path $repo "logs\fetch_150_1y_real.stderr.txt"
$runnerLogPath = Join-Path $repo "logs\fetch_150_1y_real.runner.txt"

Set-Location $repo
if (!(Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}
if (!(Test-Path (Join-Path $repo "data\\chart_retrain\\$runName"))) {
    New-Item -ItemType Directory -Path (Join-Path $repo "data\\chart_retrain\\$runName") | Out-Null
}

function Read-JsonSafely([string]$path) {
    if (!(Test-Path $path)) { return $null }
    try { return Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null }
}

function Get-Total {
    $json = Read-JsonSafely $selectedSummaryPath
    if ($null -eq $json) { return 150 }
    return @($json.selected_symbols).Count
}

function Get-Completed {
    $json = Read-JsonSafely $summaryPath
    if ($null -eq $json) { return 0 }
    try { return @($json.per_symbol.PSObject.Properties).Count } catch { return 0 }
}

function Ensure-SelectedSummary {
    if (Test-Path $selectedSummaryPath) { return }
    # Reads .env for app key/secret via load_dotenv in python module.
    & $python -u user_only_strategy\build_selected_summary_150.py `
        --base-url https://openapi.koreainvestment.com:9443 `
        --app-key $env:KIS_APP_KEY `
        --app-secret $env:KIS_APP_SECRET `
        --exclude-summary $excludeSummaryPath `
        --out-summary $selectedSummaryPath `
        --count 150 `
        1>> $stdoutPath `
        2>> $stderrPath
}

while ($true) {
    Ensure-SelectedSummary
    $total = Get-Total
    $done = Get-Completed
    if ($done -ge $total -and $total -gt 0) {
        Add-Content $runnerLogPath ("[runner] completed {0}/{1} at {2}" -f $done, $total, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
        break
    }

    Add-Content $runnerLogPath ("[runner] launch {0}/{1} at {2}" -f $done, $total, (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    & $python -u user_only_strategy\retrain_chart_classifier_live.py `
        --base-url https://openapi.koreainvestment.com:9443 `
        --quote-base-url https://openapi.koreainvestment.com:9443 `
        --use-selected-summary $selectedSummaryPath `
        --bar-minutes 3 `
        --fetch-bdays 252 `
        --exclude-recent-bdays 0 `
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

