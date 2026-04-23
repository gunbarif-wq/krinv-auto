$ErrorActionPreference = "Stop"

$repo = $PSScriptRoot
$runner = Join-Path $repo "run_fetch_12m_until_done.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`""

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName "KRINV-Fetch-12M" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Resume KRINV 12-month raw data fetch at logon" `
    -Force | Out-Null

Write-Output "Scheduled task registered: KRINV-Fetch-12M"
