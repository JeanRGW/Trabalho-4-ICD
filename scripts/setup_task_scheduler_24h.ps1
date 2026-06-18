param(
    [string]$TaskName = "DengueMonitor24h",
    [string]$StartTime = "03:00"
)

$runner = Join-Path $PSScriptRoot "run_monitor_daily.ps1"

$taskCommand = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$runner`""

schtasks /Create /TN $TaskName /SC DAILY /ST $StartTime /TR $taskCommand /F | Out-Host

Write-Host "Tarefa criada: $TaskName"
Write-Host "Horario diario: $StartTime"
Write-Host "Teste manual: schtasks /Run /TN $TaskName"
