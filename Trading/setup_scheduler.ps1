# WaveRider Signal Bot — Scheduled Task Setup
# Run this script as Administrator in PowerShell:
#   powershell -ExecutionPolicy Bypass -File setup_scheduler.ps1

# Remove old task if it exists
schtasks /delete /tn "WaveRider-DailySignal" /f 2>$null

# Create task action
$action = New-ScheduledTaskAction `
    -Execute "C:\Users\dilip\AppData\Local\Programs\Python\Python310\python.exe" `
    -Argument "D:\Experiments\Trading\waverider_signal_bot.py" `
    -WorkingDirectory "D:\Experiments\Trading"

# Trigger: Mon-Fri at 4:30 PM
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "4:30PM"

# Settings: run even on battery, start if missed, require network
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Principal: runs as current user with stored credentials (S4U = no password needed)
$principal = New-ScheduledTaskPrincipal -UserId "dilip" -LogonType S4U -RunLevel Highest

Register-ScheduledTask `
    -TaskName "WaveRider-DailySignal" `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "WaveRider T5 daily signal bot - sends Telegram alerts after market close"

Write-Host "`nScheduled task created. Verifying..." -ForegroundColor Green
Get-ScheduledTask -TaskName "WaveRider-DailySignal" | Format-List TaskName, State, Description
Get-ScheduledTaskInfo -TaskName "WaveRider-DailySignal" | Format-List LastRunTime, NextRunTime
