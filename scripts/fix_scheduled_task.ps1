# Fix Qhawarina Daily Web Update scheduled task
# Reconfigures to run whether user is logged in or not

Write-Host "Removing existing task..."
Unregister-ScheduledTask -TaskName 'Qhawarina-DailyWeb' -Confirm:$false -ErrorAction SilentlyContinue

Write-Host "Creating new task with proper settings..."

$action = New-ScheduledTaskAction -Execute 'D:\Nexus\nexus\scripts\daily_web_update.bat' -WorkingDirectory 'D:\Nexus\nexus'
$trigger = New-ScheduledTaskTrigger -Daily -At 8:00AM
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERNAME" -RunLevel Highest -LogonType S4U
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RunOnlyIfNetworkAvailable

Register-ScheduledTask -TaskName 'Qhawarina-DailyWeb' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description 'Daily Qhawarina data update (supermarket prices + political index + FX)'

Write-Host "Task configured successfully!"
Write-Host "Settings:"
Write-Host "  - Runs daily at 8:00 AM"
Write-Host "  - Runs whether user is logged in or not"
Write-Host "  - Runs with highest privileges"
Write-Host "  - Starts when available (if missed)"
Write-Host "  - Works on battery power"
Write-Host "  - Requires network connection"
