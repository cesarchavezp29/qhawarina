@echo off
REM ============================================================================
REM NEXUS — Schedule automatic data updates with multiple frequencies
REM
REM Creates 4 Windows Scheduled Tasks:
REM   1. NEXUS_Daily    — Daily 8:00 AM: BCRP + panel rebuild
REM   2. NEXUS_Weekly   — Monday 6:00 AM: Full update + charts
REM   3. NEXUS_Monthly  — 20th 6:00 AM: Full rebuild + nowcast + viz
REM   4. NEXUS_Annual   — July 1st: ENAHO download + poverty update
REM
REM Usage (run as Administrator):
REM     schedule_nexus.bat install    — Create all scheduled tasks
REM     schedule_nexus.bat remove     — Remove all scheduled tasks
REM     schedule_nexus.bat status     — Check task statuses
REM ============================================================================

set PYTHON_EXE=python
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set UPDATE_SCRIPT=%SCRIPT_DIR%update_nexus.py

if "%1"=="install" goto install
if "%1"=="remove" goto remove
if "%1"=="status" goto status
goto usage

:install
echo Installing NEXUS scheduled tasks...
echo.

echo [1/4] Daily task (8:00 AM): BCRP data + panel rebuild
schtasks /create /tn "NEXUS_Daily" /tr "\"%PYTHON_EXE%\" \"%UPDATE_SCRIPT%\" --only bcrp" /sc daily /st 08:00 /f
if %errorlevel% neq 0 echo   WARNING: Failed to create daily task

echo [2/4] Weekly task (Mon 6:00 AM): Full update + visualizations
schtasks /create /tn "NEXUS_Weekly" /tr "\"%PYTHON_EXE%\" \"%UPDATE_SCRIPT%\"" /sc weekly /d MON /st 06:00 /f
if %errorlevel% neq 0 echo   WARNING: Failed to create weekly task

echo [3/4] Monthly task (20th 6:00 AM): Full rebuild
schtasks /create /tn "NEXUS_Monthly" /tr "\"%PYTHON_EXE%\" \"%UPDATE_SCRIPT%\" --force" /sc monthly /d 20 /st 06:00 /f
if %errorlevel% neq 0 echo   WARNING: Failed to create monthly task

echo [4/4] Annual task (July 1st): ENAHO download + poverty
schtasks /create /tn "NEXUS_Annual" /tr "\"%PYTHON_EXE%\" \"%UPDATE_SCRIPT%\" --only enaho --force" /sc monthly /mo 12 /d 1 /st 06:00 /f
if %errorlevel% neq 0 echo   WARNING: Failed to create annual task

echo.
echo Tasks installed. Check logs at: %PROJECT_DIR%\logs\
goto end

:remove
echo Removing NEXUS scheduled tasks...
schtasks /delete /tn "NEXUS_Daily" /f 2>nul
schtasks /delete /tn "NEXUS_Weekly" /f 2>nul
schtasks /delete /tn "NEXUS_Monthly" /f 2>nul
schtasks /delete /tn "NEXUS_Annual" /f 2>nul
echo Done.
goto end

:status
echo NEXUS Scheduled Task Status:
echo.
for %%t in (NEXUS_Daily NEXUS_Weekly NEXUS_Monthly NEXUS_Annual) do (
    echo --- %%t ---
    schtasks /query /tn "%%t" /v /fo list 2>nul
    if %errorlevel% neq 0 echo   NOT INSTALLED
    echo.
)
goto end

:usage
echo Usage: schedule_nexus.bat [install^|remove^|status]
echo.
echo Commands:
echo   install  — Create all 4 scheduled tasks
echo   remove   — Remove all scheduled tasks
echo   status   — Check status of all tasks

:end
