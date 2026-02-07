@echo off
REM ============================================================================
REM NEXUS — Schedule automatic data updates (all sources)
REM
REM This script creates a Windows Scheduled Task that runs update_all.py
REM daily at 8:00 AM. Covers BCRP monthly, quarterly GDP, inflation, and ENAHO.
REM
REM Usage (run as Administrator):
REM     schedule_update.bat install    — Create the scheduled task
REM     schedule_update.bat remove     — Remove the scheduled task
REM     schedule_update.bat run        — Run the update manually now
REM     schedule_update.bat status     — Check task status
REM ============================================================================

set TASK_NAME=NEXUS_Data_Update
set PYTHON_EXE=python
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set UPDATE_SCRIPT=%SCRIPT_DIR%update_all.py
set LOG_FILE=%PROJECT_DIR%\logs\update_all.log

if "%1"=="install" goto install
if "%1"=="remove" goto remove
if "%1"=="run" goto run
if "%1"=="status" goto status
goto usage

:install
echo Creating scheduled task: %TASK_NAME%
echo Script: %UPDATE_SCRIPT%
echo Schedule: Daily at 08:00 AM
echo.
schtasks /create /tn "%TASK_NAME%" /tr "\"%PYTHON_EXE%\" \"%UPDATE_SCRIPT%\"" /sc daily /st 08:00 /f
if %errorlevel% equ 0 (
    echo.
    echo Task created successfully!
    echo Data will be updated daily at 8:00 AM.
    echo Check logs at: %LOG_FILE%
) else (
    echo.
    echo Failed to create task. Try running as Administrator.
)
goto end

:remove
echo Removing scheduled task: %TASK_NAME%
schtasks /delete /tn "%TASK_NAME%" /f
goto end

:run
echo Running NEXUS update now...
"%PYTHON_EXE%" "%UPDATE_SCRIPT%"
goto end

:status
echo Checking task status...
schtasks /query /tn "%TASK_NAME%" /v /fo list 2>nul
if %errorlevel% neq 0 (
    echo Task "%TASK_NAME%" is not installed.
    echo Run: schedule_update.bat install
)
goto end

:usage
echo Usage: schedule_update.bat [install^|remove^|run^|status]
echo.
echo Commands:
echo   install  — Create daily scheduled task (8:00 AM)
echo   remove   — Remove the scheduled task
echo   run      — Run update manually right now
echo   status   — Check if task is installed and its status

:end
