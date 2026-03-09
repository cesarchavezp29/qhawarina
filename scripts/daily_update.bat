@echo off
REM ============================================================================
REM NEXUS — Daily high-frequency data update
REM
REM Runs only the daily-frequency steps (fast, ~10-15 min total):
REM   1. supermarket  — Scrape 3 supermarket chains + build price index
REM   2. daily_rss    — RSS political + economic instability index
REM   3. midagri      — MIDAGRI poultry bulletins (business days)
REM   4. bcrp         — BCRP monthly series (incremental)
REM   5. panel        — Rebuild national panel (incorporates new data)
REM   6. reports      — Generate daily PDF report
REM
REM Usage:
REM     daily_update.bat           — Run all daily steps
REM     daily_update.bat --check   — Dry run (show what would happen)
REM ============================================================================

set PYTHON_EXE=C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..
set UPDATE_SCRIPT=%SCRIPT_DIR%update_nexus.py

echo ============================================================
echo  NEXUS Daily Update — %date% %time%
echo ============================================================
echo.

set FAILED=0

echo [1/6] Supermarket prices (BPP)...
"%PYTHON_EXE%" "%UPDATE_SCRIPT%" --only supermarket %*
if %errorlevel% neq 0 set /a FAILED+=1

echo.
echo [2/6] Daily RSS instability index...
"%PYTHON_EXE%" "%UPDATE_SCRIPT%" --only daily_rss %*
if %errorlevel% neq 0 set /a FAILED+=1

echo.
echo [3/6] MIDAGRI poultry bulletins...
"%PYTHON_EXE%" "%UPDATE_SCRIPT%" --only midagri %*
if %errorlevel% neq 0 set /a FAILED+=1

echo.
echo [4/6] BCRP monthly series...
"%PYTHON_EXE%" "%UPDATE_SCRIPT%" --only bcrp %*
if %errorlevel% neq 0 set /a FAILED+=1

echo.
echo [5/6] Rebuild national panel...
"%PYTHON_EXE%" "%UPDATE_SCRIPT%" --only panel %*
if %errorlevel% neq 0 set /a FAILED+=1

echo.
echo [6/6] Generate daily report...
"%PYTHON_EXE%" "%UPDATE_SCRIPT%" --only reports %*
if %errorlevel% neq 0 set /a FAILED+=1

echo.
echo ============================================================
if %FAILED% equ 0 (
    echo  All 6 steps completed successfully.
) else (
    echo  %FAILED% step(s) had errors. Check logs\update_nexus.log
)
echo ============================================================
