@echo off
:: =============================================================================
:: QHAWARINA — Local Scraper (22:00 PET)
:: Scrapes supermarket prices and pushes raw snapshot to GitHub.
:: GitHub Actions handles everything else:
::   build_price_index, build_daily_index, export, deploy to Vercel, email alert.
::
:: Schedule (run once as Administrator to update from 08:00 to 22:00):
::   schtasks /change /tn "Qhawarina-DailyWeb" /st 22:00
::
:: Manual run:
::   D:\Nexus\nexus\scripts\daily_web_update.bat
:: =============================================================================

set PROJECT=D:\Nexus\nexus
set LOGFILE=%PROJECT%\logs\daily_scraper.log
set PYTHON=C:\Users\User\AppData\Local\Python\bin\python.exe

cd /d %PROJECT%

echo. >> %LOGFILE%
echo ============================================================ >> %LOGFILE%
echo [%DATE% %TIME%] STARTING LOCAL SCRAPER >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo [%TIME%] === LOCAL SCRAPER (22:00 PET) STARTED ===

%PYTHON% --version >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] FATAL: Python not found at %PYTHON% >> %LOGFILE%
    echo [%TIME%] FATAL: Python not found
    exit /b 1
)

:: ------------------------------------------------------------------
:: BLOCK A: Scrape supermarket prices
:: ------------------------------------------------------------------

echo [%TIME%] [A] Scraping supermarket prices (Plaza Vea / Metro / Wong)...
echo [%TIME%] [A] Scraping... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\scrape_supermarket_prices.py --store all >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Scraper failed — no snapshot written >> %LOGFILE%
    echo [%TIME%] WARNING: Scraper failed
    echo [%TIME%] GitHub Actions will run on schedule at 22:30 and use yesterday's data
    goto :done
)
echo [%TIME%] [A] DONE >> %LOGFILE%

:: ------------------------------------------------------------------
:: BLOCK B: Commit snapshot and push — triggers GitHub Actions
:: ------------------------------------------------------------------

echo [%TIME%] [B] Committing snapshot and pushing to GitHub...
echo [%TIME%] [B] Git push... >> %LOGFILE%

git add data\raw\supermarket\snapshots\ >> %LOGFILE% 2>&1
git diff --cached --quiet
if %errorlevel% neq 0 (
    for /f "tokens=1-3 delims=/ " %%a in ("%DATE%") do set DATESTR=%%c-%%a-%%b
    git commit -m "scrape: %DATESTR% supermarket snapshot" >> %LOGFILE% 2>&1
    git push >> %LOGFILE% 2>&1
    echo [%TIME%] [B] Pushed — GitHub Actions now processing and deploying >> %LOGFILE%
    echo [%TIME%] [B] Pushed to GitHub — Actions triggered
) else (
    echo [%TIME%] [B] No new snapshot — nothing to push >> %LOGFILE%
    echo [%TIME%] [B] Nothing to push
)

:: ------------------------------------------------------------------
:: BLOCK C: Generate PDF reports
:: ------------------------------------------------------------------

echo [%TIME%] [C] Generating daily PDF report...
echo [%TIME%] [C] Generating daily report... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\generate_reports.py --type daily >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Daily report generation failed — check logs >> %LOGFILE%
    echo [%TIME%] WARNING: Daily report generation failed
) else (
    echo [%TIME%] [C] Daily report DONE >> %LOGFILE%
    echo [%TIME%] [C] Daily report generated
)

REM On Mondays only, generate weekly report
powershell -Command "if ((Get-Date).DayOfWeek -eq 'Monday') { & '%PYTHON%' '%PROJECT%\scripts\generate_reports.py' --type weekly }" >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Weekly report generation failed >> %LOGFILE%
) else (
    echo [%TIME%] [C] Weekly report check done >> %LOGFILE%
)

:done
echo ============================================================ >> %LOGFILE%
echo [%DATE% %TIME%] LOCAL SCRAPER COMPLETE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo [%TIME%] === DONE. GitHub Actions will email when deploy is complete ===
