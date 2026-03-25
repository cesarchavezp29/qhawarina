@echo off
:: =============================================================================
:: QHAWARINA — Daily Supermarket Price Scraper + Index Builder
:: Run daily at 07:00 AM via Windows Task Scheduler (backup at 14:00)
::
:: Setup (run once as Administrator):
::   schtasks /create /tn "Qhawarina-DailyPrices-AM" /tr "D:\Nexus\nexus\scripts\daily_prices.bat" /sc daily /st 07:00 /ru SYSTEM
::   schtasks /create /tn "Qhawarina-DailyPrices-PM" /tr "D:\Nexus\nexus\scripts\daily_prices.bat" /sc daily /st 14:00 /ru SYSTEM
::
:: Manual run:
::   D:\Nexus\nexus\scripts\daily_prices.bat
:: =============================================================================

set PROJECT=D:\Nexus\nexus
set LOGFILE=%PROJECT%\logs\daily_prices.log
set TODAY=%DATE:~-4%-%DATE:~3,2%-%DATE:~0,2%
set PYTHON=C:\Users\User\AppData\Local\Python\pythoncore-3.14-64\python.exe

if not exist "%PYTHON%" (
    set PYTHON=C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe
)
if not exist "%PYTHON%" (
    for /f "usebackq tokens=*" %%i in (`where python 2^>nul`) do set PYTHON=%%i & goto :found_python
    echo [%TIME%] FATAL: Python not found >> %LOGFILE%
    exit /b 1
)
:found_python

echo. >> %LOGFILE%
echo ============================================================ >> %LOGFILE%
echo [%DATE% %TIME%] STARTING DAILY PRICE PIPELINE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

:: ── Skip if snapshot for today already exists (idempotent) ──────────────────
if exist "%PROJECT%\data\raw\supermarket\snapshots\%TODAY%.parquet" (
    echo [%TIME%] Snapshot for %TODAY% already exists — skipping scrape >> %LOGFILE%
    goto :build_index
)

:: ── Step 1: Scrape all stores (up to 3 attempts) ────────────────────────────
echo [%TIME%] Step 1: Scraping supermarket prices... >> %LOGFILE%

set ATTEMPT=0
:retry_scrape
set /a ATTEMPT+=1
echo [%TIME%] Scrape attempt %ATTEMPT%/3 >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\scrape_supermarket_prices.py --store all >> %LOGFILE% 2>&1
if %errorlevel% equ 0 goto :scrape_ok
if %ATTEMPT% lss 3 (
    echo [%TIME%] Scrape failed, waiting 5 minutes before retry... >> %LOGFILE%
    timeout /t 300 /nobreak > nul
    goto :retry_scrape
)

:: All 3 attempts failed
echo [%TIME%] ERROR: Scraping failed after 3 attempts — sending alert >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\send_pipeline_alert.py "daily_prices" "Scraping failed after 3 attempts on %TODAY%" >> %LOGFILE% 2>&1
exit /b 1

:scrape_ok
echo [%TIME%] Step 1 DONE >> %LOGFILE%

:: ── Step 2: Build chain-linked price index ──────────────────────────────────
:build_index
echo [%TIME%] Step 2: Building Jevons price index... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\build_price_index.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Index build failed — sending alert >> %LOGFILE%
    %PYTHON% %PROJECT%\scripts\send_pipeline_alert.py "daily_prices" "Index build failed on %TODAY%" >> %LOGFILE% 2>&1
    exit /b 1
)
echo [%TIME%] Step 2 DONE >> %LOGFILE%

:: ── Step 3: Copy JSON to Qhawarina website ──────────────────────────────────
echo [%TIME%] Step 3: Syncing to website... >> %LOGFILE%
copy /Y %PROJECT%\exports\data\daily_price_index.json D:\qhawarina\public\assets\data\daily_price_index.json >> %LOGFILE% 2>&1
echo [%TIME%] Step 3 DONE >> %LOGFILE%

echo [%DATE% %TIME%] PIPELINE COMPLETE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%
