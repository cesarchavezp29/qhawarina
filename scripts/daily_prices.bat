@echo off
:: =============================================================================
:: QHAWARINA — Daily Supermarket Price Scraper + Index Builder
:: Run daily at 07:00 AM via Windows Task Scheduler
::
:: Setup (run once as Administrator):
::   schtasks /create /tn "Qhawarina-DailyPrices" /tr "D:\Nexus\nexus\scripts\daily_prices.bat" /sc daily /st 07:00 /ru SYSTEM
::
:: Manual run:
::   D:\Nexus\nexus\scripts\daily_prices.bat
:: =============================================================================

set PROJECT=D:\Nexus\nexus
set LOGFILE=%PROJECT%\logs\daily_prices.log
set TODAY=%DATE:~-4%-%DATE:~3,2%-%DATE:~0,2%
set PYTHON=C:\Users\User\AppData\Local\Python\bin\python.exe

echo. >> %LOGFILE%
echo ============================================================ >> %LOGFILE%
echo [%DATE% %TIME%] STARTING DAILY PRICE PIPELINE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

:: Step 1: Scrape all stores
echo [%TIME%] Step 1: Scraping supermarket prices... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\scrape_supermarket_prices.py --store all >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Scraping failed >> %LOGFILE%
    exit /b 1
)
echo [%TIME%] Step 1 DONE >> %LOGFILE%

:: Step 2: Build chain-linked price index
echo [%TIME%] Step 2: Building Jevons price index... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\build_price_index.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Index build failed >> %LOGFILE%
    exit /b 1
)
echo [%TIME%] Step 2 DONE >> %LOGFILE%

:: Step 3: Copy JSON to Qhawarina website
echo [%TIME%] Step 3: Syncing to website... >> %LOGFILE%
copy /Y %PROJECT%\exports\data\daily_price_index.json D:\qhawarina\public\assets\data\daily_price_index.json >> %LOGFILE% 2>&1
echo [%TIME%] Step 3 DONE >> %LOGFILE%

echo [%DATE% %TIME%] PIPELINE COMPLETE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%
