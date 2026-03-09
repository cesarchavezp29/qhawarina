@echo off
:: =============================================================================
:: QHAWARINA — Daily Political + Economic Instability Index
:: Run daily at 08:00 AM via Windows Task Scheduler
::
:: Setup (run once as Administrator):
::   schtasks /create /tn "Qhawarina-DailyPolitical" /tr "D:\Nexus\nexus\scripts\daily_political.bat" /sc daily /st 08:00 /ru SYSTEM
::
:: Manual run:
::   D:\Nexus\nexus\scripts\daily_political.bat
:: =============================================================================

set PROJECT=D:\Nexus\nexus
set LOGFILE=%PROJECT%\logs\daily_political.log
set WEBSITE=D:\qhawarina\public\assets\data
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
echo [%DATE% %TIME%] STARTING DAILY POLITICAL PIPELINE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

:: Step 1: Backfill GDELT gaps (last 7 days)
echo [%TIME%] Step 1: GDELT backfill (last 7 days)... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\backfill_gdelt.py --days 7 >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: GDELT backfill failed, continuing anyway >> %LOGFILE%
)
echo [%TIME%] Step 1 DONE >> %LOGFILE%

:: Step 2: Fetch RSS + build daily index
echo [%TIME%] Step 2: RSS fetch + build daily index... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\build_daily_index.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Daily index build failed >> %LOGFILE%
    exit /b 1
)
echo [%TIME%] Step 2 DONE >> %LOGFILE%

:: Step 3: Export to website JSON
echo [%TIME%] Step 3: Exporting to website... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\export_web_data.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Export failed >> %LOGFILE%
    exit /b 1
)
echo [%TIME%] Step 3 DONE >> %LOGFILE%

:: Step 4: Copy JSON to Qhawarina website
echo [%TIME%] Step 4: Copying to website... >> %LOGFILE%
copy /Y %PROJECT%\exports\data\political_index_daily.json %WEBSITE%\political_index_daily.json >> %LOGFILE% 2>&1
echo [%TIME%] Step 4 DONE >> %LOGFILE%

echo [%DATE% %TIME%] PIPELINE COMPLETE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%
