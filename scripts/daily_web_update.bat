@echo off
:: =============================================================================
:: QHAWARINA — Unified Daily Web Update
:: Updates: supermarket prices (BPP) + political instability index + FX data
:: Skips heavy DFM models (GDP/inflation/poverty — run weekly instead).
:: Runtime: ~5-10 minutes.
::
:: Setup (run once as Administrator):
::   schtasks /create /tn "Qhawarina-DailyWeb" /tr "D:\Nexus\nexus\scripts\daily_web_update.bat" /sc daily /st 08:00 /ru "%USERNAME%"
::
:: Manual run:
::   D:\Nexus\nexus\scripts\daily_web_update.bat
:: =============================================================================

set PROJECT=D:\Nexus\nexus
set WEBSITE=D:\qhawarina\public\assets\data
set EXPORTS=%PROJECT%\exports\data
set LOGFILE=%PROJECT%\logs\daily_web_update.log
set PYTHON=C:\Users\User\AppData\Local\Python\bin\python.exe

:: Change to project directory to ensure relative paths work
cd /d %PROJECT%

:: Log header
echo. >> %LOGFILE%
echo ============================================================ >> %LOGFILE%
echo [%DATE% %TIME%] STARTING DAILY WEB UPDATE >> %LOGFILE%
echo [%DATE% %TIME%] Python: %PYTHON% >> %LOGFILE%
echo [%DATE% %TIME%] Working dir: %CD% >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo [%TIME%] === DAILY WEB UPDATE STARTED ===

:: Test Python availability
%PYTHON% --version >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] FATAL: Python not found at %PYTHON% >> %LOGFILE%
    echo [%TIME%] FATAL: Python not found
    exit /b 1
)

:: ------------------------------------------------------------------
:: BLOCK A: Supermarket Prices (BPP)
:: ------------------------------------------------------------------

echo [%TIME%] [A1/2] Scraping supermarket prices (Plaza Vea / Metro / Wong)...
echo [%TIME%] [A1/2] Scraping supermarket prices... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\scrape_supermarket_prices.py --store all >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Scraper failed - skipping price index >> %LOGFILE%
    echo [%TIME%] WARNING: Scraper failed
    goto political
)
echo [%TIME%] [A1/2] DONE >> %LOGFILE%

echo [%TIME%] [A2/2] Building Jevons chain-linked price index...
echo [%TIME%] [A2/2] Building price index... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\build_price_index.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Price index failed >> %LOGFILE%
    echo [%TIME%] WARNING: Price index failed
)
echo [%TIME%] [A2/2] DONE >> %LOGFILE%

:political
:: ------------------------------------------------------------------
:: BLOCK B: Political Instability Index
:: ------------------------------------------------------------------

echo [%TIME%] [B1/2] Fetching RSS feeds + building daily political index...
echo [%TIME%] [B1/2] Building daily index... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\build_daily_index.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Daily index build failed >> %LOGFILE%
    echo [%TIME%] ERROR: Daily index build failed
    goto export
)
echo [%TIME%] [B1/2] DONE >> %LOGFILE%

:: ------------------------------------------------------------------
:: BLOCK C: Export + Sync
:: ------------------------------------------------------------------

:export
echo [%TIME%] [C1/2] Exporting daily JSON files (political + FX + prices)...
echo [%TIME%] [C1/2] Daily export... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\export_web_data.py --daily >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Export failed >> %LOGFILE%
    echo [%TIME%] ERROR: Export failed
    goto done
)
echo [%TIME%] [C1/2] DONE >> %LOGFILE%

echo [%TIME%] [C2/2] Syncing to Qhawarina website...
echo [%TIME%] [C2/2] Syncing files... >> %LOGFILE%

copy /Y %EXPORTS%\daily_price_index.json    %WEBSITE%\daily_price_index.json    >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\political_index_daily.json %WEBSITE%\political_index_daily.json >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\fx_interventions.json     %WEBSITE%\fx_interventions.json     >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\gdp_nowcast.json          %WEBSITE%\gdp_nowcast.json          >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\inflation_nowcast.json    %WEBSITE%\inflation_nowcast.json    >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\poverty_nowcast.json      %WEBSITE%\poverty_nowcast.json      >> %LOGFILE% 2>&1

echo [%TIME%] [C2/2] DONE >> %LOGFILE%

:: ------------------------------------------------------------------
:: BLOCK D: Push to GitHub so Vercel redeploys with fresh data
:: ------------------------------------------------------------------

echo [%TIME%] [D1/2] Committing updated data to GitHub...
echo [%TIME%] [D1/2] Git commit + push... >> %LOGFILE%

cd /d D:\qhawarina
git add public\assets\data\ >> %LOGFILE% 2>&1
git diff --cached --quiet
if %errorlevel% neq 0 (
    for /f "tokens=1-3 delims=/ " %%a in ("%DATE%") do set DATESTR=%%c-%%a-%%b
    git commit -m "data: daily update %DATESTR%" >> %LOGFILE% 2>&1
    git push >> %LOGFILE% 2>&1
    echo [%TIME%] [D1/2] Pushed to GitHub — Vercel redeploying >> %LOGFILE%
    echo [%TIME%] [D1/2] Pushed to GitHub
) else (
    echo [%TIME%] [D1/2] No data changes — skipping push >> %LOGFILE%
    echo [%TIME%] [D1/2] No changes to push
)

:: ------------------------------------------------------------------
:: BLOCK E: Post tweet
:: ------------------------------------------------------------------

cd /d %PROJECT%
echo [%TIME%] [E1/1] Posting daily tweet...
echo [%TIME%] [E1/1] Posting tweet... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\post_daily_tweet.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Tweet failed — check log >> %LOGFILE%
    echo [%TIME%] WARNING: Tweet failed
) else (
    echo [%TIME%] [E1/1] Tweet posted >> %LOGFILE%
)

:done
echo ============================================================ >> %LOGFILE%
echo [%DATE% %TIME%] DAILY WEB UPDATE COMPLETE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo [%TIME%] === DONE. Log: %LOGFILE% ===
