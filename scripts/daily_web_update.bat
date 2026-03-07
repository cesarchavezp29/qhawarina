@echo off
:: =============================================================================
:: QHAWARINA — Unified Daily Web Update
:: Updates: supermarket prices (BPP) + political instability index + FX data
:: Skips heavy DFM models (GDP/inflation/poverty — run weekly instead).
:: Runtime: ~5-10 minutes.
::
:: Pipeline flow:
::   A:   Scrape supermarket prices
::   A.1: Validate supermarket data quality (exit 1 = bad data → skip B)
::   B:   Build Jevons price index (only if A.1 passed)
::   C:   Fetch RSS feeds + build political instability index (Claude Haiku)
::   C.1: Validate RSS data quality
::   D:   Export daily JSON files (political + FX + prices)
::   E:   Copy exports to website + copy pipeline_status.json
::   F:   Git add, commit, push (triggers Vercel redeploy)
::   G:   Send Gmail alert (success or failure summary)
::   H:   Generate daily social media charts
::   I:   Post tweet
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
set VALIDATE=%PROJECT%\scripts\validate_pipeline.py

:: Track whether supermarket data is valid for Block B
set SUPERMARKET_VALID=0

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
:: BLOCK A: Supermarket Prices — Scrape
:: ------------------------------------------------------------------

echo [%TIME%] [A] Scraping supermarket prices (Plaza Vea / Metro / Wong)...
echo [%TIME%] [A] Scraping supermarket prices... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\scrape_supermarket_prices.py --store all >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Scraper failed >> %LOGFILE%
    echo [%TIME%] WARNING: Scraper failed
    goto :rss_block
)
echo [%TIME%] [A] DONE >> %LOGFILE%

:: ------------------------------------------------------------------
:: BLOCK A.1: Validate Supermarket Data Quality
:: ------------------------------------------------------------------

echo [%TIME%] [A.1] Validating supermarket data quality...
echo [%TIME%] [A.1] Validate supermarket... >> %LOGFILE%
%PYTHON% %VALIDATE% --check supermarket >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Supermarket validation FAILED — skipping price index build >> %LOGFILE%
    echo [%TIME%] WARNING: Supermarket validation FAILED — skipping price index
    set SUPERMARKET_VALID=0
    goto :rss_block
)
set SUPERMARKET_VALID=1
echo [%TIME%] [A.1] Supermarket validation PASSED >> %LOGFILE%
echo [%TIME%] [A.1] PASSED

:: ------------------------------------------------------------------
:: BLOCK B: Build Price Index (only if A.1 passed)
:: ------------------------------------------------------------------

if "%SUPERMARKET_VALID%"=="1" (
    echo [%TIME%] [B] Building Jevons chain-linked price index...
    echo [%TIME%] [B] Building price index... >> %LOGFILE%
    %PYTHON% %PROJECT%\scripts\build_price_index.py >> %LOGFILE% 2>&1
    if %errorlevel% neq 0 (
        echo [%TIME%] WARNING: Price index failed >> %LOGFILE%
        echo [%TIME%] WARNING: Price index failed
    ) else (
        echo [%TIME%] [B] DONE >> %LOGFILE%
    )
)

:: ------------------------------------------------------------------
:: BLOCK B2: Weekly GDP + Inflation nowcast (Sundays only)
:: Skipped on weekdays — models run once per week (BCRP data monthly)
:: ------------------------------------------------------------------

for /f %%d in ('powershell -nologo -command "(Get-Date).DayOfWeek"') do set DOW=%%d
if /i "%DOW%"=="Sunday" (
    echo [%TIME%] [B2] Weekly GDP+inflation nowcast (Sunday run)...
    echo [%TIME%] [B2] GDP+inflation nowcast (Sunday)... >> %LOGFILE%
    %PYTHON% %PROJECT%\scripts\generate_nowcast.py >> %LOGFILE% 2>&1
    if %errorlevel% neq 0 (
        echo [%TIME%] WARNING: GDP/inflation nowcast failed >> %LOGFILE%
        echo [%TIME%] WARNING: GDP/inflation nowcast failed
    ) else (
        echo [%TIME%] [B2] DONE >> %LOGFILE%
    )
) else (
    echo [%TIME%] [B2] Skipping GDP nowcast (runs on Sundays only)
)

:rss_block
:: ------------------------------------------------------------------
:: BLOCK C: Political Instability Index — RSS + Claude Haiku
:: ------------------------------------------------------------------

echo [%TIME%] [C] Fetching RSS feeds + classifying with Claude Haiku...
echo [%TIME%] [C] Building daily political index... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\build_daily_index.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Daily index build failed >> %LOGFILE%
    echo [%TIME%] ERROR: Daily index build failed
    goto :export_block
)
echo [%TIME%] [C] DONE >> %LOGFILE%

:: ------------------------------------------------------------------
:: BLOCK C.1: Validate RSS Data Quality
:: ------------------------------------------------------------------

echo [%TIME%] [C.1] Validating RSS data quality...
echo [%TIME%] [C.1] Validate RSS... >> %LOGFILE%
%PYTHON% %VALIDATE% --check rss >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: RSS validation FAILED — political index may be stale >> %LOGFILE%
    echo [%TIME%] WARNING: RSS validation FAILED
) else (
    echo [%TIME%] [C.1] RSS validation PASSED >> %LOGFILE%
    echo [%TIME%] [C.1] PASSED
)

:export_block
:: ------------------------------------------------------------------
:: BLOCK D: Export + Sync JSON files
:: ------------------------------------------------------------------

echo [%TIME%] [D] Exporting daily JSON files (political + FX + prices)...
echo [%TIME%] [D] Daily export... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\export_web_data.py --daily >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] ERROR: Export failed >> %LOGFILE%
    echo [%TIME%] ERROR: Export failed
    goto :alert_block
)
echo [%TIME%] [D] DONE >> %LOGFILE%

:: ------------------------------------------------------------------
:: BLOCK E: Copy exports to website (including pipeline_status.json)
:: ------------------------------------------------------------------

echo [%TIME%] [E] Syncing to Qhawarina website...
echo [%TIME%] [E] Syncing files... >> %LOGFILE%

copy /Y %EXPORTS%\daily_price_index.json    %WEBSITE%\daily_price_index.json    >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\political_index_daily.json %WEBSITE%\political_index_daily.json >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\fx_interventions.json     %WEBSITE%\fx_interventions.json     >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\gdp_nowcast.json          %WEBSITE%\gdp_nowcast.json          >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\inflation_nowcast.json    %WEBSITE%\inflation_nowcast.json    >> %LOGFILE% 2>&1
copy /Y %EXPORTS%\poverty_nowcast.json      %WEBSITE%\poverty_nowcast.json      >> %LOGFILE% 2>&1

:: pipeline_status.json is copied by --alert email step (Block G)
:: but also copy it here in case the alert step is skipped
if exist %PROJECT%\data\pipeline_status.json (
    copy /Y %PROJECT%\data\pipeline_status.json %WEBSITE%\pipeline_status.json >> %LOGFILE% 2>&1
)

echo [%TIME%] [E] DONE >> %LOGFILE%

:: ------------------------------------------------------------------
:: BLOCK F: Git push (triggers Vercel redeploy)
:: ------------------------------------------------------------------

echo [%TIME%] [F] Committing updated data to GitHub...
echo [%TIME%] [F] Git commit + push... >> %LOGFILE%

cd /d D:\qhawarina
git add public\assets\data\ >> %LOGFILE% 2>&1
git diff --cached --quiet
if %errorlevel% neq 0 (
    for /f "tokens=1-3 delims=/ " %%a in ("%DATE%") do set DATESTR=%%c-%%a-%%b
    git commit -m "data: daily update %DATESTR%" >> %LOGFILE% 2>&1
    git push >> %LOGFILE% 2>&1
    echo [%TIME%] [F] Pushed to GitHub — Vercel redeploying >> %LOGFILE%
    echo [%TIME%] [F] Pushed to GitHub
) else (
    echo [%TIME%] [F] No data changes — skipping push >> %LOGFILE%
    echo [%TIME%] [F] No changes to push
)

:alert_block
:: ------------------------------------------------------------------
:: BLOCK G: Send Gmail alert (success or failure summary)
:: ------------------------------------------------------------------

cd /d %PROJECT%
echo [%TIME%] [G] Sending pipeline status email...
echo [%TIME%] [G] Email alert... >> %LOGFILE%
%PYTHON% %VALIDATE% --alert email >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Email alert failed — check GMAIL_ADDRESS / GMAIL_APP_PASSWORD in .env >> %LOGFILE%
    echo [%TIME%] WARNING: Email alert failed
) else (
    echo [%TIME%] [G] Email sent >> %LOGFILE%
)

:: ------------------------------------------------------------------
:: BLOCK H: Generate daily social media charts
:: ------------------------------------------------------------------

echo [%TIME%] [H] Generating daily charts...
echo [%TIME%] [H] Generating charts... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\generate_daily_charts.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Chart generation failed >> %LOGFILE%
    echo [%TIME%] WARNING: Chart generation failed
) else (
    echo [%TIME%] [H] Charts saved >> %LOGFILE%
)

:: ------------------------------------------------------------------
:: BLOCK I: Post tweet
:: ------------------------------------------------------------------

echo [%TIME%] [I] Posting daily tweet...
echo [%TIME%] [I] Posting tweet... >> %LOGFILE%
%PYTHON% %PROJECT%\scripts\post_daily_tweet.py >> %LOGFILE% 2>&1
if %errorlevel% neq 0 (
    echo [%TIME%] WARNING: Tweet failed — check log >> %LOGFILE%
    echo [%TIME%] WARNING: Tweet failed
) else (
    echo [%TIME%] [I] Tweet posted >> %LOGFILE%
)

:done
echo ============================================================ >> %LOGFILE%
echo [%DATE% %TIME%] DAILY WEB UPDATE COMPLETE >> %LOGFILE%
echo ============================================================ >> %LOGFILE%

echo [%TIME%] === DONE. Log: %LOGFILE% ===
