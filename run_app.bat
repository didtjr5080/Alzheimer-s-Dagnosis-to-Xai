@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  CLIPtoXAI - clip_xai_app launcher
echo  (Merged OASIS-3+ADNI tab + OASIS-3 standalone/legacy tab)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] python was not found on PATH.
    echo Please install Python and make sure it is on PATH, then retry.
    pause
    exit /b 1
)

echo [1/2] Checking dependencies... (fast no-op if already installed)
python -m pip install -q -r clip_xai_app\requirements.txt
if errorlevel 1 (
    echo [ERROR] Dependency install failed. See the log above for details.
    pause
    exit /b 1
)

echo [2/2] Starting the app. Open http://127.0.0.1:7860 in your browser once it's ready.
echo       (Loading the models can take from a few seconds up to a minute.)
echo       Press Ctrl+C in this window to stop the server.
echo.
python clip_xai_app\app.py

echo.
echo App stopped.
pause
