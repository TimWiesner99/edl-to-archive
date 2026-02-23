@echo off
REM ── EDL-to-Archive Launcher (Windows) ──────────────────────────────
REM This script checks for uv (Python package manager), installs it if
REM needed, then launches the application. uv automatically handles
REM downloading the correct Python version and installing dependencies.
REM ────────────────────────────────────────────────────────────────────

cd /d "%~dp0"

REM Check if uv is already on PATH
where uv >nul 2>&1
if %errorlevel% equ 0 goto :run

REM Not on PATH — check common install location
if exist "%USERPROFILE%\.local\bin\uv.exe" goto :add_to_path

REM Not installed — install it
echo uv is not installed. Installing uv...
echo.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
if %errorlevel% neq 0 (
    echo.
    echo Failed to install uv automatically.
    echo Please install uv manually from: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)
echo.
echo uv installed successfully.
echo.

:add_to_path
set "PATH=%USERPROFILE%\.local\bin;%PATH%"

:run
echo Starting EDL-to-Archive...
echo (first launch may take a moment while dependencies are installed)
echo.
uv run gui.py

if %errorlevel% neq 0 (
    echo.
    echo The application exited with an error.
    pause
)
