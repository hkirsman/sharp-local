@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv not found at "%cd%\.venv"
  echo Create the venv and install dependencies first — see docs\windows-setup.md
  exit /b 1
)

echo Installing PyInstaller into .venv (if needed^)...
".venv\Scripts\python.exe" -m pip install -q -U pyinstaller
if errorlevel 1 exit /b 1

echo Building SharpBatch...
".venv\Scripts\python.exe" -m PyInstaller packaging\sharp_batch.spec
if errorlevel 1 exit /b 1

echo Building SharpWeb...
".venv\Scripts\python.exe" -m PyInstaller packaging\sharp_web.spec
if errorlevel 1 exit /b 1

echo.
echo ========================================================================
echo Build finished OK.
echo.
echo Batch tool (GUI/CLI^):
echo   %cd%\dist\SharpBatch\SharpBatch.exe
echo   Folder: %cd%\dist\SharpBatch\
echo.
echo Web UI (Flask server — open http://127.0.0.1:8765 after starting^):
echo   %cd%\dist\SharpWeb\SharpWeb.exe
echo   Folder: %cd%\dist\SharpWeb\
echo.
echo Ship each app by zipping the whole folder above (include _internal^).
echo ========================================================================
if /i "%~1"=="nopause" (
  endlocal
  exit /b 0
)
echo.
pause
endlocal
exit /b 0
