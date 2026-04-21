@echo off
:: Force working directory to this folder
cd /d "%~dp0"
title Vehicle Proximity Detector

echo.
echo  ============================================================
echo   Vehicle Proximity Detection System
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from python.org
    pause
    exit /b 1
)
echo [OK] Python found.
echo.

:: Create venv if needed
if exist "venv\Scripts\activate.bat" (
    echo [OK] Virtual environment exists.
) else (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
    echo [OK] Created.
)
echo.

:: Activate
call venv\Scripts\activate.bat
echo [OK] Activated venv.
echo.

:: Install PyTorch
echo [SETUP] Installing PyTorch with CUDA 12.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --quiet
echo [OK] PyTorch installed.
echo.

:: Install deps
echo [SETUP] Installing dependencies...
pip install -r requirements.txt gradio imageio-ffmpeg --quiet
echo [OK] Dependencies installed.
echo.

:: CUDA check
echo [CHECK] Verifying GPU...
python -c "import torch; print(f'[OK] GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '[WARN] No CUDA')"
echo.

:: Launch
echo  ============================================================
echo   Starting server at http://localhost:7860
echo   DO NOT close this window while using the app.
echo  ============================================================
echo.
python app.py

echo.
echo  Server stopped. Press any key to close.
pause
