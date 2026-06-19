@echo off
echo ===================================================
echo  CameraTrackingPose - GUI Application
echo ===================================================

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: venv not found. Run setup_venv.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
python -m src.ui_app
pause
