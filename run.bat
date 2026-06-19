@echo off
echo ===================================================
echo  CameraTrackingPose - Environment Setup (Python 3.10)
echo ===================================================

echo [0/5] Locating Python 3.10...
set PYTHON310=

:: Try py launcher first (python.org installer)
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('py -3.10 -c "import sys; print(sys.executable)"') do set PYTHON310=%%P
    goto :found_python
)

:: Try common install paths
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python310\python.exe"
    "C:\Program Files\Python310\python.exe"
    "C:\Users\User\anaconda3\envs\trackerpose\python.exe"
) do (
    if exist %%P ( set PYTHON310=%%P & goto :found_python )
)

echo ERROR: Python 3.10 not found.
echo Install via winget: winget install Python.Python.3.10
pause & exit /b 1

:found_python
echo  Found: %PYTHON310%

echo [0/5] check for Python
if not exist venv (
    %PYTHON310% -m venv venv
)

echo [1/5] Activate environment...
call venv\Scripts\activate.bat

echo [1.1/5] Re-inject and upgrade a fresh pip directly via python
python -m ensurepip --upgrade

echo [1.2/5] Upgrade pip and setuptools to the latest versions
python -m pip install --force-reinstall "setuptools>=60.2.0,<82.0.1"
python -m pip install --upgrade pip wheel
python -m pip install "numpy==1.23.5" "six>=1.11.0"

echo [2/5] Installing PyTorch 2.1.2 + CUDA 12.1...
pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ERROR: PyTorch install failed. Check your internet connection.
    pause & exit /b 1
)

python -c "import torch; print('Torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo [4.1/5] Installing chumpy...
:: The crucial fix: Disable build isolation so chumpy can use the system pip
python -m pip install "chumpy==0.70" --no-build-isolation

echo [4.2/5] Installing remaining dependencies...
python -m pip install -r requirements.txt

echo [4.3/5] Installing MMPose ecosystem...
python -m pip install -U openmim

echo [4.4/5] Installing MMPose dependencies via mim...
python -m mim install mmengine

echo [4.5/5] Installing mmcv...
python -m mim install "mmcv>=2.0.1,<2.2.0"

echo [4.6/5] Installing mmdet and mmpose...
python -m mim install "mmdet>=3.1.0,<3.3.0"
python -m mim install "mmpose>=1.3.0"

echo [5/5] checking installations...
python -c "exec('try:\n    import torch\n    print(\'>>> Torch:\', torch.__version__, \'| CUDA available:\', torch.cuda.is_available())\nexcept:\n    print(\'>>> Torch not installed or broken\')')"

python -c "exec('try:\n    import chumpy\n    print(\'>>> Chumpy:\', chumpy.__version__)\nexcept:\n    print(\'>>> Chumpy not installed or broken\')')"

python -c "exec('try:\n    import mmengine\n    print(\'>>> mmengine:\', mmengine.__version__)\nexcept:\n    print(\'>>> mmengine not installed or broken\')')"

python -c "exec('try:\n    import mmcv\n    print(\'>>> mmcv:\', mmcv.__version__)\nexcept:\n    print(\'>>> mmcv not installed or broken\')')"

python -c "exec('try:\n    import mmdet\n    print(\'>>> mmdet:\', mmdet.__version__)\nexcept:\n    print(\'>>> mmdet not installed or broken\')')"

python -c "exec('try:\n    import mmpose\n    print(\'>>> mmpose:\', mmpose.__version__)\nexcept:\n    print(\'>>> mmpose not installed or broken\')')"

echo Please review the above output to confirm all packages are installed correctly.
pause

echo.
echo ===================================================
echo  Setup complete!
echo  Activate venv  : venv\Scripts\activate
echo  Run GUI app    : run_ui.bat
echo  Run CLI        : python -m src.main --cameras 0 1
echo  Open notebook  : jupyter lab Training\pipeline.ipynb
echo ===================================================