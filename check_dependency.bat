echo [1/2] Activate environment...
call venv\Scripts\activate.bat

echo [2/2] checking installations...
python -c "exec('try:\n    import torch\n    print(\'>>> Torch:\', torch.__version__, \'| CUDA available:\', torch.cuda.is_available())\nexcept:\n    print(\'>>> Torch not installed or broken\')')"

python -c "exec('try:\n    import chumpy\n    print(\'>>> Chumpy:\', chumpy.__version__)\nexcept:\n    print(\'>>> Chumpy not installed or broken\')')"

python -c "exec('try:\n    import mmengine\n    print(\'>>> mmengine:\', mmengine.__version__)\nexcept:\n    print(\'>>> mmengine not installed or broken\')')"

python -c "exec('try:\n    import mmcv\n    print(\'>>> mmcv:\', mmcv.__version__)\nexcept:\n    print(\'>>> mmcv not installed or broken\')')"

python -c "exec('try:\n    import mmdet\n    print(\'>>> mmdet:\', mmdet.__version__)\nexcept:\n    print(\'>>> mmdet not installed or broken\')')"

python -c "exec('try:\n    import mmpose\n    print(\'>>> mmpose:\', mmpose.__version__)\nexcept:\n    print(\'>>> mmpose not installed or broken\')')"