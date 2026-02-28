@echo off
:: Alpha-Trade — One-time setup script for a new machine
:: Run from an elevated (Administrator) command prompt
:: -------------------------------------------------------

echo ============================================================
echo  Alpha-Trade: New Machine Setup
echo ============================================================

:: 1. Install Miniconda3 via winget (silent)
echo [1/4] Installing Miniconda3 ...
winget install --id Anaconda.Miniconda3 --silent --accept-package-agreements --accept-source-agreements

:: Refresh PATH so conda is available
set "CONDA=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not exist "%CONDA%" set "CONDA=%USERPROFILE%\AppData\Local\miniconda3\Scripts\conda.exe"

:: 2. Create isolated environment
echo [2/4] Creating conda environment 'alphatrade' (Python 3.11) ...
"%CONDA%" create -n alphatrade python=3.11 -y

:: 3. Install core packages
echo [3/4] Installing packages ...
set "PIP=%USERPROFILE%\miniconda3\envs\alphatrade\Scripts\pip.exe"
if not exist "%PIP%" set "PIP=%USERPROFILE%\AppData\Local\miniconda3\envs\alphatrade\Scripts\pip.exe"
"%PIP%" install -r requirements.txt

:: 4. PyTorch — GPU vs CPU auto-detect
echo [4/4] Installing PyTorch ...
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo GPU detected — installing PyTorch with CUDA 12.1 ...
    "%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else (
    echo No GPU detected — installing CPU-only PyTorch ...
    "%PIP%" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

echo ============================================================
echo  Done!
echo  Activate env :  conda activate alphatrade
echo  Run pipeline :  python alphago_layering.py --version v8
echo ============================================================
pause
