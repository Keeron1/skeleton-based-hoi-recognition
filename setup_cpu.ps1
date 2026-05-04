Write-Host "Creating virtual environment..."
py -3.10 -m venv venv

Write-Host "Upgrading pip..."
.\venv\Scripts\python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch (CPU only)..."
.\venv\Scripts\python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 `
  --index-url https://download.pytorch.org/whl/cpu

Write-Host "Locking NumPy..."
.\venv\Scripts\python -m pip install "numpy>=1.26,<2"

Write-Host "Installing base dependencies..."
.\venv\Scripts\pip install -r requirements.txt --no-deps

Write-Host "Installing missing Ultralytics dependencies..."
.\venv\Scripts\pip install matplotlib psutil polars ultralytics-thop

Write-Host "Installing OpenMMLab Tools"
.\venv\Scripts\python -m pip install chumpy==0.70 --no-build-isolation
.\venv\Scripts\python -m pip install openmim
.\venv\Scripts\python -m mim install mmengine
.\venv\Scripts\python -m mim install "mmcv==2.2.0"
.\venv\Scripts\python -m mim install "mmdet>=3.0.0"
.\venv\Scripts\python -m mim install mmpose

Write-Host "Installing Jupyter kernel..."
.\venv\Scripts\python -m pip install ipykernel
.\venv\Scripts\python -m ipykernel install --user --name=hoi-env --display-name "HOI Env"

Write-Host "Setup complete."