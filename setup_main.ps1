Write-Host "Creating virtual environment (py 3.10.11)..."
py -3.10 -m venv venv

Write-Host "Upgrading pip..."
.\venv\Scripts\python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch with CUDA 11.8 support (RTX 3070)..."
.\venv\Scripts\python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 `
  --index-url https://download.pytorch.org/whl/cu118

Write-Host "Locking NumPy..."
.\venv\Scripts\python -m pip install "numpy>=1.26,<2"

Write-Host "Installing base dependencies..."
.\venv\Scripts\pip install -r requirements.txt

Write-Host "Installing OpenMMLab Tools"
.\venv\Scripts\python -m pip install chumpy==0.70 --no-build-isolation

.\venv\Scripts\python -m pip install openmim
.\venv\Scripts\python -m mim install mmengine
.\venv\Scripts\python -m mim install "mmcv==2.1.0"
.\venv\Scripts\python -m mim install "mmdet==3.2.0"
.\venv\Scripts\python -m mim install "mmpose==1.3.2"

Write-Host "Installing Jupyter kernel..."
.\venv\Scripts\python -m pip install ipykernel
.\venv\Scripts\python -m ipykernel install --user --name=hoi-env --display-name "HOI Env"

Write-Host "Setup complete."