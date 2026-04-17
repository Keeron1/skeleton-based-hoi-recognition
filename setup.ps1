Write-Host "Creating virtual environment..."
python -m venv venv

Write-Host "Upgrading pip..."
.\venv\Scripts\python -m pip install --upgrade pip

Write-Host "Installing dependencies..."
.\venv\Scripts\pip install -r requirements.txt

Write-Host "Installing Jupyter kernel..."
.\venv\Scripts\python -m pip install ipykernel
.\venv\Scripts\python -m ipykernel install --user --name=hoi-env --display-name "HOI Env"

Write-Host "Setup complete."