# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

# Install required packages
pip install fastapi==0.111.1
pip install uvicorn==0.23.2
pip install scikit-learn==1.5.1
pip install numpy
pip install pydantic
pip install joblib
pip install python-multipart
pip install python-dateutil

# Create model directory
New-Item -ItemType Directory -Force -Path .\model

# Print completion message
Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "To activate the virtual environment, run: .\venv\Scripts\Activate.ps1" -ForegroundColor Cyan
