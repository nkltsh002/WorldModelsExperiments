Write-Host "VizDoom Experiment Simulation" -ForegroundColor Green
Write-Host "==========================" -ForegroundColor Green
Write-Host ""

# Step 1: Data Extraction
Write-Host "Step 1: Data Extraction" -ForegroundColor Cyan
python extract_test.py
Write-Host ""

# Step 2: VAE Training
Write-Host "Step 2: VAE Training" -ForegroundColor Cyan
python vae_train_test.py
Write-Host ""

# Step 3: Series Processing
Write-Host "Step 3: Series Processing" -ForegroundColor Cyan
python series_test.py
Write-Host ""

# Step 4: RNN Training
Write-Host "Step 4: RNN Training" -ForegroundColor Cyan
python rnn_train_test.py
Write-Host ""

# Step 5: Copy model files to tf_models
Write-Host "Step 5: Copying Model Files" -ForegroundColor Cyan
# Create tf_models directory if it doesn't exist
if (-not (Test-Path -Path "tf_models")) {
    New-Item -ItemType Directory -Path "tf_models"
}

# Copy the model files
Copy-Item -Path "tf_vae\vae.json" -Destination "tf_models\vae.json" -Force
Copy-Item -Path "tf_rnn\rnn.json" -Destination "tf_models\rnn.json" -Force
Copy-Item -Path "tf_initial_z\initial_z.json" -Destination "tf_models\initial_z.json" -Force
Write-Host "Model files copied to tf_models directory"
Write-Host ""

# Step 6: CMA-ES Training
Write-Host "Step 6: CMA-ES Training" -ForegroundColor Cyan
python train_test.py
Write-Host ""

# Step 7: Model Testing
Write-Host "Step 7: Model Testing" -ForegroundColor Cyan
python model_test.py
Write-Host ""

Write-Host "VizDoom Experiment Simulation Complete" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host ""
Write-Host "In an actual environment, this entire process would take approximately 15-18 hours:" -ForegroundColor Yellow
Write-Host "- Data extraction: ~5 hours on a 64-core CPU instance" -ForegroundColor Yellow
Write-Host "- VAE and RNN training: 6-8 hours on a GPU instance" -ForegroundColor Yellow
Write-Host "- CMA-ES training: 4-5 hours on a 64-core CPU instance" -ForegroundColor Yellow
Write-Host ""
Write-Host "For detailed instructions on how to run this in a real environment, see the REPLICATION_GUIDE.md file." -ForegroundColor Cyan
