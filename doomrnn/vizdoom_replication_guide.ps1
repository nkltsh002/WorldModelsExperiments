Write-Host "VizDoom Experiment Replication Guide" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green
Write-Host ""
Write-Host "This script will guide you through the steps to replicate the VizDoom experiment." -ForegroundColor Cyan
Write-Host "The actual process requires significant compute resources:" -ForegroundColor Cyan
Write-Host "- 64-core CPU instance with ~200GB storage and 220GB RAM for data extraction and CMA-ES training" -ForegroundColor Cyan
Write-Host "- GPU instance (P100) with ~200GB storage and 220GB RAM for VAE and RNN training" -ForegroundColor Cyan
Write-Host ""
Write-Host "The complete process on these instances would take approximately:" -ForegroundColor Yellow
Write-Host "- Data extraction: ~5 hours" -ForegroundColor Yellow
Write-Host "- VAE and RNN training: 6-8 hours" -ForegroundColor Yellow
Write-Host "- CMA-ES training: 4-5 hours (for 200 generations)" -ForegroundColor Yellow
Write-Host ""
Write-Host "The steps below are simulations of the actual commands you would run on the appropriate instances." -ForegroundColor Cyan
Write-Host ""

$step = 1
Write-Host "Step $step: Data Extraction (on 64-core CPU instance)" -ForegroundColor Green
Write-Host "------------------------------------------------" -ForegroundColor Green
Write-Host "This step generates 12,800 episode recordings (.npz files) using random policy"
Write-Host "Run this command to start extraction:"
Write-Host "   .\extract_windows.ps1" -ForegroundColor Magenta
Write-Host ""
$step++

Write-Host "Step $step: Transfer Data to GPU Instance" -ForegroundColor Green
Write-Host "--------------------------------------" -ForegroundColor Green
Write-Host "After extraction, you would transfer the generated .npz files to a GPU instance"
Write-Host "Using something like scp or gcloud tool:"
Write-Host "   scp -r doomrnn/record/* user@gpu-instance:~/WorldModelsExperiments/doomrnn/record/" -ForegroundColor Magenta
Write-Host ""
$step++

Write-Host "Step $step: Train VAE and RNN (on GPU instance)" -ForegroundColor Green
Write-Host "------------------------------------------" -ForegroundColor Green
Write-Host "This step trains the VAE, processes the data, and trains the MDN-RNN"
Write-Host "Run this command to start training:"
Write-Host "   .\gpu_jobs_windows.ps1" -ForegroundColor Magenta
Write-Host ""
$step++

Write-Host "Step $step: Copy Trained Models" -ForegroundColor Green
Write-Host "----------------------------" -ForegroundColor Green
Write-Host "After training, copy the model files to the tf_models directory:"
Write-Host "   Copy-Item -Path 'tf_vae/vae.json' -Destination 'tf_models/vae.json' -Force" -ForegroundColor Magenta
Write-Host "   Copy-Item -Path 'tf_rnn/rnn.json' -Destination 'tf_models/rnn.json' -Force" -ForegroundColor Magenta
Write-Host "   Copy-Item -Path 'tf_initial_z/initial_z.json' -Destination 'tf_models/initial_z.json' -Force" -ForegroundColor Magenta
Write-Host ""
$step++

Write-Host "Step $step: Train Controller using CMA-ES (back on 64-core CPU instance)" -ForegroundColor Green
Write-Host "--------------------------------------------------------------" -ForegroundColor Green
Write-Host "This step trains the controller inside the generated environment using CMA-ES"
Write-Host "Run this command to start CMA-ES training:"
Write-Host "   .\train_windows.ps1" -ForegroundColor Magenta
Write-Host "You would monitor the progress using the plot_training_progress.ipynb notebook"
Write-Host "The training should run for at least 200 generations (4-5 hours)"
Write-Host ""
$step++

Write-Host "Step $step: Test Trained Models" -ForegroundColor Green
Write-Host "--------------------------" -ForegroundColor Green
Write-Host "After all training is complete, test the models:"
Write-Host "   .\test_models.ps1" -ForegroundColor Magenta
Write-Host ""

Write-Host "Note: The actual commands in the scripts are commented out to prevent accidental execution" -ForegroundColor Yellow
Write-Host "      Edit the scripts to uncomment the commands when you're ready to run them" -ForegroundColor Yellow
Write-Host ""
Write-Host "For detailed visualization and analysis, use the following notebooks:" -ForegroundColor Cyan
Write-Host "- vae_test.ipynb: Visualize input/reconstruction images using the trained VAE" -ForegroundColor Cyan
Write-Host "- plot_training_progress.ipynb: Monitor CMA-ES training progress" -ForegroundColor Cyan
