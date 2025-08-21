# Docker Setup for VizDoom Experiments

$ErrorActionPreference = "Stop"

Write-Host "VizDoom Experiment Docker Setup" -ForegroundColor Green
Write-Host "===============================" -ForegroundColor Green
Write-Host ""

# Check if Docker is installed
try {
    docker --version
    Write-Host "Docker is installed" -ForegroundColor Green
}
catch {
    Write-Host "Docker is not installed. Please install Docker Desktop or Docker for Windows." -ForegroundColor Red
    exit 1
}

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Cyan
docker build -t vizdoom-experiments .

Write-Host ""
Write-Host "Docker image built successfully." -ForegroundColor Green
Write-Host ""

# Menu for running different parts of the process
function Show-Menu {
    Write-Host "Choose an option:" -ForegroundColor Yellow
    Write-Host "1. Run data extraction (CPU)" -ForegroundColor Yellow
    Write-Host "2. Run VAE and RNN training (GPU)" -ForegroundColor Yellow
    Write-Host "3. Run CMA-ES training (CPU)" -ForegroundColor Yellow
    Write-Host "4. Run interactive shell" -ForegroundColor Yellow
    Write-Host "5. Exit" -ForegroundColor Yellow
    Write-Host ""
}

Show-Menu
$option = Read-Host "Enter option (1-5)"

switch ($option) {
    "1" {
        Write-Host "Running data extraction..." -ForegroundColor Cyan
        docker-compose up cpu
    }
    "2" {
        Write-Host "Running VAE and RNN training..." -ForegroundColor Cyan
        docker-compose up gpu
    }
    "3" {
        Write-Host "Running CMA-ES training..." -ForegroundColor Cyan
        docker-compose up train
    }
    "4" {
        Write-Host "Starting interactive shell..." -ForegroundColor Cyan
        docker run -it --rm -v ${PWD}:/app vizdoom-experiments
    }
    "5" {
        Write-Host "Exiting..." -ForegroundColor Cyan
        exit 0
    }
    default {
        Write-Host "Invalid option. Exiting..." -ForegroundColor Red
        exit 1
    }
}
