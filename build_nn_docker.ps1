$ErrorActionPreference = "Continue"

# Create results directory
New-Item -ItemType Directory -Path "results" -Force | Out-Null

# Build the Docker image
Write-Host "Building WorldModels Neural Network Docker image..."
docker build -t worldmodels-nn -f Dockerfile.worldmodels_nn .

# Run the test script
Write-Host "Running neural network test..."
docker run --rm -v "${PWD}/results:/app/results" worldmodels-nn python /app/test_nn.py > results/nn_test_output.txt

# Display the test results
Write-Host "Test Results:"
Get-Content results/nn_test_output.txt

Write-Host "Docker image built successfully. To run an interactive shell, use:"
Write-Host "docker run --rm -it -v `"${PWD}/results:/app/results`" worldmodels-nn"
