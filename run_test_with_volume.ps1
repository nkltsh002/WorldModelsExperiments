$ErrorActionPreference = "Continue"

# Create results directory if it doesn't exist
New-Item -ItemType Directory -Path "results" -Force | Out-Null

# Build the Docker image
docker build -t world-models-test-volume -f Dockerfile.test_volume .

# Run the Docker container with volume mounted
docker run --rm -v "${PWD}/results:/app/results" world-models-test-volume

# Show test results
Write-Host "Test results:"
if (Test-Path "results/test_results.txt") {
    Get-Content "results/test_results.txt"
} else {
    Write-Host "No test results file found."
}
