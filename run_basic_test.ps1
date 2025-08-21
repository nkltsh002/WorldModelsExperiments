param (
    [string]$testScript = "test_basic.py"
)

# Build the Docker image
docker build -t world-models-basic -f Dockerfile.basic .

# Run the Docker container with the test script
docker run --rm -it world-models-basic -c "cd /app && python $testScript"
