param (
    [string]$testScript = "test_env.py"
)

# Build the Docker image
docker build -t world-models-tf -f Dockerfile.tensorflow .

# Run the Docker container with the test script
docker run --rm -it world-models-tf -c "cd /app && python $testScript"
