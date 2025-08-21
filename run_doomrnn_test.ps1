param (
    [string]$testScript = "test_doomrnn.py"
)

# Build the Docker image
docker build -t world-models-doomrnn -f Dockerfile.doomrnn .

# Run the Docker container with the test script
docker run --rm -it world-models-doomrnn -c "cd /app && python $testScript"
