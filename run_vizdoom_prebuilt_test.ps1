param (
    [string]$testScript = "test_vizdoom.py"
)

# Build the Docker image
docker build -t vizdoom-prebuilt -f Dockerfile.vizdoom_prebuilt .

# Run the Docker container with the test script
docker run --rm -it vizdoom-prebuilt -c "cd /app && python3 $testScript"
