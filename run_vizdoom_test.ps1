param (
    [string]$testScript = "extract.py"
)

# Build the Docker image
docker build -t vizdoom-world-models -f Dockerfile.vizdoom .

# Run the Docker container with the test script
docker run --rm -it vizdoom-world-models -c "cd /app/doomrnn && python $testScript --help"
