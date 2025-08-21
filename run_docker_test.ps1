# Run VizDoom Docker Test

Write-Host "Building the Docker image for VizDoom test..." -ForegroundColor Cyan
docker build -t vizdoom-test -f Dockerfile.small .

Write-Host "`nRunning the test container..." -ForegroundColor Cyan
docker run --rm -v "${PWD}/doomrnn:/app/doomrnn" vizdoom-test -c "cd /app && python /app/doomrnn/test_docker.py"

Write-Host "`nRunning a small extraction test..." -ForegroundColor Cyan
docker run --rm -v "${PWD}/doomrnn:/app/doomrnn" vizdoom-test -c "cd /app && python /app/doomrnn/extract_small_docker.py"

Write-Host "`nTests completed!" -ForegroundColor Green
