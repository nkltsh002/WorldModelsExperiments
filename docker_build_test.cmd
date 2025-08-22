@echo off
echo Testing Docker Build Process
echo.

echo Step 1: Testing Docker installation...
docker --version > docker_version.txt 2>&1
if %errorlevel% neq 0 (
    echo Docker not running or not installed correctly
    exit /b 1
)

echo Step 2: Building Docker image...
docker build -t worldmodels-vizdoom -f Dockerfile.vizdoom_fixed . > docker_build.txt 2>&1
if %errorlevel% neq 0 (
    echo Docker build failed. Check docker_build.txt for details
    exit /b 1
)

echo Step 3: Verifying image...
docker images worldmodels-vizdoom > docker_images.txt 2>&1
if %errorlevel% neq 0 (
    echo Failed to list Docker images
    exit /b 1
)

echo Step 4: Testing container...
docker run -it --rm worldmodels-vizdoom python3 test_vizdoom.py > docker_test.txt 2>&1
if %errorlevel% neq 0 (
    echo Container test failed. Check docker_test.txt for details
    exit /b 1
)

echo.
echo Build process complete. Check the .txt files for detailed output.
echo.
