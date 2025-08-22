@echo off
echo Building Docker image...
docker build -t worldmodels-vizdoom -f Dockerfile.vizdoom_minimal . > docker_build.log 2>&1
if %errorlevel% neq 0 (
    echo Build failed. Check docker_build.log for details.
    exit /b 1
)

echo Testing ViZDoom installation...
docker run -it --rm worldmodels-vizdoom python3 test_vizdoom.py

echo Running World Models...
docker run -it --rm -v %cd%/doomrnn:/app/doomrnn worldmodels-vizdoom python3 /app/doomrnn/model.py doomreal render log/doomrnn.cma.16.64.best.json
