$ErrorActionPreference = 'Continue'
Write-Host "Starting Docker build..."
docker build --no-cache -t worldmodels-vizdoom -f Dockerfile.vizdoom . 2>&1 | Tee-Object -FilePath "docker_build.log"
if ($LASTEXITCODE -eq 0) {
    Write-Host "Build completed successfully!"
} else {
    Write-Host "Build failed. Check docker_build.log for details."
}
