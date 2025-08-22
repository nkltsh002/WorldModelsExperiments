$ErrorActionPreference = "Continue"

# Function to run command and save output
function Run-CommandWithLog {
    param($Command, $LogFile)
    Write-Host "Running: $Command"
    Write-Host "Output will be saved to: $LogFile"
    try {
        $output = Invoke-Expression $Command *>&1
        $output | Out-File -FilePath $LogFile
        return $true
    }
    catch {
        $_.Exception.Message | Out-File -FilePath $LogFile -Append
        return $false
    }
}

# Check Docker service
Write-Host "Checking Docker service..."
Run-CommandWithLog "Get-Service docker" "docker_service_status.log"

# Start Docker if needed
Write-Host "Starting Docker service..."
Start-Service docker
Start-Sleep -Seconds 5

# Test Docker
Write-Host "Testing Docker with hello-world..."
Run-CommandWithLog "docker run hello-world" "docker_test.log"

# Build image
Write-Host "Building Docker image..."
Run-CommandWithLog "docker build -t worldmodels-vizdoom -f Dockerfile.vizdoom_fixed ." "docker_build.log"

# Check image
Write-Host "Checking if image was created..."
Run-CommandWithLog "docker images worldmodels-vizdoom" "docker_images.log"

# Test container
Write-Host "Testing container..."
Run-CommandWithLog "docker run -it --rm worldmodels-vizdoom python3 test_vizdoom.py" "docker_test_container.log"

Write-Host "`nAll done! Check the .log files for details."
