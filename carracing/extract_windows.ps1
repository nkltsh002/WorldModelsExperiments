# PowerShell script for extracting random episodes on Windows
# Start 4 parallel processes for data collection

$numWorkers = 4  # Adjust based on your CPU cores

for ($i=1; $i -le $numWorkers; $i++) {
    Write-Host "Starting worker $i"
    Start-Process -FilePath "python" -ArgumentList "extract.py" -WindowStyle Minimized
    Start-Sleep -Seconds 2.0  # Give time for process to start
}

Write-Host "All workers started. Check the 'record' folder for saved episodes."
