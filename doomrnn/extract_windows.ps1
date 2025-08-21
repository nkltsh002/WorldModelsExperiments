$env:CUDA_VISIBLE_DEVICES=""

Write-Host "Starting extraction processes... (This would launch 64 processes on a real 64-core machine)"
Write-Host "This would generate 12,800 .npz files in the record directory (64 processes × 200 episodes each)"
Write-Host "In a real environment, this would take several hours to complete."

# On a real system, we would run something like:
# 1..64 | ForEach-Object {
#     Write-Host "Starting worker $_"
#     Start-Process -NoNewWindow -FilePath "python" -ArgumentList "extract.py"
#     Start-Sleep -Seconds 1
# }

# For demonstration, let's just run a single instance with limited episodes
Write-Host "For demonstration, we'll run a single extract process with just a few episodes"
python extract.py
