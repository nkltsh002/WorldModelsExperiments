Write-Host "Testing the trained models..."
Write-Host "This would run: python model.py doomreal render log/doomrnn.cma.16.64.best.json"
Write-Host "This will test the model in the actual VizDoom environment and visualize the episodes"
Write-Host "`nYou can also try:"
Write-Host "python model.py doomrnn render log/doomrnn.cma.16.64.best.json (to test in the generated environment)"
Write-Host "python model.py doomreal norender log/doomrnn.cma.16.64.best.json (to run 100 episodes without visualization)"

# Uncomment this line to actually run the test
# python model.py doomreal render log/doomrnn.cma.16.64.best.json
