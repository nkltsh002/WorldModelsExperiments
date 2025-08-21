Write-Host "Starting VAE training..."
Write-Host "This would run: python vae_train.py"
Write-Host "After VAE training, the model would be saved in tf_models/vae.json"

Write-Host "`nPre-processing recorded dataset using trained VAE..."
Write-Host "This would run: python series.py"
Write-Host "A new dataset would be created in the series subdirectory"

Write-Host "`nTraining MDN-RNN using the processed dataset..."
Write-Host "This would run: python rnn_train.py"
Write-Host "This would produce models in tf_models/rnn.json and tf_models/initial_z.json"

Write-Host "`nIn a real environment, the entire process would take 6-8 hours on a GPU instance"

# Uncomment these lines to actually run the processes
# python vae_train.py
# python series.py
# python rnn_train.py
