# Build the Docker image
docker build -t worldmodels-cartpole -f cartpole/Dockerfile .

# Run the Docker container with the VAE training script
docker run --rm -v "${PWD}/models:/app/models" -v "${PWD}/results:/app/results" worldmodels-cartpole vae_train.py

# Run the Docker container with the RNN training script
docker run --rm -v "${PWD}/models:/app/models" -v "${PWD}/results:/app/results" worldmodels-cartpole rnn_train.py

# Run the Docker container with the controller training script
docker run --rm -v "${PWD}/models:/app/models" -v "${PWD}/results:/app/results" worldmodels-cartpole train.py

# Run the Docker container with the full model
docker run --rm -v "${PWD}/models:/app/models" -v "${PWD}/results:/app/results" worldmodels-cartpole run_model.py
