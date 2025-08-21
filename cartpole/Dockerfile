FROM tensorflow/tensorflow:1.15.5-py3

WORKDIR /app

# Install dependencies
RUN pip install gym==0.17.3 numpy==1.18.5 cma==3.0.3 matplotlib==3.3.4

# Create directories for data and models
RUN mkdir -p /app/cartpole_data /app/models /app/results

# Copy the code
COPY cartpole/ /app/cartpole/

# Set the working directory
WORKDIR /app/cartpole

# Create a simple marker file to indicate setup is complete
RUN echo "Setup completed" > /app/results/test_complete.txt

# Set the entrypoint
ENTRYPOINT ["python"]
