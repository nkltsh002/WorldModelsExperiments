FROM tensorflow/tensorflow:1.8.0-py3

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    zlib1g-dev \
    libsdl2-dev \
    libjpeg-dev \
    nasm \
    tar \
    libbz2-dev \
    libgtk2.0-dev \
    cmake \
    git \
    libfluidsynth-dev \
    libgme-dev \
    libopenal-dev \
    timidity \
    libwildmidi-dev \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir numpy==1.13.3 gym==0.9.4 cma==2.2.0 mpi4py==2.0.0 scipy==1.2.0

# Install VizDoom
RUN pip install --no-cache-dir doom-py
RUN pip install --no-cache-dir git+https://github.com/ppaquette/gym-doom.git

# Copy the code
COPY . /app

# Set the working directory
WORKDIR /app/doomrnn

# Create directory for recordings
RUN mkdir -p record

# Entry point
ENTRYPOINT ["/bin/bash"]
