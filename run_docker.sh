#!/bin/bash
# Docker Setup for VizDoom Experiments

set -e

echo -e "\e[32mVizDoom Experiment Docker Setup\e[0m"
echo -e "\e[32m===============================\e[0m"
echo ""

# Check if Docker is installed
if command -v docker &> /dev/null; then
    echo -e "\e[32mDocker is installed\e[0m"
else
    echo -e "\e[31mDocker is not installed. Please install Docker.\e[0m"
    exit 1
fi

# Build the Docker image
echo -e "\e[36mBuilding Docker image...\e[0m"
docker build -t vizdoom-experiments .

echo ""
echo -e "\e[32mDocker image built successfully.\e[0m"
echo ""

# Menu for running different parts of the process
function show_menu {
    echo -e "\e[33mChoose an option:\e[0m"
    echo -e "\e[33m1. Run data extraction (CPU)\e[0m"
    echo -e "\e[33m2. Run VAE and RNN training (GPU)\e[0m"
    echo -e "\e[33m3. Run CMA-ES training (CPU)\e[0m"
    echo -e "\e[33m4. Run interactive shell\e[0m"
    echo -e "\e[33m5. Exit\e[0m"
    echo ""
}

show_menu
read -p "Enter option (1-5): " option

case $option in
    1)
        echo -e "\e[36mRunning data extraction...\e[0m"
        docker-compose up cpu
        ;;
    2)
        echo -e "\e[36mRunning VAE and RNN training...\e[0m"
        docker-compose up gpu
        ;;
    3)
        echo -e "\e[36mRunning CMA-ES training...\e[0m"
        docker-compose up train
        ;;
    4)
        echo -e "\e[36mStarting interactive shell...\e[0m"
        docker run -it --rm -v $(pwd):/app vizdoom-experiments
        ;;
    5)
        echo -e "\e[36mExiting...\e[0m"
        exit 0
        ;;
    *)
        echo -e "\e[31mInvalid option. Exiting...\e[0m"
        exit 1
        ;;
esac
