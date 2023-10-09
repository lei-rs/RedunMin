#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y pkg-config
sudo apt-get install -y nasm

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -r requirements.txt