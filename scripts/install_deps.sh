#!/bin/bash -e

# Install dependencies
echo "Installing dependencies..."
sudo apt-get update
sudo apt-get install -y pkg-config
sudo apt-get install -y nasm

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
. "$HOME"/miniconda3/etc/profile.d/conda.sh

conda create -y -n main python=3.10
conda activate main

pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install 'haliax @ git+https://github.com/stanford-crfm/haliax.git'
pip install -r ../requirements.txt