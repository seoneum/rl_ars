#!/bin/bash
# A100 Quick Setup Script

echo "=== A100 Training Environment Setup ==="

# Install Python 3.11 if needed
if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt update
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install packages
pip install --upgrade pip
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install mujoco numpy tqdm

# Test
python -c "import jax; print('JAX devices:', jax.devices())"

echo "=== Setup Complete ==="
echo "Run: python train_a100.py"