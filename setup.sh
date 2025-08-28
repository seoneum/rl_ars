#!/bin/bash
# A100 Quick Setup Script with CUDA12 Runtime Libraries

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

# Upgrade pip
pip install --upgrade pip

# Install CUDA12 runtime libraries (required for JAX CUDA12)
echo "Installing CUDA12 runtime libraries..."
pip install -U \
    nvidia-cuda-runtime-cu12 \
    nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-cupti-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cufft-cu12 \
    nvidia-nvjitlink-cu12 \
    nvidia-nccl-cu12 \
    nvidia-cudnn-cu12

# Install JAX with CUDA12 support
echo "Installing JAX with CUDA12 support..."
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other requirements
echo "Installing other packages..."
pip install mujoco numpy tqdm

# Test JAX GPU support
echo "Testing JAX GPU support..."
python -c "
import jax
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
gpu_available = any(str(d).lower().count('gpu') > 0 or str(d).lower().count('cuda') > 0 for d in jax.devices())
if gpu_available:
    print('✅ GPU detected successfully!')
else:
    print('⚠️  No GPU detected - will use CPU')
"

echo "=== Setup Complete ==="
echo "Run: python train_a100.py"