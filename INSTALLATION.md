# 🚀 Installation Guide

## 📋 Prerequisites

### System Requirements
- **Python**: 3.11
- **CUDA**: 12.x (for GPU acceleration)
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (recommended)

## 🔧 Installation Steps

### 1. Install Python 3.11

#### Ubuntu/Debian:
```bash
# Add deadsnakes PPA
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update

# Install Python 3.11
sudo apt install python3.11 python3.11-venv python3.11-dev

# Verify installation
python3.11 --version
```

#### macOS:
```bash
# Using Homebrew
brew install python@3.11

# Verify installation
python3.11 --version
```

### 2. Install CUDA 12

#### Linux (Ubuntu):
```bash
# Download and install CUDA 12
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-12-0

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
nvidia-smi
```

### 3. Install uv (Fast Python Package Manager)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if not automatically added)
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify installation
uv --version
```

### 4. Create Virtual Environment with uv

```bash
# Clone the repository
git clone https://github.com/seoneum/rl_ars.git
cd rl_ars

# Checkout the standing improvement branch
git checkout standing-improvement

# Create virtual environment with Python 3.11
uv venv --python 3.11

# Activate virtual environment
source .venv/bin/activate  # On Linux/macOS
# or
.venv\Scripts\activate  # On Windows
```

### 5. Install Dependencies

```bash
# Install JAX with CUDA support
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install MuJoCo and other dependencies
uv pip install mujoco
uv pip install numpy tqdm

# Optional: Install visualization dependencies
uv pip install matplotlib opencv-python

# Verify JAX GPU support
python -c "import jax; print('GPU devices:', jax.devices())"
```

## 📦 Alternative: Using requirements.txt

Create a `requirements.txt` file:

```txt
# Core dependencies
jax[cuda12_pip]==0.4.23
jaxlib==0.4.23+cuda12.cudnn89
mujoco==3.1.1
numpy==1.24.3
tqdm==4.66.1

# Optional visualization
matplotlib==3.7.2
opencv-python==4.8.1.78
```

Then install:
```bash
uv pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## 🐳 Docker Alternative (Recommended for consistency)

### Dockerfile:
```dockerfile
FROM nvidia/cuda:12.0.0-cudnn8-runtime-ubuntu20.04

# Install Python 3.11
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Create venv and install dependencies
RUN uv venv --python 3.11 \
    && . .venv/bin/activate \
    && uv pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy application code
COPY . .

CMD [".venv/bin/python", "train_standing.py"]
```

### Build and run:
```bash
# Build Docker image
docker build -t quadruped-rl .

# Run with GPU support
docker run --gpus all -it --rm \
    -v $(pwd)/data:/app/data \
    quadruped-rl
```

## ✅ Verification

After installation, verify everything works:

```python
# verify_installation.py
import sys
print(f"Python version: {sys.version}")

import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

import mujoco
print(f"MuJoCo version: {mujoco.__version__}")

import numpy as np
print(f"NumPy version: {np.__version__}")

# Test JAX GPU
x = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
y = jax.numpy.dot(x, x.T)
print(f"JAX GPU computation successful: {y.shape}")
```

Run verification:
```bash
python verify_installation.py
```

## 🎮 Quick Start

Once everything is installed:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start training (Phase 1: Learning to stand)
python train_standing.py phase1

# Continue training (Phase 2: Stabilization)
python train_standing.py phase2

# Test the trained policy
python train_standing.py test

# Visualize the policy (requires display)
python visualize_standing.py --duration 30
```

## 🔧 Troubleshooting

### CUDA not found
- Ensure CUDA 12 is properly installed: `nvcc --version`
- Check NVIDIA driver: `nvidia-smi`
- Reinstall JAX with CUDA: `uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

### JAX not detecting GPU
- Check CUDA installation: `nvidia-smi`
- Verify CUDA paths are set correctly
- Try: `export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.0`

### MuJoCo visualization issues
- Install system dependencies: `sudo apt-get install libglfw3 libglfw3-dev`
- For headless servers, use `export MUJOCO_GL=egl` or `export MUJOCO_GL=osmesa`

### uv command not found
- Add to PATH: `export PATH="$HOME/.cargo/bin:$PATH"`
- Restart terminal or run: `source ~/.bashrc`

## 📚 Additional Resources

- [JAX Installation Guide](https://github.com/google/jax#installation)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

## 💡 Tips

1. **GPU Memory**: Monitor GPU memory usage with `nvidia-smi -l 1`
2. **Performance**: Use `XLA_FLAGS="--xla_gpu_autotune_level=2"` for better performance
3. **Debugging**: Set `JAX_TRACEBACK_FILTERING=off` for full error traces
4. **CPU Fallback**: If no GPU, JAX will automatically use CPU (slower but works)

---

For issues or questions, please open an issue on [GitHub](https://github.com/seoneum/rl_ars/issues).