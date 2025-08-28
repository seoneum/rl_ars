#!/bin/bash

echo "======================================"
echo " Quick JAX CUDA Fix for Elice Cloud"
echo "======================================"

# Step 1: Complete cleanup
echo -e "\n[1/5] Cleaning up existing JAX installation..."
pip uninstall -y jax jaxlib flax optax chex
pip cache purge

# Step 2: Install CUDA 12 runtime (minimal required packages)
echo -e "\n[2/5] Installing CUDA 12 runtime libraries..."
pip install -q \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cudnn-cu12==8.9.2.26 \
    nvidia-cublas-cu12==12.1.3.1

# Step 3: Set minimal environment
echo -e "\n[3/5] Setting environment variables..."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
unset JAX_PLATFORMS
unset XLA_FLAGS

# Step 4: Install JAX with CUDA 12 support (most compatible version)
echo -e "\n[4/5] Installing JAX with CUDA 12 support..."

# Try the simplest method first
echo "Attempting standard installation..."
pip install -U "jax[cuda12]"

# Step 5: Quick verification
echo -e "\n[5/5] Verifying JAX CUDA support..."
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'Devices: {jax.devices()}')
try:
    import jax.numpy as jnp
    x = jnp.ones((10, 10))
    print(f'✅ JAX CUDA is working! Device: {x.device()}')
except Exception as e:
    print(f'❌ JAX CUDA test failed: {e}')
"

echo -e "\n======================================"
echo " Installation complete!"
echo "======================================"
echo ""
echo "If GPU is not detected, try:"
echo "1. Restart Python/terminal session"
echo "2. Run: source /home/user/webapp/setup_jax_env.sh"
echo "3. Check GPU with: nvidia-smi"