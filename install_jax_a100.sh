#!/bin/bash
# Minimal JAX CUDA installation for A100

echo "Installing JAX for A100 (CUDA 12)..."

# 1. Clean everything
pip uninstall -y jax jaxlib 2>/dev/null

# 2. Install the exact version that works on A100
pip install --upgrade pip

# 3. Install JAX with CUDA 12 - This WILL work on A100
pip install jax[cuda12]

# If that fails, try explicit version
if [ $? -ne 0 ]; then
    echo "Trying explicit CUDA 12 version..."
    pip install https://storage.googleapis.com/jax-releases/cuda12/jaxlib-0.4.31+cuda12.cudnn91-cp311-cp311-manylinux2014_x86_64.whl
    pip install jax==0.4.31
fi

# 4. Test
python -c "import jax; print('JAX devices:', jax.devices())"

echo "Done! Now run: python train_a100.py"