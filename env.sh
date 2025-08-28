#!/bin/bash
# Environment setup for Quadruped RL Training
# Requirements: Python 3.11, CUDA 12, uv package manager

# ====================================
# Python Virtual Environment (uv)
# ====================================
# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠️  No virtual environment found. Creating one with uv..."
    uv venv --python 3.11
    source .venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# ====================================
# CUDA Configuration
# ====================================
# Set CUDA paths (adjust if your CUDA is installed elsewhere)
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# CUDA device configuration
export CUDA_VISIBLE_DEVICES=0  # Use first GPU (change if needed)

# ====================================
# MuJoCo Configuration
# ====================================
# Rendering backend options:
# - osmesa: Software rendering (headless servers)
# - egl: Hardware accelerated (headless with GPU)
# - glfw: Desktop with display
export MUJOCO_GL=egl  # Changed from osmesa for better GPU utilization

# ====================================
# JAX/XLA Configuration
# ====================================
# Enable compilation cache for faster startup
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR=${JAX_CACHE_DIR:-/tmp/jax_cache}
mkdir -p $JAX_COMPILATION_CACHE_DIR

# Memory allocation settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Better for shared GPU environments
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85  # Leave some memory for other processes

# XLA optimization flags for better performance
export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true \
--xla_gpu_triton_gemm_any=true \
--xla_gpu_cuda_data_dir=$CUDA_HOME \
--xla_gpu_autotune_level=2"

# JAX specific optimizations
export JAX_ENABLE_X64=false  # Use float32 for speed (float64 not needed)
export JAX_DEBUG_NANS=false  # Disable in production for speed
export JAX_DISABLE_JIT=false  # Ensure JIT is enabled

# ====================================
# System Performance Settings
# ====================================
# OpenMP settings for CPU parallelism
export OMP_NUM_THREADS=4  # Adjust based on your CPU
export MKL_NUM_THREADS=4

# Python optimizations
export PYTHONUNBUFFERED=1  # See output in real-time

# ====================================
# Verification
# ====================================
echo "======================================"
echo "Environment Configuration Loaded"
echo "======================================"
echo "Python: $(python --version 2>&1)"
echo "CUDA: $CUDA_HOME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "MuJoCo GL: $MUJOCO_GL"
echo "JAX Cache: $JAX_COMPILATION_CACHE_DIR"
echo "======================================" 
echo ""
echo "To verify JAX GPU support, run:"
echo "  python -c 'import jax; print(jax.devices())'"
echo ""
