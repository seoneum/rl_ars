#!/usr/bin/env python3
"""
Complete JAX CUDA fix and training runner for A100.
This script will:
1. Detect and fix JAX CUDA issues
2. Run the training with proper GPU support
"""

import subprocess
import sys
import os

def run_cmd(cmd):
    """Run command and return output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def fix_jax_cuda():
    """Fix JAX CUDA installation."""
    print("="*60)
    print(" Fixing JAX CUDA for A100")
    print("="*60)
    
    # 1. Check GPU
    print("\n[1] Checking GPU...")
    stdout, stderr, code = run_cmd("nvidia-smi --query-gpu=name --format=csv,noheader")
    if code != 0:
        print("❌ No GPU detected!")
        return False
    print(f"✅ GPU found: {stdout.strip()}")
    
    # 2. Clean old installations
    print("\n[2] Cleaning old JAX installations...")
    run_cmd("pip uninstall -y jax jaxlib")
    
    # 3. Install CUDA runtime
    print("\n[3] Installing CUDA runtime...")
    cuda_libs = [
        "nvidia-cuda-runtime-cu12",
        "nvidia-cudnn-cu12", 
        "nvidia-cublas-cu12"
    ]
    for lib in cuda_libs:
        run_cmd(f"pip install -q {lib}")
    
    # 4. Install JAX with CUDA
    print("\n[4] Installing JAX with CUDA support...")
    
    # Method A: Try standard installation
    stdout, stderr, code = run_cmd("pip install -U 'jax[cuda12]'")
    if code == 0:
        print("✅ JAX installed with CUDA 12")
    else:
        # Method B: Try specific version
        print("Trying alternative installation...")
        run_cmd("pip install jax==0.4.31 jaxlib==0.4.31+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    
    # 5. Verify
    print("\n[5] Verifying JAX CUDA...")
    test_code = """
import jax
print(f'JAX version: {jax.__version__}')
devices = jax.devices()
print(f'Devices: {devices}')
if any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices):
    print('✅ GPU detected!')
    exit(0)
else:
    print('❌ No GPU detected')
    exit(1)
"""
    
    with open("/tmp/test_jax.py", "w") as f:
        f.write(test_code)
    
    stdout, stderr, code = run_cmd("python /tmp/test_jax.py")
    print(stdout)
    
    return code == 0

def run_training():
    """Run the training script."""
    print("\n" + "="*60)
    print(" Starting Training")
    print("="*60)
    
    # Set environment for best performance
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    
    # Import and run training
    try:
        # Import here after JAX is fixed
        import jax
        import jax.numpy as jnp
        
        print(f"\nJAX Backend: {jax.default_backend()}")
        print(f"Devices: {jax.devices()}")
        
        # Now import and run the actual training
        exec(open("/home/user/webapp/train_a100.py").read())
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        
        # Fallback: Try running as subprocess
        print("\nTrying to run as subprocess...")
        stdout, stderr, code = run_cmd("cd /home/user/webapp && python train_a100.py --checkpoint_path checkpoints/a100.ckpt --batch_size 1024")
        print(stdout)
        if stderr:
            print(f"Errors: {stderr}")
        
        return code == 0
    
    return True

def main():
    """Main entry point."""
    
    # First, fix JAX CUDA
    if not fix_jax_cuda():
        print("\n⚠️  JAX CUDA fix failed, but will try to continue...")
    
    # Run training
    run_training()

if __name__ == "__main__":
    main()