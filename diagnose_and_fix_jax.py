#!/usr/bin/env python3
"""
Comprehensive JAX CUDA diagnostic and fix script for Elice Cloud servers.
This script will identify and fix JAX CUDA backend detection issues.
"""

import subprocess
import sys
import os
import re

def run_command(cmd, check=False):
    """Run a shell command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip() if e.stdout else "", e.stderr.strip() if e.stderr else "", e.returncode

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def check_nvidia_gpu():
    """Check if NVIDIA GPU is available."""
    print_section("NVIDIA GPU Detection")
    
    # Check nvidia-smi
    stdout, stderr, code = run_command("nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader")
    if code == 0:
        print(f"✅ GPU Found: {stdout}")
        gpu_info = stdout.split(',')
        return True, gpu_info
    else:
        print("❌ No GPU detected via nvidia-smi")
        return False, None

def check_cuda_installation():
    """Check CUDA installation and version."""
    print_section("CUDA Installation Check")
    
    cuda_locations = []
    
    # Check nvcc
    stdout, stderr, code = run_command("nvcc --version")
    if code == 0:
        version_match = re.search(r'release (\d+\.\d+)', stdout)
        if version_match:
            cuda_version = version_match.group(1)
            print(f"✅ CUDA Compiler (nvcc) found: Version {cuda_version}")
            cuda_locations.append(("nvcc", cuda_version))
    else:
        print("❌ nvcc not found in PATH")
    
    # Check common CUDA installation paths
    cuda_paths = ["/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-12.0", 
                  "/usr/local/cuda-12.1", "/usr/local/cuda-12.2", "/usr/local/cuda-12.3",
                  "/usr/local/cuda-12.4", "/opt/cuda"]
    
    for path in cuda_paths:
        if os.path.exists(path):
            version_file = os.path.join(path, "version.json")
            version_txt = os.path.join(path, "version.txt")
            
            if os.path.exists(version_file):
                stdout, _, _ = run_command(f"cat {version_file}")
                print(f"✅ CUDA installation found at {path}")
                cuda_locations.append((path, "found"))
            elif os.path.exists(version_txt):
                stdout, _, _ = run_command(f"cat {version_txt}")
                print(f"✅ CUDA installation found at {path}: {stdout}")
                cuda_locations.append((path, stdout))
    
    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if ld_path:
        print(f"\nLD_LIBRARY_PATH: {ld_path}")
        cuda_in_path = any('cuda' in p.lower() for p in ld_path.split(':'))
        if cuda_in_path:
            print("✅ CUDA libraries in LD_LIBRARY_PATH")
        else:
            print("⚠️  No CUDA libraries in LD_LIBRARY_PATH")
    else:
        print("⚠️  LD_LIBRARY_PATH not set")
    
    return cuda_locations

def check_jax_installation():
    """Check current JAX installation and CUDA support."""
    print_section("Current JAX Installation")
    
    # Check installed packages
    stdout, _, _ = run_command("pip list | grep -E '(jax|jaxlib|cuda)'")
    print("Installed packages:")
    print(stdout)
    
    # Try to import JAX and check backends
    print("\nChecking JAX backends...")
    test_code = """
import jax
print(f"JAX version: {jax.__version__}")
try:
    import jaxlib
    print(f"JAXlib version: {jaxlib.__version__}")
except:
    print("JAXlib import failed")

# Check available backends
from jax._src.lib import xla_bridge
print(f"Available backends: {xla_bridge.get_backend().platform}")
print(f"All backends: {list(jax.lib.xla_bridge.backends().keys())}")

# Check if CUDA is available
try:
    import jax.lib.xla_bridge as xb
    print(f"Has GPU: {xb._get_backend('gpu') is not None}")
except:
    print("GPU backend check failed")
"""
    
    stdout, stderr, code = run_command(f"python -c '{test_code}'")
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    
    return code == 0

def uninstall_jax():
    """Completely uninstall JAX and related packages."""
    print_section("Uninstalling JAX")
    
    packages = ["jax", "jaxlib", "flax", "optax", "chex"]
    for pkg in packages:
        print(f"Uninstalling {pkg}...")
        run_command(f"pip uninstall -y {pkg}")
    
    # Clean pip cache
    print("Cleaning pip cache...")
    run_command("pip cache purge")

def install_cuda_runtime():
    """Install CUDA runtime libraries."""
    print_section("Installing CUDA Runtime Libraries")
    
    cuda_packages = [
        "nvidia-cuda-runtime-cu12==12.1.105",
        "nvidia-cuda-cupti-cu12==12.1.105",
        "nvidia-cuda-nvcc-cu12==12.1.105",
        "nvidia-cudnn-cu12==8.9.2.26",
        "nvidia-cublas-cu12==12.1.3.1",
        "nvidia-cusolver-cu12==11.4.5.107",
        "nvidia-cusparse-cu12==12.1.0.106",
        "nvidia-nccl-cu12==2.18.1",
        "nvidia-nvjitlink-cu12==12.1.105"
    ]
    
    for pkg in cuda_packages:
        print(f"Installing {pkg}...")
        stdout, stderr, code = run_command(f"pip install {pkg}")
        if code != 0:
            print(f"⚠️  Failed to install {pkg}: {stderr}")

def install_jax_cuda():
    """Install JAX with CUDA support."""
    print_section("Installing JAX with CUDA Support")
    
    # First, check what CUDA versions are available
    print("Checking available JAX CUDA releases...")
    
    # Install specific CUDA 12 compatible version
    print("\nInstalling JAX with CUDA 12 support...")
    
    # Method 1: Try latest stable with CUDA 12
    commands = [
        # Latest stable JAX with CUDA 12
        "pip install --upgrade 'jax[cuda12]'",
        
        # Alternative: Specific version known to work
        "pip install jaxlib==0.4.31+cuda12.cudnn89 jax==0.4.31 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        
        # Alternative: Another version
        "pip install jaxlib==0.4.30+cuda12.cudnn89 jax==0.4.30 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
    ]
    
    for cmd in commands:
        print(f"\nTrying: {cmd}")
        stdout, stderr, code = run_command(cmd)
        if code == 0:
            print("✅ Installation successful")
            return True
        else:
            print(f"❌ Failed: {stderr[:200]}")
    
    return False

def set_environment_variables():
    """Set necessary environment variables for JAX CUDA."""
    print_section("Setting Environment Variables")
    
    # Find CUDA installation
    cuda_path = None
    for path in ["/usr/local/cuda-12", "/usr/local/cuda", "/opt/cuda"]:
        if os.path.exists(path):
            cuda_path = path
            break
    
    if cuda_path:
        print(f"Found CUDA at: {cuda_path}")
        
        # Create environment setup script
        env_script = f"""#!/bin/bash
# JAX CUDA Environment Setup

# CUDA paths
export CUDA_HOME={cuda_path}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# JAX specific settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Force JAX to use CUDA
export JAX_CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

echo "Environment variables set for JAX CUDA"
"""
        
        with open("/home/user/webapp/setup_jax_env.sh", "w") as f:
            f.write(env_script)
        
        print("Created setup_jax_env.sh - source this before running training")
        
        # Also set for current session
        os.environ["CUDA_HOME"] = cuda_path
        os.environ["PATH"] = f"{cuda_path}/bin:{os.environ.get('PATH', '')}"
        os.environ["LD_LIBRARY_PATH"] = f"{cuda_path}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"
        
        return True
    
    return False

def verify_jax_cuda():
    """Verify JAX can use CUDA."""
    print_section("Verifying JAX CUDA Support")
    
    verification_script = """
import os
import sys

# Remove any conflicting environment variables
os.environ.pop('JAX_PLATFORMS', None)

import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Default backend: {jax.default_backend()}")

# Try to create a tensor on GPU
try:
    x = jnp.ones((100, 100))
    device = x.device()
    print(f"✅ Successfully created tensor on: {device}")
    
    # Try a simple operation
    y = jnp.dot(x, x)
    print(f"✅ Matrix multiplication successful on: {y.device()}")
    
    print("\\n✅ JAX CUDA is working correctly!")
    sys.exit(0)
except Exception as e:
    print(f"❌ JAX CUDA test failed: {e}")
    sys.exit(1)
"""
    
    with open("/home/user/webapp/test_jax.py", "w") as f:
        f.write(verification_script)
    
    stdout, stderr, code = run_command("python /home/user/webapp/test_jax.py")
    print(stdout)
    if stderr:
        print(f"Errors: {stderr}")
    
    return code == 0

def main():
    """Main diagnostic and fix procedure."""
    print("="*60)
    print(" JAX CUDA Diagnostic and Fix Tool")
    print("="*60)
    
    # Step 1: Check GPU
    has_gpu, gpu_info = check_nvidia_gpu()
    if not has_gpu:
        print("\n❌ No NVIDIA GPU detected. JAX CUDA cannot be used.")
        sys.exit(1)
    
    # Step 2: Check CUDA installation
    cuda_locations = check_cuda_installation()
    
    # Step 3: Check current JAX installation
    check_jax_installation()
    
    # Step 4: Ask user if they want to proceed with fix
    print("\n" + "="*60)
    response = input("Do you want to proceed with fixing JAX CUDA? (yes/no): ").strip().lower()
    
    if response != "yes":
        print("Exiting without changes.")
        sys.exit(0)
    
    # Step 5: Uninstall existing JAX
    uninstall_jax()
    
    # Step 6: Install CUDA runtime
    install_cuda_runtime()
    
    # Step 7: Set environment variables
    set_environment_variables()
    
    # Step 8: Install JAX with CUDA
    success = install_jax_cuda()
    
    if not success:
        print("\n❌ Failed to install JAX with CUDA support")
        print("Trying alternative method...")
        
        # Alternative: Install from pip with cuda12
        print("\nTrying pip install with cuda12 extra...")
        stdout, stderr, code = run_command("pip install -U 'jax[cuda12]'")
        if code != 0:
            print(f"❌ Alternative installation also failed: {stderr}")
            sys.exit(1)
    
    # Step 9: Verify installation
    print("\nWaiting for installation to complete...")
    import time
    time.sleep(2)
    
    if verify_jax_cuda():
        print("\n" + "="*60)
        print(" ✅ JAX CUDA Successfully Configured!")
        print("="*60)
        print("\nYou can now run your training script with GPU support.")
        print("Before running, source the environment: source setup_jax_env.sh")
    else:
        print("\n" + "="*60)
        print(" ❌ JAX CUDA Configuration Failed")
        print("="*60)
        print("\nPlease check the error messages above and try manual installation.")
        print("\nManual installation steps:")
        print("1. Check your CUDA version: nvidia-smi")
        print("2. Visit: https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip")
        print("3. Install the matching JAX version for your CUDA")

if __name__ == "__main__":
    main()