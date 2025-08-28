#!/usr/bin/env python3
"""
Installation verification script for Quadruped RL Training
Checks all required dependencies and GPU support
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major != 3 or version.minor != 11:
        print(f"  ⚠️  Warning: Python 3.11 is recommended, you have {version.major}.{version.minor}")
    return version.major == 3 and version.minor >= 10

def check_cuda():
    """Check CUDA availability"""
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            # Parse CUDA version from output
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"✓ CUDA: {line.strip()}")
                    return True
    except FileNotFoundError:
        print("✗ CUDA: nvcc not found")
        return False
    return False

def check_nvidia_gpu():
    """Check NVIDIA GPU availability"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'NVIDIA' in line and 'Driver' in line:
                    print(f"✓ GPU Driver: {line.strip()}")
                    return True
    except FileNotFoundError:
        print("✗ GPU: nvidia-smi not found")
        return False
    return False

def check_jax():
    """Check JAX installation and GPU support"""
    try:
        import jax
        print(f"✓ JAX version: {jax.__version__}")
        
        devices = jax.devices()
        print(f"  Available devices: {devices}")
        
        gpu_devices = [d for d in devices if d.device_kind != 'cpu']
        if gpu_devices:
            print(f"  ✓ GPU devices found: {len(gpu_devices)}")
            
            # Test GPU computation
            try:
                key = jax.random.PRNGKey(0)
                x = jax.random.normal(key, (1000, 1000))
                y = jax.numpy.dot(x, x.T)
                y.block_until_ready()  # Ensure computation completes
                print("  ✓ GPU computation test passed")
                return True
            except Exception as e:
                print(f"  ✗ GPU computation failed: {e}")
                return False
        else:
            print("  ⚠️  No GPU devices found, will use CPU (slower)")
            return True
    except ImportError as e:
        print(f"✗ JAX not installed: {e}")
        return False

def check_mujoco():
    """Check MuJoCo installation"""
    try:
        import mujoco
        print(f"✓ MuJoCo version: {mujoco.__version__}")
        
        # Try to import mjx
        try:
            from mujoco import mjx
            print("  ✓ MJX (JAX support) available")
        except ImportError:
            print("  ✗ MJX not available")
        
        # Test loading a simple model
        try:
            xml = """<mujoco><worldbody><body><geom type="sphere" size="0.1"/></body></worldbody></mujoco>"""
            model = mujoco.MjModel.from_xml_string(xml)
            print("  ✓ MuJoCo model loading test passed")
            return True
        except Exception as e:
            print(f"  ✗ MuJoCo model loading failed: {e}")
            return False
    except ImportError as e:
        print(f"✗ MuJoCo not installed: {e}")
        return False

def check_other_dependencies():
    """Check other required dependencies"""
    dependencies = {
        'numpy': 'NumPy',
        'tqdm': 'tqdm',
    }
    
    all_ok = True
    for module, name in dependencies.items():
        try:
            m = __import__(module)
            version = getattr(m, '__version__', 'unknown')
            print(f"✓ {name} version: {version}")
        except ImportError:
            print(f"✗ {name} not installed")
            all_ok = False
    
    return all_ok

def check_project_files():
    """Check if required project files exist"""
    required_files = [
        'quadruped.xml',
        'mjx_ars_train.py',
        'train_standing.py',
    ]
    
    print("\n📁 Project files:")
    all_ok = True
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} not found")
            all_ok = False
    
    return all_ok

def main():
    """Run all checks"""
    print("=" * 60)
    print("🔍 Quadruped RL Training - Installation Verification")
    print("=" * 60)
    
    results = {}
    
    print("\n🐍 Python Environment:")
    results['python'] = check_python_version()
    
    print("\n🎮 GPU Support:")
    results['nvidia'] = check_nvidia_gpu()
    results['cuda'] = check_cuda()
    
    print("\n📦 Core Dependencies:")
    results['jax'] = check_jax()
    results['mujoco'] = check_mujoco()
    results['other'] = check_other_dependencies()
    
    results['files'] = check_project_files()
    
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)
    
    critical_ok = results['python'] and results['jax'] and results['mujoco'] and results['files']
    gpu_ok = results['nvidia'] and results['cuda']
    
    if critical_ok:
        if gpu_ok:
            print("✅ All systems operational! GPU acceleration available.")
            print("\n🚀 Ready to start training:")
            print("   python train_standing.py phase1")
        else:
            print("⚠️  System operational but no GPU detected.")
            print("   Training will work but be significantly slower.")
            print("\n🚀 You can still start training:")
            print("   python train_standing.py phase1")
    else:
        print("❌ Some critical dependencies are missing.")
        print("\n📚 Please follow the installation guide:")
        print("   cat INSTALLATION.md")
    
    print("\n💡 Tips:")
    if not gpu_ok:
        print("  - For GPU support, ensure CUDA 12 is installed")
        print("  - Check nvidia-smi output")
    if not results['jax']:
        print("  - Install JAX with: uv pip install 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")
    if not results['mujoco']:
        print("  - Install MuJoCo with: uv pip install mujoco")
    
    return 0 if critical_ok else 1

if __name__ == "__main__":
    sys.exit(main())