#!/usr/bin/env python3
"""
A100 GPU Optimized Training Script
Maximizes GPU utilization for high-end GPUs
"""

import subprocess
import os
import sys

def setup_a100_env():
    """Set optimal environment for A100"""
    env_vars = {
        # JAX/XLA optimizations for A100
        'JAX_PLATFORMS': 'gpu',
        'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',  # Dynamic allocation
        'XLA_PYTHON_CLIENT_MEM_FRACTION': '0.95',  # Use most GPU memory
        'XLA_FLAGS': (
            '--xla_gpu_autotune_level=3 '  # Maximum autotuning
            '--xla_gpu_enable_async_collectives=true '
            '--xla_gpu_enable_latency_hiding_scheduler=true '
            '--xla_gpu_enable_highest_priority_async_stream=true '
            '--xla_gpu_force_compilation_parallelism=32 '  # A100 has many SMs
            '--xla_gpu_enable_triton_gemm=true '
            '--xla_gpu_triton_gemm_any=true'
        ),
        'JAX_ENABLE_X64': 'false',  # Use float32 for speed
        'JAX_DEBUG_NANS': 'false',  # Disable for production
        'JAX_ENABLE_COMPILATION_CACHE': '1',
        'JAX_COMPILATION_CACHE_DIR': '/tmp/jax_cache_a100',
        
        # CUDA settings
        'CUDA_LAUNCH_BLOCKING': '0',  # Async kernel launches
        'TF_FORCE_GPU_ALLOW_GROWTH': 'false',  # Preallocate
        'TF_GPU_THREAD_MODE': 'gpu_private',
        
        # MuJoCo
        'MUJOCO_GL': 'egl',  # GPU-accelerated headless rendering
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"Set {key}={value}")
    
    return env_vars

def train_phase1_a100():
    """Phase 1 training optimized for A100"""
    print("=" * 60)
    print("A100 Optimized Phase 1: Learn to Stand")
    print("=" * 60)
    
    # Set environment
    setup_a100_env()
    
    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", "ars_standing_a100_phase1.npz",
        
        # A100 optimal batch sizes (10x larger than default)
        "--num-envs", "1024",  # Was 256
        "--num-dirs", "64",     # Was 32
        "--top-dirs", "16",     # Was 8
        "--dir-chunk", "64",    # No chunking - process all at once
        
        # Training parameters
        "--iterations", "200",  # Fewer iterations needed with larger batches
        "--episode-length", "200",
        "--action-repeat", "3",
        "--step-size", "0.010",
        "--noise-std", "0.015",
        
        # Initial pose (sitting)
        "--crouch-init-ratio", "0.80",
        "--crouch-init-noise", "0.02",
        "--init-pitch", "-0.08",
        
        # Knee targets
        "--knee-band-low", "0.50",
        "--knee-band-high", "0.70",
        "--knee-band-weight", "2.0",
        "--knee-center", "0.60",
        "--knee-center-weight", "0.8",
        
        # Standing rewards
        "--target-z-low", "0.42",
        "--target-z-high", "0.52",
        "--stand-bonus", "0.60",
        "--stand-shape-weight", "2.5",
        
        # Evaluation frequency
        "--eval-every", "5",  # More frequent with fewer iterations
        "--ckpt-every", "5",
    ]
    
    print("\nCommand:", " ".join(cmd))
    subprocess.run(cmd)

def train_phase2_a100():
    """Phase 2 training optimized for A100"""
    print("=" * 60)
    print("A100 Optimized Phase 2: Stabilization")
    print("=" * 60)
    
    import shutil
    if os.path.exists("ars_standing_a100_phase1.npz") and not os.path.exists("ars_standing_a100_phase2.npz"):
        shutil.copy("ars_standing_a100_phase1.npz", "ars_standing_a100_phase2.npz")
        print("Copied Phase 1 checkpoint to Phase 2")
    
    # Set environment
    setup_a100_env()
    
    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", "ars_standing_a100_phase2.npz",
        "--resume",
        
        # A100 optimal batch sizes
        "--num-envs", "1024",
        "--num-dirs", "64",
        "--top-dirs", "16",
        "--dir-chunk", "64",
        
        # Training parameters (finer control)
        "--iterations", "300",
        "--episode-length", "250",
        "--action-repeat", "3",
        "--step-size", "0.006",
        "--noise-std", "0.010",
        
        # More stable targets
        "--crouch-init-ratio", "0.75",
        "--target-z-low", "0.45",
        "--target-z-high", "0.55",
        "--stand-bonus", "0.80",
        "--stand-shape-weight", "3.0",
        
        # Stricter penalties
        "--tilt-penalty-weight", "0.10",
        "--angvel-penalty-weight", "0.10",
        "--base-vel-penalty-weight", "0.15",
        
        "--eval-every", "10",
        "--ckpt-every", "10",
    ]
    
    print("\nCommand:", " ".join(cmd))
    subprocess.run(cmd)

def benchmark_configs():
    """Test different configurations to find optimal settings"""
    print("=" * 60)
    print("Benchmarking Different Configurations")
    print("=" * 60)
    
    configs = [
        {"num_envs": 256, "num_dirs": 16, "dir_chunk": 8, "name": "Default"},
        {"num_envs": 512, "num_dirs": 32, "dir_chunk": 16, "name": "Medium"},
        {"num_envs": 1024, "num_dirs": 32, "dir_chunk": 32, "name": "Large Batch"},
        {"num_envs": 1024, "num_dirs": 64, "dir_chunk": 64, "name": "A100 Optimal"},
        {"num_envs": 2048, "num_dirs": 32, "dir_chunk": 32, "name": "Max Batch"},
    ]
    
    setup_a100_env()
    
    for config in configs:
        print(f"\nTesting: {config['name']}")
        print(f"  num_envs={config['num_envs']}, num_dirs={config['num_dirs']}, dir_chunk={config['dir_chunk']}")
        
        cmd = [
            "python", "mjx_ars_train.py",
            "--xml", "quadruped.xml",
            "--save-path", f"benchmark_{config['name'].replace(' ', '_')}.npz",
            "--iterations", "5",  # Just a few iterations for benchmark
            "--num-envs", str(config['num_envs']),
            "--num-dirs", str(config['num_dirs']),
            "--dir-chunk", str(config['dir_chunk']),
            "--episode-length", "100",
            "--eval-every", "999",  # Skip evaluation
        ]
        
        import time
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            total_steps = config['num_envs'] * config['num_dirs'] * 2 * 100 * 5  # 2 for +/-, 100 episode, 5 iters
            steps_per_sec = total_steps / elapsed
            print(f"  ✓ Time: {elapsed:.1f}s, Steps/sec: {steps_per_sec:.0f}")
        else:
            print(f"  ✗ Failed: Check GPU memory")
        
        # Clean up
        try:
            os.remove(f"benchmark_{config['name'].replace(' ', '_')}.npz")
        except:
            pass

def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "phase1":
            train_phase1_a100()
        elif sys.argv[1] == "phase2":
            train_phase2_a100()
        elif sys.argv[1] == "benchmark":
            benchmark_configs()
        else:
            print("Usage: python train_a100_optimized.py [phase1|phase2|benchmark]")
    else:
        print("A100 Optimized Training")
        print("=" * 60)
        print("This script is optimized for NVIDIA A100 GPUs")
        print("")
        print("Key optimizations:")
        print("  - 10x larger batch sizes (1024 envs vs 128)")
        print("  - No direction chunking (process all at once)")
        print("  - Optimized XLA flags for A100 architecture")
        print("  - 95% GPU memory utilization")
        print("")
        print("Commands:")
        print("  python train_a100_optimized.py phase1    # Learn to stand")
        print("  python train_a100_optimized.py phase2    # Stabilization")
        print("  python train_a100_optimized.py benchmark # Test configurations")
        print("")
        print("Note: These settings may cause OOM on smaller GPUs!")

if __name__ == "__main__":
    main()