#!/usr/bin/env python3
"""
GPU Utilization Benchmark and Diagnosis Tool
Identifies bottlenecks in the training pipeline
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure JAX uses GPU
os.environ['JAX_PLATFORMS'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jax import random, jit, vmap
import mujoco
from mujoco import mjx

def print_gpu_info():
    """Print GPU information"""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    
    for i, device in enumerate(devices):
        print(f"Device {i}: {device.device_kind} - {device}")
        
        if device.device_kind == 'gpu':
            # Try to get GPU memory info
            try:
                from jax.lib import xla_bridge
                backend = xla_bridge.get_backend()
                print(f"  Platform: {backend.platform}")
                
                # Get compute capability for NVIDIA GPUs
                if hasattr(device, 'compute_capability'):
                    print(f"  Compute capability: {device.compute_capability}")
            except:
                pass
    
    # Try nvidia-smi for more info
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', 
                               '--format=csv,noheader'], capture_output=True, text=True)
        if result.returncode == 0:
            print("\nNVIDIA GPU Details:")
            print(result.stdout)
    except:
        pass
    
    print("=" * 60)

def benchmark_batch_sizes(xml_path):
    """Benchmark different batch sizes to find optimal configuration"""
    print("\n" + "=" * 60)
    print("Batch Size Benchmark")
    print("=" * 60)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(xml_path)
    model_mjx = mjx.put_model(model)
    
    batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
    episode_length = 100
    
    results = []
    
    for batch_size in batch_sizes:
        try:
            print(f"\nTesting batch_size={batch_size}...")
            
            # Create batched environment
            @jit
            @vmap
            def reset_fn(key):
                data = mjx.make_data(model_mjx)
                return data
            
            @jit
            @vmap
            def step_fn(data, action):
                data = mjx.step(model_mjx, data)
                return data
            
            # Initialize
            keys = random.split(random.PRNGKey(0), batch_size)
            data_batch = reset_fn(keys)
            actions = jnp.zeros((batch_size, model.nu))
            
            # Warmup
            for _ in range(10):
                data_batch = step_fn(data_batch, actions)
            data_batch.qpos.block_until_ready()
            
            # Benchmark
            start_time = time.time()
            for _ in range(episode_length):
                data_batch = step_fn(data_batch, actions)
            data_batch.qpos.block_until_ready()  # Wait for completion
            elapsed = time.time() - start_time
            
            steps_per_sec = (batch_size * episode_length) / elapsed
            gpu_efficiency = steps_per_sec / batch_size  # Steps per env per sec
            
            results.append({
                'batch_size': batch_size,
                'time': elapsed,
                'steps_per_sec': steps_per_sec,
                'efficiency': gpu_efficiency
            })
            
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Steps/sec: {steps_per_sec:.0f}")
            print(f"  Efficiency: {gpu_efficiency:.1f} steps/env/sec")
            
        except Exception as e:
            print(f"  Failed: {e}")
            break
    
    # Find optimal batch size
    if results:
        best = max(results, key=lambda x: x['steps_per_sec'])
        print(f"\n🎯 Optimal batch size: {best['batch_size']} ({best['steps_per_sec']:.0f} steps/sec)")
    
    return results

def benchmark_computation():
    """Benchmark pure computation to test GPU utilization"""
    print("\n" + "=" * 60)
    print("Pure Computation Benchmark")
    print("=" * 60)
    
    sizes = [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192)]
    
    for size in sizes:
        print(f"\nMatrix multiplication {size[0]}x{size[1]}:")
        
        # Create random matrices
        key = random.PRNGKey(0)
        A = random.normal(key, size, dtype=jnp.float32)
        B = random.normal(random.split(key)[0], size, dtype=jnp.float32)
        
        # Compile
        @jit
        def matmul(A, B):
            return jnp.dot(A, B)
        
        # Warmup
        C = matmul(A, B)
        C.block_until_ready()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            C = matmul(A, B)
        C.block_until_ready()
        elapsed = time.time() - start
        
        flops = 10 * 2 * size[0] * size[1] * size[1]  # 10 iterations
        tflops = flops / elapsed / 1e12
        
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Performance: {tflops:.2f} TFLOPS")
        
        # Memory usage estimate
        memory_gb = (2 * size[0] * size[1] * 4) / 1e9  # 2 matrices, 4 bytes per float32
        print(f"  Memory used: ~{memory_gb:.1f} GB")

def analyze_bottlenecks():
    """Analyze common bottlenecks"""
    print("\n" + "=" * 60)
    print("Bottleneck Analysis")
    print("=" * 60)
    
    issues = []
    
    # Check 1: Batch size
    print("\n1. Batch Size Analysis:")
    print("   Current defaults: num_envs=128, num_dirs=16")
    print("   Total parallel: 128 * 16 * 2 = 4096 rollouts")
    print("   ⚠️  For A100: Consider num_envs=512 or 1024")
    print("   ⚠️  For 3060: num_envs=128-256 is reasonable")
    issues.append("batch_size")
    
    # Check 2: Memory allocation
    preallocate = os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE', 'true')
    if preallocate == 'true':
        print("\n2. Memory Preallocation: ENABLED")
        print("   ⚠️  This can cause issues on shared GPUs")
        print("   Fix: export XLA_PYTHON_CLIENT_PREALLOCATE=false")
        issues.append("memory_preallocation")
    else:
        print("\n2. Memory Preallocation: DISABLED ✓")
    
    # Check 3: Compilation cache
    cache = os.environ.get('JAX_ENABLE_COMPILATION_CACHE', '0')
    if cache != '1':
        print("\n3. Compilation Cache: DISABLED")
        print("   ⚠️  Recompilation overhead on every run")
        print("   Fix: export JAX_ENABLE_COMPILATION_CACHE=1")
        issues.append("compilation_cache")
    else:
        print("\n3. Compilation Cache: ENABLED ✓")
    
    # Check 4: Dir chunking
    print("\n4. Direction Chunking:")
    print("   Current: dir_chunk=8 (processes 8 directions at a time)")
    print("   ⚠️  This serializes computation!")
    print("   For A100: Remove chunking or set dir_chunk=32")
    print("   For 3060: dir_chunk=16 might be better")
    issues.append("dir_chunking")
    
    # Check 5: Policy network size
    print("\n5. Policy Network:")
    print("   Current: Linear (obs_dim x act_dim)")
    print("   Typical: ~(29 x 8) = 232 parameters")
    print("   ⚠️  Too small to utilize GPU effectively!")
    print("   Consider: Larger networks or more complex policies")
    issues.append("network_size")
    
    return issues

def suggest_optimizations(gpu_type='unknown'):
    """Suggest optimizations based on GPU type"""
    print("\n" + "=" * 60)
    print(f"Optimization Suggestions for {gpu_type}")
    print("=" * 60)
    
    if 'A100' in gpu_type or 'a100' in gpu_type.lower():
        print("""
🚀 A100 Optimizations:

1. Increase batch sizes:
   --num-envs 1024
   --num-dirs 64
   --dir-chunk 64  (or remove chunking)

2. Environment variables:
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
   export JAX_ENABLE_X64=false
   export XLA_FLAGS="--xla_gpu_autotune_level=3"

3. Training command:
   python mjx_ars_train.py \\
     --xml quadruped.xml \\
     --num-envs 1024 \\
     --num-dirs 64 \\
     --dir-chunk 64 \\
     --episode-length 200 \\
     --iterations 1000

4. Consider MLP policy instead of linear for better GPU usage
""")
    
    elif '3060' in gpu_type or '3070' in gpu_type or '3080' in gpu_type:
        print("""
💻 RTX 3060/3070/3080 Optimizations:

1. Moderate batch sizes:
   --num-envs 256
   --num-dirs 32
   --dir-chunk 16

2. Environment variables:
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
   export JAX_ENABLE_X64=false

3. Training command:
   python mjx_ars_train.py \\
     --xml quadruped.xml \\
     --num-envs 256 \\
     --num-dirs 32 \\
     --dir-chunk 16 \\
     --episode-length 200
""")
    
    else:
        print("""
📊 General GPU Optimizations:

1. Find optimal batch size:
   python benchmark_gpu.py --benchmark-batch

2. Monitor GPU usage:
   nvidia-smi -l 1  (in separate terminal)

3. Key parameters to tune:
   - num_envs: Number of parallel environments
   - num_dirs: Number of ARS directions
   - dir_chunk: How many directions to process at once
""")

def main():
    parser = argparse.ArgumentParser(description='GPU Benchmark and Diagnosis')
    parser.add_argument('--xml', default='quadruped.xml', help='Model XML file')
    parser.add_argument('--benchmark-batch', action='store_true', help='Run batch size benchmark')
    parser.add_argument('--benchmark-compute', action='store_true', help='Run computation benchmark')
    parser.add_argument('--analyze', action='store_true', help='Analyze bottlenecks')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    
    args = parser.parse_args()
    
    # Print GPU info
    print_gpu_info()
    
    # Detect GPU type
    gpu_type = 'unknown'
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_type = result.stdout.strip()
    except:
        pass
    
    if args.all or args.benchmark_compute:
        benchmark_computation()
    
    if args.all or args.benchmark_batch:
        if os.path.exists(args.xml):
            benchmark_batch_sizes(args.xml)
        else:
            print(f"Warning: {args.xml} not found, skipping batch benchmark")
    
    if args.all or args.analyze:
        analyze_bottlenecks()
    
    # Always show suggestions
    suggest_optimizations(gpu_type)
    
    print("\n" + "=" * 60)
    print("Diagnosis complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()