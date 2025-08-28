#!/usr/bin/env python3
"""
A100 80GB Optimized Training Script for JAX CUDA12
- Removes legacy/unknown XLA flags
- Uses preallocation with mem_fraction for stable performance
- Scales batch sizes based on available VRAM
"""

import os
import sys
import shutil
import subprocess

def get_total_gpu_mem_gb():
    """Returns the total memory of the most powerful GPU in GB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        if not out:
            return None
        mems = [int(x) for x in out if x.strip().isdigit()]
        if not mems:
            return None
        return max(mems) / 1024.0  # Convert MiB to GB
    except Exception:
        return None

def setup_a100_env(preallocate=True, mem_fraction=0.92):
    """Sets up environment variables for A100 on JAX CUDA12."""
    # Safe defaults for JAX CUDA12
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_a100"))
    os.makedirs(cache_dir, exist_ok=True)

    env = {
        "JAX_PLATFORMS": "gpu",
        "JAX_ENABLE_X64": "false",
        "JAX_DEBUG_NANS": "false",
        "JAX_ENABLE_COMPILATION_CACHE": "1",
        "JAX_COMPILATION_CACHE_DIR": cache_dir,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true" if preallocate else "false",
        "CUDA_LAUNCH_BLOCKING": "0",
        "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
        "TF_CPP_MIN_LOG_LEVEL": os.environ.get("TF_CPP_MIN_LOG_LEVEL", "1"),
        # Do NOT set XLA_FLAGS here to avoid unknown flag crashes with modern jaxlib
    }
    if preallocate:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)

    print("--- Setting Environment Variables ---")
    for k, v in env.items():
        os.environ[k] = v
        print(f"  Set {k}={v}")
    print("------------------------------------")


def pick_batch_sizes():
    """Chooses conservative but fast defaults based on VRAM."""
    gb = get_total_gpu_mem_gb()
    # Defaults aimed for A100 80GB; with fallbacks for other GPUs
    if gb is None:
        print("Could not detect VRAM. Using default batch sizes.")
        return dict(num_envs=1024, num_dirs=64, top_dirs=16, dir_chunk=64)
    
    print(f"Detected {gb:.1f}GB of VRAM. Adjusting batch sizes.")
    if gb >= 70:  # A100 80GB
        return dict(num_envs=1536, num_dirs=64, top_dirs=16, dir_chunk=64)
    elif gb >= 38:  # A100 40GB / similar
        return dict(num_envs=1024, num_dirs=64, top_dirs=16, dir_chunk=64)
    elif gb >= 18: # 20-24GB class GPUs
        return dict(num_envs=512, num_dirs=32, top_dirs=8, dir_chunk=32)
    else: # Lower VRAM GPUs
        return dict(num_envs=256, num_dirs=16, top_dirs=8, dir_chunk=16)

def train_phase1_a100():
    """Runs the first phase of training: learning to stand."""
    print("=" * 60)
    print("A100 80GB Optimized Phase 1: Learn to Stand")
    print("=" * 60)

    setup_a100_env(preallocate=True, mem_fraction=0.92)
    bs = pick_batch_sizes()

    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", "ars_standing_a100_phase1.npz",
        "--num-envs", str(bs["num_envs"]),
        "--num-dirs", str(bs["num_dirs"]),
        "--top-dirs", str(bs["top_dirs"]),
        "--dir-chunk", str(bs["dir_chunk"]),
        "--iterations", "180",
        "--episode-length", "200",
        "--action-repeat", "3",
        "--step-size", "0.010",
        "--noise-std", "0.015",
        "--crouch-init-ratio", "0.80",
        "--crouch-init-noise", "0.02",
        "--init-pitch", "-0.08",
        "--knee-band-low", "0.50",
        "--knee-band-high", "0.70",
        "--knee-band-weight", "2.0",
        "--knee-center", "0.60",
        "--knee-center-weight", "0.8",
        "--target-z-low", "0.42",
        "--target-z-high", "0.52",
        "--stand-bonus", "0.60",
        "--stand-shape-weight", "2.5",
        "--eval-every", "5",
        "--ckpt-every", "5",
    ]
    print("\nCommand:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def train_phase2_a100():
    """Runs the second phase of training: stabilization."""
    print("=" * 60)
    print("A100 80GB Optimized Phase 2: Stabilization")
    print("=" * 60)

    if os.path.exists("ars_standing_a100_phase1.npz") and not os.path.exists("ars_standing_a100_phase2.npz"):
        shutil.copy("ars_standing_a100_phase1.npz", "ars_standing_a100_phase2.npz")
        print("Copied Phase 1 checkpoint to Phase 2 to start.")

    setup_a100_env(preallocate=True, mem_fraction=0.92)
    bs = pick_batch_sizes()

    cmd = [
        "python", "mjx_ars_train.py",
        "--xml", "quadruped.xml",
        "--save-path", "ars_standing_a100_phase2.npz",
        "--resume",
        "--num-envs", str(bs["num_envs"]),
        "--num-dirs", str(bs["num_dirs"]),
        "--top-dirs", str(bs["top_dirs"]),
        "--dir-chunk", str(bs["dir_chunk"]),
        "--iterations", "280",
        "--episode-length", "250",
        "--action-repeat", "3",
        "--step-size", "0.006",
        "--noise-std", "0.010",
        "--crouch-init-ratio", "0.75",
        "--target-z-low", "0.45",
        "--target-z-high", "0.55",
        "--stand-bonus", "0.80",
        "--stand-shape-weight", "3.0",
        "--tilt-penalty-weight", "0.10",
        "--angvel-penalty-weight", "0.10",
        "--base-vel-penalty-weight", "0.15",
        "--eval-every", "10",
        "--ckpt-every", "10",
    ]
    print("\nCommand:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def benchmark_configs():
    """Runs a quick benchmark across different batch sizes."""
    print("=" * 60)
    print("Benchmarking Different Configurations")
    print("=" * 60)
    setup_a100_env(preallocate=True, mem_fraction=0.92)

    configs = [
        {"num_envs": 256,  "num_dirs": 16, "dir_chunk": 16, "name": "Small"},
        {"num_envs": 512,  "num_dirs": 32, "dir_chunk": 32, "name": "Medium"},
        {"num_envs": 1024, "num_dirs": 64, "dir_chunk": 64, "name": "Large"},
        {"num_envs": 1536, "num_dirs": 64, "dir_chunk": 64, "name": "A100_80G"},
        {"num_envs": 2048, "num_dirs": 64, "dir_chunk": 64, "name": "Max_Try"},
    ]

    import time
    for cfg in configs:
        print(f"\nTesting: {cfg['name']}")
        print(f"  num_envs={cfg['num_envs']} num_dirs={cfg['num_dirs']} dir_chunk={cfg['dir_chunk']}")
        cmd = [
            "python", "mjx_ars_train.py",
            "--xml", "quadruped.xml",
            "--save-path", f"benchmark_{cfg['name']}.npz",
            "--iterations", "5",
            "--num-envs", str(cfg["num_envs"]),
            "--num-dirs", str(cfg["num_dirs"]),
            "--dir-chunk", str(cfg["dir_chunk"]),
            "--episode-length", "100",
            "--eval-every", "999",
        ]
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        
        if result.returncode == 0:
            total_steps = cfg["num_envs"] * cfg["num_dirs"] * 2 * 100 * 5
            sps = total_steps / max(elapsed, 1e-6)
            print(f"  ✓ Time: {elapsed:.1f}s, Steps/sec: {int(sps):,}")
        else:
            print("  ✗ Failed (likely OOM). Try reducing num_envs or dir_chunk.")
        
        try:
            os.remove(f"benchmark_{cfg['name']}.npz")
        except OSError:
            pass

def main():
    """Main entry point for the script."""
    if len(sys.argv) > 1:
        if sys.argv[1] == "phase1":
            train_phase1_a100()
        elif sys.argv[1] == "phase2":
            train_phase2_a100()
        elif sys.argv[1] == "benchmark":
            benchmark_configs()
        else:
            print(f"Error: Unknown command '{sys.argv[1]}'")
            print("Usage: python train_a100_optimized.py [phase1|phase2|benchmark]")
            sys.exit(1)
    else:
        print("A100 80GB Optimized Training Script (JAX CUDA12)")
        print("=" * 60)
        print("This script runs training phases optimized for A100 GPUs.")
        print("\nKey features:")
        print("  - No XLA_FLAGS to avoid crashes with modern JAX.")
        print("  - Pre-allocates ~92% of VRAM for stable performance.")
        print("  - Batch sizes auto-scale based on detected VRAM.")
        print("\nUsage:")
        print("  python train_a100_optimized.py phase1")
        print("  python train_a100_optimized.py phase2")
        print("  python train_a100_optimized.py benchmark")

if __name__ == "__main__":
    main()


