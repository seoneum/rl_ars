#!/usr/bin/env python3
"""
Installation verification script for Quadruped RL Training
A100 80GB + JAX CUDA12 friendly:
- nvcc 없는 환경도 OK (드라이버 + jax[cuda12_pip]이면 충분)
- 실제 JAX GPU matmul 수행으로 가속 여부 판정
"""

import sys
import os
import subprocess

def check_python_version():
    v = sys.version_info
    print(f"✓ Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor < 10:
        print(f"  ⚠️  Warning: Python 3.10+ 권장 (현재 {v.major}.{v.minor})")
    if v.minor != 11:
        print("  ℹ️  Python 3.11 권장")
    return True

def check_nvidia_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if lines:
                name, mem, driver = [x.strip() for x in lines[0].split(',')]
                print(f"✓ GPU: {name} | Memory: {mem} | Driver: {driver}")
                return True
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found (컨테이너/드라이버 확인 필요)")
    return False

def check_cuda_toolkit_optional():
    # nvcc는 없어도 JAX GPU 가속 가능. 참고용으로만 출력.
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"✓ CUDA Toolkit: {line.strip()}")
                    return True
    except FileNotFoundError:
        print("ℹ️  CUDA Toolkit (nvcc) 없음 - 필수 아님")
        return False
    return False

def check_jax():
    try:
        import jax, jax.numpy as jnp
        try:
            import jaxlib
            print(f"✓ JAX: {jax.__version__} | jaxlib: {jaxlib.__version__}")
        except Exception:
            print(f"✓ JAX: {jax.__version__}")

        devices = jax.devices()
        print(f"  Devices: {devices}")
        has_gpu = any(d.platform == 'gpu' for d in devices)
        if has_gpu:
            # 실제 GPU에서 matmul 테스트
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (2048, 2048), dtype=jnp.float32)
            y = x @ x.T
            y.block_until_ready()
            dev = y.device()
            print(f"  ✓ GPU computation OK on {dev}")
            return True, True
        else:
            print("  ⚠️  No GPU devices found by JAX (CPU fallback)")
            return True, False
    except ImportError as e:
        print(f"✗ JAX not installed: {e}")
        return False, False
    except Exception as e:
        print(f"✗ JAX runtime error: {e}")
        return False, False

def check_mujoco():
    try:
        import mujoco
        print(f"✓ MuJoCo: {mujoco.__version__}")
        # MJX 존재 여부
        try:
            from mujoco import mjx  # noqa
            print("  ✓ MJX (JAX support) available")
        except Exception:
            print("  ℹ️  MJX not available (선택 사항)")

        # XML 로딩 테스트
        xml = "<mujoco><worldbody><body><geom type='sphere' size='0.1'/></body></worldbody></mujoco>"
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        # 오프스크린 렌더(환경에 따라 EGL/OSMesa)
        try:
            renderer = mujoco.Renderer(model, 320, 240)
            renderer.update_scene(data)
            _ = renderer.render()
            print("  ✓ Offscreen rendering OK")
        except Exception as e:
            print(f"  ℹ️  Rendering not tested: {e}")
        return True
    except ImportError as e:
        print(f"✗ MuJoCo not installed: {e}")
        return False
    except Exception as e:
        print(f"✗ MuJoCo runtime error: {e}")
        return False

def check_other_dependencies():
    deps = ['numpy', 'tqdm', 'matplotlib', 'scipy']
    all_ok = True
    for mod in deps:
        try:
            m = __import__(mod)
            ver = getattr(m, '__version__', 'unknown')
            print(f"✓ {mod} {ver}")
        except ImportError:
            print(f"✗ {mod} not installed")
            all_ok = False
    return all_ok

def check_project_files():
    required = ['quadruped.xml', 'mjx_ars_train.py', 'train_standing.py']
    print("\n📁 Project files:")
    all_ok = True
    for f in required:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} not found")
            all_ok = False
    return all_ok

def main():
    print("=" * 60)
    print("🔍 Quadruped RL Training - Installation Verification")
    print("=" * 60)

    results = {}
    print("\n🐍 Python Environment:")
    results['python'] = check_python_version()

    print("\n🎮 GPU / Driver:")
    results['nvidia'] = check_nvidia_gpu()
    check_cuda_toolkit_optional()  # 참고용 출력

    print("\n📦 Core Dependencies:")
    results['jax_ok'], results['jax_gpu'] = check_jax()
    results['mujoco'] = check_mujoco()
    results['other'] = check_other_dependencies()
    results['files'] = check_project_files()

    print("\n" + "=" * 60)
    print("📊 Summary:")
    print("=" * 60)

    critical_ok = results['python'] and results['jax_ok'] and results['mujoco'] and results['files']
    gpu_ok = results['jax_gpu'] and results['nvidia']

    if critical_ok:
        if gpu_ok:
            print("✅ All systems operational! JAX GPU acceleration available.")
            print("\n🚀 Ready to start training:")
            print("   python train_a100_optimized.py phase1")
        else:
            print("⚠️  System operational but GPU acceleration not confirmed.")
            print("   - nvidia-smi/drivers 또는 JAX GPU 디바이스 확인 필요")
            print("\n🚀 CPU로도 실행은 가능:")
            print("   python train_standing.py phase1")
    else:
        print("❌ Some critical dependencies are missing or invalid.")
        print("\n📚 Please follow the installation guide or setup script:")
        print("   bash setup_elice.sh")

    print("\n💡 Tips:")
    if not gpu_ok:
        print("  - 드라이버 버전과 CUDA 런타임 호환 확인 (A100 + CUDA 12)")
        print("  - JAX는 nvcc 없이도 GPU 사용 가능 (드라이버 필수)")
        print("  - env.sh에서 MUJOCO_GL=egl 설정 권장(서버/헤드리스)")
    return 0 if critical_ok else 1

if __name__ == "__main__":
    sys.exit(main())
