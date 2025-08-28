#!/bin/bash
# ===================================================================
# Quadruped RL Training 환경 설정 (서버 환경 최적화)
# - JAX CUDA12 휠은 CUDA Toolkit 경로가 없어도 드라이버만으로 동작합니다.
# - CUDA 경로는 하드코딩하지 않고, 존재할 경우에만 자동으로 감지합니다.
# ===================================================================

# ============ JAX / XLA 필수 설정 ============
# [핵심] CPU/TPU가 아닌 CUDA 백엔드를 사용하도록 강제합니다.
export JAX_PLATFORMS=cuda

# 컴파일 캐시를 활성화하여 2번째 실행부터 속도를 높입니다.
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_a100}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || true

# [핵심] VRAM 할당 정책 (A100 80GB 권장)
# VRAM의 92%를 미리 할당하여 성능을 안정화합니다.
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

# 기타 안정적인 기본값 설정
export JAX_ENABLE_X64=false
export JAX_DEBUG_NANS=false
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"

# [핵심] 과거 플래그로 인한 충돌을 방지하기 위해 XLA_FLAGS는 설정하지 않습니다.
unset XLA_FLAGS

# ============ MuJoCo 렌더링 설정 ============
# egl: GPU 가속 헤드리스 (서버), glfw: GUI (데스크탑), osmesa: 소프트웨어
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# ============ 선택 사항: GPU 지정 ============
# 특정 GPU만 사용하려면 주석을 해제하세요. (예: export CUDA_VISIBLE_DEVICES=0,1)
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ============ 선택 사항: CUDA Toolkit 자동 감지 (JAX에 불필요) ============
# JAX는 CUDA Toolkit이 필요 없지만, nvcc가 존재하면 경로를 추가해 줍니다.
if command -v nvcc >/dev/null 2>&1; then
  # nvcc 명령어 위치를 기반으로 CUDA_HOME을 추측합니다.
  CUDA_HOME_GUESS="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME="$CUDA_HOME_GUESS"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
elif [ -d /usr/local/cuda ]; then
  # 일반적인 기본 설치 경로를 확인합니다.
  export CUDA_HOME=/usr/local/cuda
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# ============ CPU / 런타임 설정 ============
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1 # Python 출력 버퍼링 비활성화

# ============ 설정 완료 및 요약 출력 ============
echo "======================================"
echo "Environment Configuration Loaded"
echo "======================================"
echo "Python:        $(python --version 2>&1)"
echo "JAX Backend:   $JAX_PLATFORMS"
echo "MuJoCo GL:     $MUJOCO_GL"
echo "JAX Cache:     $JAX_COMPILATION_CACHE_DIR"
if [ -n "$CUDA_HOME" ]; then
  echo "CUDA_HOME:     $CUDA_HOME (Auto-detected, optional for JAX)"
else
  echo "CUDA_HOME:     Not set (OK for JAX)"
fi
echo "======================================"
echo "To verify JAX GPU: python -c 'import jax; print(jax.devices())'"
echo ""
