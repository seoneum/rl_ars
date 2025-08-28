#!/bin/bash
# Elice Cloud A100 완전 초기화 및 설치 스크립트

echo "==================================="
echo " Elice Cloud A100 Setup"
echo "==================================="

# 1. Python 3.11 확인
python --version

# 2. uv 설치 (빠른 패키지 매니저)
pip install uv

# 3. 가상환경 생성 및 활성화
uv venv .venv --python 3.11
source .venv/bin/activate

# 4. 핵심: JAX를 CUDA 12용으로 설치 (A100은 CUDA 12 필요)
# Elice Cloud는 CUDA 12.2가 설치되어 있음
uv pip install --upgrade pip

# 5. JAX CUDA 12 설치 - 이게 핵심이다
uv pip install jax[cuda12]

# 6. MuJoCo 설치
uv pip install mujoco mujoco-mjx tqdm

# 7. 확인
python -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
print('GPU Available:', 'gpu' in str(jax.devices()[0]).lower())
"

echo ""
echo "설치 완료!"
echo "실행: python train_a100.py"