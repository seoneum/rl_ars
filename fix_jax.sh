#!/bin/bash
# JAX CUDA 문제 완전 해결 스크립트

echo "=== JAX CUDA 문제 해결 ==="

# 1. 기존 JAX 완전 제거
echo "1. 기존 JAX 제거 중..."
pip uninstall -y jax jaxlib

# 2. pip 캐시 정리
echo "2. 캐시 정리 중..."
pip cache purge

# 3. JAX CUDA12 버전 설치 (정확한 버전 지정)
echo "3. JAX CUDA12 설치 중..."
pip install --upgrade pip
pip install jaxlib==0.4.23+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.4.23

# 4. 테스트
echo "4. JAX 테스트..."
python3 -c "
import jax
print('JAX version:', jax.__version__)
print('Devices:', jax.devices())
"

echo "=== 완료 ==="