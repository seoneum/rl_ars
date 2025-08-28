# Elice Cloud A100 서버 처음부터 설정하기

## 문제의 핵심
Elice Cloud는 시스템 Python에 이미 여러 패키지가 설치되어 있어서 충돌이 발생합니다.
**반드시 가상환경을 만들어서 깨끗한 상태에서 시작해야 합니다.**

## 완전 초기화 방법 (이것만 따라하세요)

### 1. 서버 접속 후 프로젝트 폴더 생성
```bash
mkdir ~/quadruped_rl
cd ~/quadruped_rl
```

### 2. 파일 복사
```bash
# quadruped.xml 복사
# train_a100.py 복사
```

### 3. uv로 가상환경 생성 (중요!)
```bash
# uv 설치
pip install uv

# Python 3.11 가상환경 생성
uv venv .venv --python 3.11

# 가상환경 활성화
source .venv/bin/activate
```

### 4. JAX CUDA 12 설치 (핵심!)
```bash
# pip 업그레이드
uv pip install --upgrade pip

# JAX CUDA 12 버전 설치 (A100은 CUDA 12 필요)
uv pip install jax[cuda12]

# 나머지 패키지
uv pip install mujoco mujoco-mjx tqdm
```

### 5. 확인
```bash
python -c "import jax; print(jax.devices())"
# 출력: [cuda:0]  <- 이게 나와야 정상
```

### 6. 실행
```bash
python train_a100.py
```

## 주의사항

1. **절대 시스템 Python에 직접 설치하지 마세요**
2. **가상환경 활성화를 잊지 마세요** (`source .venv/bin/activate`)
3. **jax[cuda12]로 설치해야 합니다** (jaxlib 따로 설치 X)

## 만약 그래도 안 되면

시스템 CUDA 버전 확인:
```bash
nvcc --version  # CUDA 12.2여야 함
nvidia-smi      # Driver 525.60.13 이상
```

JAX 재설치:
```bash
uv pip uninstall jax jaxlib
uv pip install jax[cuda12]
```