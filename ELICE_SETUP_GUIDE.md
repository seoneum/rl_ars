# 🚀 엘리스 클라우드 서버 초기 설정 가이드

## 📋 시작하기 전에
- **필요한 GPU**: NVIDIA GPU (A100, V100, RTX 3090 등)
- **OS**: Ubuntu 20.04 이상 권장
- **필요 시간**: 약 20-30분

---

## 🔧 Step 1: 시스템 기본 패키지 업데이트

VS Code 터미널을 열고 다음 명령어를 실행하세요:

```bash
# 시스템 패키지 업데이트
sudo apt update && sudo apt upgrade -y

# 필수 도구 설치
sudo apt install -y \
    curl \
    wget \
    git \
    build-essential \
    software-properties-common \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglfw3-dev \
    libosmesa6-dev \
    patchelf
```

---

## 🐍 Step 2: Python 3.11 설치

```bash
# Python 3.11 저장소 추가
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

# Python 3.11 설치
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3.11-distutils

# 기본 Python 버전 확인
python3.11 --version
```

---

## 🎮 Step 3: CUDA 12 설치 (이미 설치되어 있다면 건너뛰기)

### CUDA 설치 확인
```bash
# CUDA 버전 확인
nvcc --version

# GPU 확인
nvidia-smi
```

### CUDA가 없다면 설치
```bash
# CUDA 12.0 설치
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0

# 환경 변수 설정
echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
nvcc --version
nvidia-smi
```

---

## 📦 Step 4: uv 설치 (빠른 Python 패키지 관리자)

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# PATH에 추가
echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 설치 확인
uv --version
```

---

## 📂 Step 5: 프로젝트 클론 및 설정

```bash
# 홈 디렉토리로 이동
cd ~

# 프로젝트 클론
git clone https://github.com/seoneum/rl_ars.git
cd rl_ars

# standing-improvement 브랜치로 체크아웃
git checkout standing-improvement

# 디렉토리 구조 확인
ls -la
```

---

## 🌍 Step 6: Python 가상환경 생성 및 활성화

```bash
# 프로젝트 디렉토리에서 실행
cd ~/rl_ars

# uv로 Python 3.11 가상환경 생성
uv venv --python 3.11

# 가상환경 활성화
source .venv/bin/activate

# Python 버전 확인 (3.11이어야 함)
python --version
```

---

## 📚 Step 7: 의존성 패키지 설치

```bash
# JAX with CUDA 12 support 설치
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 기본 패키지 설치
uv pip install \
    mujoco==3.1.1 \
    numpy==1.24.3 \
    tqdm==4.66.1 \
    matplotlib==3.7.2

# 또는 requirements.txt 사용
uv pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## ✅ Step 8: 설치 확인

```bash
# 환경 설정 로드
source env.sh

# 설치 확인 스크립트 실행
python verify_installation.py
```

예상 출력:
```
✓ Python version: 3.11.x
✓ JAX version: 0.4.23
  GPU devices found: 1
  ✓ GPU computation test passed
✓ MuJoCo version: 3.1.1
✅ All systems operational! GPU acceleration available.
```

---

## 🎮 Step 9: 학습 시작

### 방법 1: 간편 실행 스크립트
```bash
# 환경 설정 및 Phase 1 학습 시작
source env.sh
./run_training.sh phase1
```

### 방법 2: 직접 실행
```bash
# Phase 1: 앉은 자세에서 일어서기 학습
python train_standing.py phase1

# Phase 2: 안정화 학습
python train_standing.py phase2

# 테스트
python train_standing.py test
```

### 방법 3: A100 최적화 (A100 서버인 경우)
```bash
# A100 최적화 스크립트 사용
python train_a100_optimized.py phase1
```

---

## 📊 Step 10: 실시간 모니터링

새 터미널을 열어서:

```bash
# GPU 사용률 모니터링
watch -n 1 nvidia-smi

# 또는 더 자세한 모니터링
nvidia-smi dmon -s pucvmet
```

---

## 🔧 VS Code에서 작업하기 편하게 설정

### 1. VS Code 확장 설치 (선택사항)
- Python
- Jupyter
- GitLens

### 2. VS Code 터미널에서 자동으로 가상환경 활성화
```bash
# .vscode/settings.json 생성
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "terminal.integrated.env.linux": {
        "PATH": "${workspaceFolder}/.venv/bin:${env:PATH}"
    }
}
EOF
```

### 3. 작업 공간 저장
- File → Save Workspace As... → `rl_ars.code-workspace`

---

## 🚨 문제 해결

### 1. CUDA 관련 오류
```bash
# CUDA 경로 확인
ls /usr/local/cuda*

# 올바른 CUDA 버전으로 경로 수정
export CUDA_HOME=/usr/local/cuda-12.0  # 실제 설치된 버전으로 변경
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 2. JAX GPU 인식 안됨
```bash
# JAX 재설치
uv pip uninstall jax jaxlib -y
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 테스트
python -c "import jax; print(jax.devices())"
```

### 3. MuJoCo 렌더링 오류
```bash
# Headless 환경용 설정
export MUJOCO_GL=egl  # 또는 osmesa

# 필요한 라이브러리 설치
sudo apt install -y libegl1-mesa libegl1-mesa-dev
```

### 4. 메모리 부족 (OOM)
```bash
# 배치 크기 줄이기
python mjx_ars_train.py \
    --xml quadruped.xml \
    --num-envs 64 \     # 줄임
    --num-dirs 8 \      # 줄임
    --save-path test.npz
```

---

## 📝 빠른 체크리스트

```bash
# 모든 설정을 한번에 확인
echo "=== System Check ==="
echo "Python: $(python3.11 --version)"
echo "CUDA: $(nvcc --version | grep release)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "uv: $(uv --version)"
echo "Git branch: $(cd ~/rl_ars && git branch --show-current)"
echo "Virtual env: $VIRTUAL_ENV"

# JAX GPU 확인
python -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'GPU devices: {jax.devices(\"gpu\")}')"
```

---

## 🎯 전체 실행 스크립트 (복사해서 한번에 실행)

```bash
#!/bin/bash
# 전체 설치 스크립트 - setup_elice.sh로 저장 후 실행

set -e  # 오류 발생시 중단

echo "🚀 엘리스 클라우드 서버 설정 시작..."

# 1. 시스템 업데이트
sudo apt update && sudo apt upgrade -y
sudo apt install -y curl wget git build-essential python3-pip

# 2. Python 3.11 설치
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 3. uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 4. 프로젝트 클론
cd ~
git clone https://github.com/seoneum/rl_ars.git
cd rl_ars
git checkout standing-improvement

# 5. 가상환경 생성
uv venv --python 3.11
source .venv/bin/activate

# 6. 패키지 설치
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install mujoco numpy tqdm matplotlib

# 7. 환경 설정
source env.sh

# 8. 설치 확인
python verify_installation.py

echo "✅ 설치 완료! 이제 학습을 시작할 수 있습니다."
echo "실행: ./run_training.sh phase1"
```

---

## 🏁 설치 완료 후

```bash
# 학습 시작
cd ~/rl_ars
source .venv/bin/activate
source env.sh
./run_training.sh phase1

# 성공 메시지가 나타나면 완료!
```

---

**💡 팁**: 
- 엘리스 클라우드 세션이 끊겨도 학습이 계속되게 하려면 `tmux` 사용
- `tmux new -s training` → 학습 실행 → `Ctrl+B, D`로 분리
- 다시 연결: `tmux attach -t training`

문제가 있으면 GitHub Issues에 문의하세요!