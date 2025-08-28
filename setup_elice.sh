#!/bin/bash
# ============================================
# 엘리스 클라우드 서버 자동 설정 스크립트
# Quadruped RL Training Environment (A100 80GB + JAX CUDA12)
# - uv는 pipx로 사전 설치, venv는 사전 생성/활성화된 상태를 가정
# - venv가 활성화되어 있지 않으면 .venv 자동 활성화 시도
# - 파이썬 3.11 필수
# ============================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
print_status(){ echo -e "${GREEN}[✓]${NC} $1"; }
print_error(){ echo -e "${RED}[✗]${NC} $1"; }
print_warning(){ echo -e "${YELLOW}[!]${NC} $1"; }
print_info(){ echo -e "${BLUE}[i]${NC} $1"; }

echo -e "${GREEN}"
echo "============================================"
echo "  엘리스 클라우드 서버 자동 설정 스크립트"
echo "  Quadruped RL Training (A100 80GB, JAX CUDA12)"
echo "============================================"
echo -e "${NC}"

# 0) Root 사용 금지
if [ "$EUID" -eq 0 ]; then
  print_error "Please run this script without sudo."
  exit 1
fi

# 1) 시스템 패키지 업데이트
print_info "Step 1/10: Updating system packages..."
sudo apt-get update -qq
sudo apt-get upgrade -y -qq
print_status "System packages updated."

# 2) 필수 시스템 패키지 설치
print_info "Step 2/10: Installing essential packages..."
sudo apt-get install -y -qq \
  curl wget git build-essential software-properties-common \
  python3-pip python3-dev \
  libgl1-mesa-glx libglfw3 libglfw3-dev libosmesa6-dev libegl1-mesa libegl1-mesa-dev \
  patchelf tmux htop \
  hwloc libhwloc-dev > /dev/null 2>&1
print_status "Essential packages installed."

# 3) NVIDIA GPU 및 드라이버 확인
print_info "Step 3/10: Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
  GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -n1)
  print_status "GPU found: $GPU_INFO"
else
  print_warning "nvidia-smi not found. Please check NVIDIA driver or container settings."
fi

# 4) 프로젝트 저장소 클론 또는 업데이트
print_info "Step 4/10: Cloning/updating project repository..."
cd ~
if [ -d "rl_ars" ]; then
  print_warning "Project directory 'rl_ars' already exists. Updating..."
  cd rl_ars
  git fetch origin
  git checkout standing-improvement
  git pull origin standing-improvement
else
  git clone https://github.com/seoneum/rl_ars.git
  cd rl_ars
  git checkout standing-improvement
fi
print_status "Project repository is up to date."

# 5) uv 설치 확인 (pipx로 설치된 것을 가정)
print_info "Step 5/10: Checking for uv (expected via pipx)..."
if ! command -v uv &> /dev/null; then
  # pipx의 기본 PATH가 설정되지 않은 경우를 대비해 경로 추가
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi
if ! command -v uv &> /dev/null; then
  print_error "uv command not found. Please install it first:
  - pipx install uv
  - pipx ensurepath"
  exit 1
fi
print_status "uv found at: $(command -v uv)"

# 6) Python 3.11 가상 환경 확인 및 활성화
print_info "Step 6/10: Checking for active Python 3.11 virtual environment..."
if [ -z "$VIRTUAL_ENV" ]; then
  if [ -f ".venv/bin/activate" ]; then
    print_warning "Virtual environment not active. Activating ./.venv ..."
    source .venv/bin/activate
  else
    print_error "No active venv, and ./.venv not found. Please create and activate it first:
  1. python3.11 -m venv .venv  (or 'uv venv --python 3.11')
  2. source .venv/bin/activate
Then re-run this script."
    exit 1
  fi
fi

PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
if [ "$PY_VER" != "3.11" ]; then
  print_error "Python 3.11 is required inside the venv. Current version is $PY_VER.
Please recreate the virtual environment with Python 3.11:
  - sudo apt install -y python3.11 python3.11-venv python3.11-dev
  - rm -rf .venv && python3.11 -m venv .venv
  - source .venv/bin/activate"
  exit 1
fi
print_status "Virtual environment is active with Python $PY_VER."

# 7) Python 패키지 설치
print_info "Step 7/10: Installing Python packages into the active venv..."
python -m pip install -U pip setuptools wheel > /dev/null 2>&1

# JAX for CUDA 12
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html > /dev/null 2>&1

# Other required packages
uv pip install \
  mujoco==3.1.1 \
  numpy==1.24.3 \
  tqdm==4.66.1 \
  matplotlib==3.7.2 \
  scipy==1.11.4 > /dev/null 2>&1

print_status "Python packages installed."

# 8) env.sh 환경 설정 파일 생성 (없는 경우에만)
print_info "Step 8/10: Creating env.sh (if missing)..."
if [ ! -f "env.sh" ]; then
  cat > env.sh << 'EOF'
# Recommended environment for JAX + A100 80GB (no XLA_FLAGS)
export JAX_PLATFORMS=gpu
export JAX_ENABLE_X64=false
export JAX_DEBUG_NANS=false
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$HOME/.cache/jax_a100}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || true
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export CUDA_LAUNCH_BLOCKING=0
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-1}"
# Intentionally no XLA_FLAGS to avoid unknown flag crashes with modern jaxlib
EOF
  print_status "Created default env.sh."
else
  print_status "env.sh already exists, not modified."
fi

# 9) 설치 검증
print_info "Step 9/10: Verifying installation..."
[ -f "env.sh" ] && source env.sh

python - << 'PY'
import sys, jax
try:
    import mujoco
except Exception as e:
    mujoco = None
print(f'  Python: {sys.version.split()[0]}')
try:
    import jaxlib
    print(f'  JAX: {jax.__version__} | jaxlib: {jaxlib.__version__}')
except Exception:
    print(f'  JAX: {jax.__version__}')
try:
    devices = jax.devices()
    print(f'  JAX Devices: {[d.platform for d in devices]}')
    print(f'  GPU Available: {"✓" if any(d.platform == "gpu" for d in devices) else "✗"}')
except Exception as e:
    print(f'  Error checking JAX devices: {e}')
if mujoco:
    print(f'  MuJoCo: {mujoco.__version__}')
PY

print_status "Installation verified."

# 10) 편의 스크립트 생성
print_info "Step 10/10: Creating convenience scripts..."

cat > ~/rl_ars/start_training.sh << 'EOF'
#!/bin/bash
cd ~/rl_ars
source .venv/bin/activate
[ -f env.sh ] && source env.sh
echo "Environment activated. You can now run:"
echo "  python train_a100_optimized.py phase1"
echo "  ./run_training.sh phase1"
exec bash
EOF
chmod +x ~/rl_ars/start_training.sh

cat > ~/rl_ars/tmux_training.sh << 'EOF'
#!/bin/bash
# Start training in a tmux session (survives disconnection)
SESSION=${1:-training}
tmux new-session -d -s "$SESSION" "cd ~/rl_ars && source .venv/bin/activate && [ -f env.sh ] && source env.sh && python train_a100_optimized.py phase1"
echo "Training started in tmux session '$SESSION'"
echo "Useful tmux commands:"
echo "  tmux attach -t $SESSION  # Attach to the session"
echo "  tmux ls                  # List all sessions"
echo "  Ctrl+B, D                # Detach from the session"
EOF
chmod +x ~/rl_ars/tmux_training.sh

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}      🎉 설치 완료! Setup Complete! 🎉      ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Project location: ~/rl_ars"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. cd ~/rl_ars"
echo "  2. source .venv/bin/activate  # (If not already active)"
echo "  3. source env.sh"
echo "  4. python train_a100_optimized.py phase1"
echo ""

# 설치 로그 저장
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
echo "Installation completed at $(date)" > ~/rl_ars/installation.log
echo "GPU: ${GPU_NAME:-unknown}" >> ~/rl_ars/installation.log
