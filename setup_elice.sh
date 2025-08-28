#!/bin/bash
# ============================================
# 엘리스 클라우드 서버 자동 설정 스크립트
# Quadruped RL Training Environment Setup
# ============================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

# Header
echo -e "${GREEN}"
echo "============================================"
echo "  엘리스 클라우드 서버 자동 설정 스크립트"
echo "  Quadruped RL Training Environment"
echo "============================================"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
    print_error "Please run without sudo"
    exit 1
fi

# Step 1: System Update
print_info "Step 1/10: Updating system packages..."
sudo apt update -qq
sudo apt upgrade -y -qq
print_status "System updated"

# Step 2: Install Essential Packages
print_info "Step 2/10: Installing essential packages..."
sudo apt install -y -qq \
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
    libegl1-mesa \
    libegl1-mesa-dev \
    patchelf \
    tmux \
    htop > /dev/null 2>&1

print_status "Essential packages installed"

# Step 3: Check CUDA
print_info "Step 3/10: Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    print_status "CUDA $CUDA_VERSION found"
    
    # Set CUDA paths
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
else
    print_warning "CUDA not found. Installing CUDA 12.0..."
    
    # Install CUDA 12
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update -qq
    sudo apt-get -y install cuda-toolkit-12-0 -qq
    rm cuda-keyring_1.1-1_all.deb
    
    # Set CUDA paths
    export CUDA_HOME=/usr/local/cuda-12.0
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    # Add to bashrc
    echo 'export PATH=/usr/local/cuda-12.0/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    
    print_status "CUDA 12.0 installed"
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    print_status "GPU detected: $GPU_NAME"
else
    print_warning "No GPU detected or nvidia-smi not available"
fi

# Step 4: Install Python 3.11
print_info "Step 4/10: Installing Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    sudo add-apt-repository ppa:deadsnakes/ppa -y > /dev/null 2>&1
    sudo apt update -qq
    sudo apt install -y -qq python3.11 python3.11-venv python3.11-dev python3.11-distutils
    print_status "Python 3.11 installed"
else
    print_status "Python 3.11 already installed"
fi

# Step 5: Install uv
print_info "Step 5/10: Installing uv package manager..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null 2>&1
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Add to bashrc
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    print_status "uv installed"
else
    print_status "uv already installed"
fi

# Step 6: Clone Project
print_info "Step 6/10: Cloning project repository..."
cd ~
if [ -d "rl_ars" ]; then
    print_warning "Project directory already exists. Updating..."
    cd rl_ars
    git fetch origin
    git checkout standing-improvement
    git pull origin standing-improvement
else
    git clone https://github.com/seoneum/rl_ars.git
    cd rl_ars
    git checkout standing-improvement
fi
print_status "Project cloned/updated"

# Step 7: Create Virtual Environment
print_info "Step 7/10: Creating Python virtual environment..."
cd ~/rl_ars

if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists. Recreating..."
    rm -rf .venv
fi

~/.cargo/bin/uv venv --python 3.11
print_status "Virtual environment created"

# Step 8: Install Dependencies
print_info "Step 8/10: Installing Python packages (this may take a while)..."
source .venv/bin/activate

# Install JAX with CUDA support
~/.cargo/bin/uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html > /dev/null 2>&1

# Install other packages
~/.cargo/bin/uv pip install \
    mujoco==3.1.1 \
    numpy==1.24.3 \
    tqdm==4.66.1 \
    matplotlib==3.7.2 \
    scipy==1.11.4 > /dev/null 2>&1

print_status "Python packages installed"

# Step 9: Verify Installation
print_info "Step 9/10: Verifying installation..."
source env.sh > /dev/null 2>&1

# Quick verification
python -c "
import sys
import jax
import mujoco
print(f'Python: {sys.version.split()[0]}')
print(f'JAX: {jax.__version__}')
print(f'MuJoCo: {mujoco.__version__}')
devices = jax.devices()
print(f'Devices: {devices}')
if any(d.device_kind == 'gpu' for d in devices):
    print('GPU: Available ✓')
else:
    print('GPU: Not available (CPU mode)')
" > /tmp/verify_output.txt 2>&1

if [ $? -eq 0 ]; then
    cat /tmp/verify_output.txt
    print_status "Installation verified"
else
    print_error "Verification failed. Check error:"
    cat /tmp/verify_output.txt
fi

# Step 10: Create convenience scripts
print_info "Step 10/10: Creating convenience scripts..."

# Create start_training.sh
cat > ~/rl_ars/start_training.sh << 'EOF'
#!/bin/bash
cd ~/rl_ars
source .venv/bin/activate
source env.sh
echo "Environment activated. You can now run:"
echo "  ./run_training.sh phase1  # Start training"
echo "  python train_standing.py test  # Test policy"
exec bash
EOF
chmod +x ~/rl_ars/start_training.sh

# Create tmux_training.sh
cat > ~/rl_ars/tmux_training.sh << 'EOF'
#!/bin/bash
# Start training in tmux session (survives disconnection)
tmux new-session -d -s training "cd ~/rl_ars && source .venv/bin/activate && source env.sh && ./run_training.sh phase1"
echo "Training started in tmux session 'training'"
echo "Commands:"
echo "  tmux attach -t training  # View training"
echo "  tmux ls                  # List sessions"
echo "  Ctrl+B, D               # Detach from session"
EOF
chmod +x ~/rl_ars/tmux_training.sh

print_status "Convenience scripts created"

# Final Summary
echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}       🎉 설치 완료! Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Project location: ~/rl_ars"
echo ""
echo -e "${YELLOW}다음 단계 (Next Steps):${NC}"
echo ""
echo "1. 터미널에서 프로젝트로 이동:"
echo "   ${BLUE}cd ~/rl_ars${NC}"
echo ""
echo "2. 환경 활성화:"
echo "   ${BLUE}source .venv/bin/activate${NC}"
echo "   ${BLUE}source env.sh${NC}"
echo ""
echo "3. 학습 시작:"
echo "   ${BLUE}./run_training.sh phase1${NC}"
echo ""
echo "또는 간편 스크립트 사용:"
echo "   ${BLUE}./start_training.sh${NC}  # 환경 자동 활성화"
echo "   ${BLUE}./tmux_training.sh${NC}   # 백그라운드 실행"
echo ""

# GPU specific recommendations
if [[ $GPU_NAME == *"A100"* ]]; then
    echo -e "${YELLOW}A100 GPU 감지됨! 최적화된 설정 사용:${NC}"
    echo "   ${BLUE}python train_a100_optimized.py phase1${NC}"
elif [[ $GPU_NAME == *"V100"* ]]; then
    echo -e "${YELLOW}V100 GPU 감지됨! 큰 배치 크기 사용 가능:${NC}"
    echo "   ${BLUE}--num-envs 512 --num-dirs 32${NC}"
fi

echo ""
echo -e "${GREEN}Good luck with your training! 🚀${NC}"
echo ""

# Save installation log
echo "Installation completed at $(date)" > ~/rl_ars/installation.log
echo "GPU: $GPU_NAME" >> ~/rl_ars/installation.log
echo "CUDA: $CUDA_VERSION" >> ~/rl_ars/installation.log