#!/bin/bash
# Elice Cloud에서 바로 실행하는 스크립트

# 색상 코드
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}==================================="
echo " Elice Cloud A100 Quick Setup"
echo "===================================${NC}"

# 1. 가상환경 확인 및 생성
if [ ! -d ".venv" ]; then
    echo -e "\n${GREEN}[1/4] Creating virtual environment...${NC}"
    pip install uv
    uv venv .venv --python 3.11
else
    echo -e "\n${GREEN}[1/4] Virtual environment exists${NC}"
fi

# 2. 가상환경 활성화
echo -e "\n${GREEN}[2/4] Activating virtual environment...${NC}"
source .venv/bin/activate

# 3. 패키지 설치
echo -e "\n${GREEN}[3/4] Installing packages...${NC}"
uv pip install --upgrade pip
uv pip install jax[cuda12] mujoco mujoco-mjx tqdm

# 4. GPU 확인
echo -e "\n${GREEN}[4/4] Checking GPU...${NC}"
python -c "
import jax
devices = jax.devices()
print(f'Devices: {devices}')
if 'cuda' in str(devices[0]).lower() or 'gpu' in str(devices[0]).lower():
    print('✅ GPU READY!')
else:
    print('❌ GPU NOT FOUND!')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}✅ Setup complete! Running training...${NC}"
    python train_a100.py --batch_size 1024 --checkpoint_path checkpoints/a100.ckpt
else
    echo -e "\n${RED}❌ GPU setup failed!${NC}"
    echo "Try manually:"
    echo "1. source .venv/bin/activate"
    echo "2. uv pip install jax[cuda12]"
    echo "3. python train_a100.py"
fi