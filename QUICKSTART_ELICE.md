# ⚡ 엘리스 클라우드 빠른 시작 (5분 설정)

## 🎯 한 줄 설치 (복사해서 터미널에 붙여넣기)

```bash
curl -sSL https://raw.githubusercontent.com/seoneum/rl_ars/standing-improvement/setup_elice.sh | bash
```

또는 수동으로:

```bash
# 1. 설치 스크립트 다운로드
wget https://raw.githubusercontent.com/seoneum/rl_ars/standing-improvement/setup_elice.sh

# 2. 실행 권한 부여
chmod +x setup_elice.sh

# 3. 실행
./setup_elice.sh
```

---

## 🚀 설치 후 바로 학습 시작

```bash
# 프로젝트 디렉토리로 이동
cd ~/rl_ars

# 간편 실행 (환경 자동 설정)
./start_training.sh
```

---

## 📊 GPU별 최적 명령어

### A100 서버
```bash
cd ~/rl_ars
source .venv/bin/activate
python train_a100_optimized.py phase1
```

### V100 서버
```bash
cd ~/rl_ars
source .venv/bin/activate
python mjx_ars_train.py \
    --xml quadruped.xml \
    --num-envs 512 \
    --num-dirs 32 \
    --save-path v100_policy.npz
```

### T4 / 일반 GPU
```bash
cd ~/rl_ars
source .venv/bin/activate
./run_training.sh phase1
```

---

## 🔍 설치 확인

```bash
cd ~/rl_ars
source .venv/bin/activate
python -c "
import jax
print('JAX devices:', jax.devices())
print('GPU available:', any(d.device_kind=='gpu' for d in jax.devices()))
"
```

---

## 💡 VS Code 터미널 단축키 설정

VS Code에서 작업하기 편하게:

1. **터미널 열기**: `` Ctrl + ` ``
2. **새 터미널**: `Ctrl + Shift + ` ``
3. **터미널 분할**: `Ctrl + Shift + 5`

### 자동 환경 활성화 설정

VS Code 설정 (settings.json):
```json
{
    "terminal.integrated.shellArgs.linux": [
        "-c",
        "cd ~/rl_ars && source .venv/bin/activate && exec bash"
    ]
}
```

---

## 🖥️ tmux로 백그라운드 실행 (세션 유지)

```bash
# 학습을 tmux 세션에서 시작 (연결 끊어도 계속 실행)
cd ~/rl_ars
./tmux_training.sh

# 학습 상태 확인
tmux attach -t training

# tmux 세션에서 나가기 (학습은 계속)
Ctrl+B, 그 다음 D
```

---

## 📈 실시간 모니터링

터미널을 3개로 분할하여:

**터미널 1 - 학습**
```bash
cd ~/rl_ars
./run_training.sh phase1
```

**터미널 2 - GPU 모니터링**
```bash
watch -n 1 nvidia-smi
```

**터미널 3 - 로그 확인**
```bash
cd ~/rl_ars
tail -f *.log  # 로그 파일이 있는 경우
```

---

## ⚠️ 자주 발생하는 문제와 해결

### 1. "No GPU detected"
```bash
# CUDA 재설정
export CUDA_HOME=/usr/local/cuda-12.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# JAX 재설치
source .venv/bin/activate
uv pip uninstall jax jaxlib -y
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 2. "Out of Memory (OOM)"
```bash
# 배치 크기 줄이기
python mjx_ars_train.py \
    --xml quadruped.xml \
    --num-envs 64 \     # 128 → 64
    --num-dirs 8 \      # 16 → 8
    --save-path small_batch.npz
```

### 3. "Module not found"
```bash
# 가상환경 재활성화
cd ~/rl_ars
deactivate  # 기존 환경 종료
source .venv/bin/activate
source env.sh
```

---

## 📝 체크리스트

- [ ] GPU 확인: `nvidia-smi`
- [ ] CUDA 확인: `nvcc --version`
- [ ] Python 3.11: `python --version`
- [ ] JAX GPU: `python -c "import jax; print(jax.devices())"`
- [ ] 프로젝트 위치: `~/rl_ars`
- [ ] 가상환경 활성화: `.venv/bin/activate`

---

## 🎉 완료!

모든 설정이 끝났습니다. 이제 학습을 시작하세요:

```bash
cd ~/rl_ars
./start_training.sh
```

**Success Criteria:**
- Phase 1: 35-45 reward points (300 iterations)
- Phase 2: 65-75+ reward points (400 iterations)

---

## 📞 도움말

문제가 발생하면:
1. 설치 로그 확인: `cat ~/rl_ars/installation.log`
2. GPU 상태 확인: `nvidia-smi`
3. GitHub Issues: https://github.com/seoneum/rl_ars/issues