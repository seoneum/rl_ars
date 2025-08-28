# Quadruped Robot Standing Training

4족 로봇이 앉은 자세에서 일어서는 동작을 학습하는 강화학습 프로젝트

## 🚀 Quick Start (Elice Cloud A100)

```bash
# 1. 한 번에 실행
./run_on_elice.sh
```

또는 수동 설정:

```bash
# 1. 가상환경 생성
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate

# 2. 패키지 설치
uv pip install jax[cuda12] mujoco mujoco-mjx tqdm

# 3. 훈련 실행
python train_a100.py  # 기본 훈련 (900점까지)
python train_advanced.py  # 고급 훈련 (900점 이상)
```

## 📁 File Structure

### Training Scripts
- `train_a100.py` - A100 최적화 기본 훈련 (900점까지)
- `train_advanced.py` - Curriculum Learning 적용 고급 훈련 (900점 돌파)

### Visualization & Analysis
- `visualize_robot.py` - MuJoCo 시각화 및 정책 테스트
- `analyze_training.py` - 훈련 결과 분석 및 그래프 생성

### Setup Scripts
- `run_on_elice.sh` - Elice Cloud 원클릭 설정
- `setup_a100.sh` - A100 환경 설정
- `ELICE_SETUP.md` - 상세 설정 가이드

### Model
- `quadruped.xml` - 역관절 4족 로봇 MuJoCo 모델

## 🎮 Visualization

훈련된 모델 확인:

```bash
# 인터랙티브 뷰어
python visualize_robot.py --mode interactive

# 자동 평가
python visualize_robot.py --mode evaluate --episodes 10

# 특정 체크포인트 로드
python visualize_robot.py --checkpoint checkpoints/advanced_best_950.ckpt
```

### 조작법
- **Space**: 시뮬레이션 일시정지/재개
- **R**: 앉은 자세로 리셋
- **S**: 선 자세로 변경  
- **P**: 정책 실행 토글
- **Q/ESC**: 종료
- **마우스**: 카메라 회전/줌

## 📊 Training Analysis

```bash
# 훈련 진행 상황 분석
python analyze_training.py

# 그래프 생성
python analyze_training.py --plot

# 종합 보고서 생성
python analyze_training.py --report
```

## 🎯 Training Stages

### Stage 0: 앉기 → 서기 (기본)
- 목표: 80% 굽힘 → 50-70% 펴짐
- 보상: 높이 + 무릎 각도

### Stage 1: 균형 유지
- 목표: 선 자세 유지
- 보상: 균형 + 자세 안정성

### Stage 2: 걷기 준비
- 목표: 동적 균형
- 보상: 발 교차 + 전진 준비

### Stage 3: 실제 걷기
- 목표: 전진 보행
- 보상: 전진 속도 + 안정성

## 🔧 Training Parameters

### Basic Training (train_a100.py)
```python
--batch_size 1024      # A100 최적화
--num_iters 1000       # 훈련 반복 횟수
--step_size 0.03       # 학습률
--horizon 1000         # 에피소드 길이
```

### Advanced Training (train_advanced.py)
```python
--curriculum_stages 4  # 커리큘럼 단계 수
--batch_size 1024      # 배치 크기
--num_iters 2000       # 더 긴 훈련
--step_size 0.03       # 초기 학습률 (자동 조절)
```

## 📈 Performance Tips

### 900점 돌파가 안 될 때

1. **Advanced Training 사용**
   ```bash
   python train_advanced.py --num_iters 3000
   ```

2. **학습률 조정**
   ```bash
   python train_advanced.py --step_size 0.02
   ```

3. **더 많은 방향 탐색**
   ```bash
   python train_advanced.py --num_directions 64 --top_directions 32
   ```

### GPU 최적화

- A100: `--batch_size 1024` 또는 2048
- RTX 3090: `--batch_size 512`
- RTX 3060: `--batch_size 128`

## 🐛 Troubleshooting

### JAX CUDA 인식 안 됨
```bash
# 가상환경 재생성
rm -rf .venv
./run_on_elice.sh
```

### 메모리 부족
```bash
# 배치 크기 줄이기
python train_a100.py --batch_size 512
```

### 훈련 정체
```bash
# Curriculum learning 사용
python train_advanced.py
```

## 📝 Notes

- Elice Cloud에서는 **반드시 가상환경** 사용
- JAX는 `jax[cuda12]`로 설치 (A100용)
- 체크포인트는 `checkpoints/` 디렉토리에 자동 저장
- 900점 이상 달성 시 `_best_*.ckpt` 파일로 별도 저장

## 🎉 Success Criteria

- **Level 1**: 500점 - 기본 서기 동작
- **Level 2**: 700점 - 안정적 서기
- **Level 3**: 900점 - 균형 잡힌 자세
- **Level 4**: 1000점+ - 동적 균형 및 걷기 준비
- **Level 5**: 1200점+ - 실제 보행 가능