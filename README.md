# 🤖 Quadruped Robot RL Training - Standing Improvement

역관절 4족보행 로봇이 앉은 자세에서 시작하여 안정적으로 서는 것을 학습하는 강화학습 프로젝트입니다.

## 🎯 Project Goals

- **초기 자세**: 무릎 80% 굴곡 (앉은 자세)에서 시작
- **목표 자세**: 무릎 50~70% 신전하여 안정적으로 서기
- **밸런스 유지**: 제자리에서 넘어지지 않고 균형 잡기
- **에너지 효율**: 최소한의 움직임으로 자세 유지

## 📦 Installation

### Prerequisites
- **Python 3.11**
- **CUDA 12.x** (for GPU acceleration)
- **uv** (fast Python package manager)

### Quick Install with uv

```bash
# Clone repository
git clone https://github.com/seoneum/rl_ars.git
cd rl_ars
git checkout standing-improvement

# Create virtual environment with Python 3.11
uv venv --python 3.11
source .venv/bin/activate  # Linux/macOS

# Install dependencies
uv pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
uv pip install -r requirements.txt

# Verify installation
python verify_installation.py
```

📚 **Detailed installation guide**: See [INSTALLATION.md](INSTALLATION.md)

## 🚀 Quick Start

### 1. Training - Phase 1 (Learn to Stand)
```bash
# Train robot to stand up from sitting position
python train_standing.py phase1
```

### 2. Training - Phase 2 (Stabilization)
```bash
# Continue training for better stability
python train_standing.py phase2
```

### 3. Test Trained Policy
```bash
# Evaluate the trained model
python train_standing.py test
```

### 4. Visualize (Optional - requires display)
```bash
# Watch the robot in MuJoCo viewer
python visualize_standing.py --duration 30 --slow 2.0
```

## 📂 Project Structure

```
rl_ars/
├── quadruped.xml           # Robot model (inverted knee quadruped)
├── mjx_ars_train.py        # Main training script (JAX/MJX)
├── train_standing.py       # Phased training orchestrator
├── visualize_standing.py   # MuJoCo viewer for policy
├── verify_installation.py  # Check dependencies
├── requirements.txt        # Python dependencies
├── INSTALLATION.md         # Detailed setup guide
└── README_STANDING.md      # Training parameter guide
```

## 🔧 Key Improvements in This Branch

### Initial Pose Changes
- `crouch_init_ratio`: 0.20 → **0.80** (start deeply crouched)
- `init_pitch`: -0.12 → **-0.08** (less forward lean)

### Reward Structure
- `stand_bonus`: 0.20 → **0.50** (increased standing reward)
- `stand_shape_weight`: 1.20 → **2.0** (stronger shape incentive)
- `target_speed`: 0.35 → **0.0** (stationary goal)
- `streak_weight`: 0.01 → **0.05** (reward continuous standing)

### Knee Control
- Target range: **50-70%** extension
- `knee_band_weight`: 0.8 → **1.5** (stronger band reward)
- `knee_center_weight`: 0.40 → **0.60** (pull to center)

### Movement Penalties
- `overspeed_weight`: 1.2 → **2.0** (penalize any movement)
- `base_vel_penalty_weight`: 0.02 → **0.08** (reduce wobbling)
- `angvel_penalty_weight`: 0.01 → **0.05** (minimize rotation)

## 📊 Training Parameters

### Default Hyperparameters
```python
--num-envs 256         # Parallel environments
--num-dirs 32          # ARS directions
--episode-length 200   # Steps per episode
--action-repeat 3      # Frame skip
--step-size 0.008      # Learning rate
--iterations 300       # Training iterations
```

### Customization Example
```bash
python mjx_ars_train.py \
    --xml quadruped.xml \
    --save-path custom_policy.npz \
    --iterations 500 \
    --crouch-init-ratio 0.75 \
    --knee-band-low 0.45 \
    --knee-band-high 0.65
```

## 🔄 Resume Training

### Continue from Checkpoint
You can resume training from any checkpoint using `--resume`:

```bash
# Basic resume (continue for 100 more iterations)
python mjx_ars_train.py --xml quadruped.xml \
    --save-path my_policy.npz \
    --iterations 100 \
    --resume

# Using resume helper script
python resume_training.py ars_standing_phase1.npz --iterations 200

# Check checkpoint info without training
python resume_training.py ars_standing_phase1.npz --show-info

# Resume with modified parameters
python resume_training.py ars_standing_phase1.npz \
    --iterations 100 \
    --step-size 0.005 \
    --noise-std 0.008
```

### Phase Transition
```bash
# After Phase 1 completes, Phase 2 automatically continues from it
python train_standing.py phase2  # Automatically resumes from phase1.npz
```

### Resume Facts
- ✅ Preserves network weights (theta)
- ✅ Continues from exact iteration number
- ✅ Maintains random seed state
- ✅ Can override learning parameters
- ✅ Saves progress incrementally (every 10 iterations by default)

## 🎮 Monitoring Training

During training, observe these metrics:
- `mean+/mean-`: Average rewards (higher is better)
- `best`: Best rollout score
- `up`: Uprightness (close to 1.0)
- `z`: Torso height (target: 0.45-0.55m)
- `knee`: Knee extension ratio (target: 0.50-0.70)

## 🐛 Troubleshooting

### Robot keeps falling
- Decrease `z_threshold` (allow lower posture)
- Reduce `tilt_penalty_weight`
- Increase `episode_length` for more practice

### Robot won't stand up
- Increase `stand_bonus` and `stand_shape_weight`
- Adjust `crouch_init_ratio` (try 0.70-0.85)
- Increase `knee_band_weight`

### Unstable movements
- Increase movement penalties (`angvel_penalty_weight`, `base_vel_penalty_weight`)
- Reduce `step_size` for more conservative updates
- Increase `act_delta_weight` for smoother actions

## 🔬 Technical Details

- **Algorithm**: Augmented Random Search (ARS)
- **Physics**: MuJoCo simulator with JAX acceleration (MJX)
- **Parallelization**: Vectorized environments on GPU
- **Policy**: Linear mapping from observations to actions

## 📈 Expected Results & Success Criteria

### Training Progress by Phase

| Phase | Iterations | Expected Behavior | Success Criteria (Avg Reward) |
|-------|------------|------------------|------------------------------|
| **Phase 1** | 0-100 | Robot attempts to rise, often falls | 15-25 points |
| | 100-200 | Begins to stand briefly | 25-35 points |
| | 200-300 | Can stand for short periods | **35-45 points** ✓ |
| **Phase 2** | 300-400 | Stable standing, reduced wobbling | 45-55 points |
| | 400-500 | Minimal movement, good balance | 55-65 points |
| | 500-700 | Robust to perturbations | **65-75+ points** ✓ |

### Success Metrics Breakdown

**Phase 1 Success (~40 points):**
- Base reward components:
  - Stand bonus: 0.60
  - Stand shape (z-height + upright): ~1.5
  - Knee band reward: ~1.2
  - Streak reward: ~0.2
- Penalties (negative):
  - Movement penalties: ~-0.5
  - Control cost: ~-0.1
- **Total**: ~35-45 points when successfully standing

**Phase 2 Success (~70 points):**
- Enhanced rewards:
  - Stand bonus: 0.80
  - Stand shape: ~2.4 (better posture)
  - Knee band: ~1.8 (optimal range)
  - Streak reward: ~0.8 (continuous standing)
- Reduced penalties:
  - Movement: ~-0.3 (more stable)
  - Control: ~-0.05 (efficient)
- **Total**: ~65-75 points for stable standing

## 🤝 Contributing

Feel free to open issues or submit PRs for:
- Parameter tuning suggestions
- Additional reward components
- Alternative training algorithms
- Documentation improvements

## 📄 License

This project is part of reinforcement learning research for quadruped locomotion.

## 🔗 Links

- [JAX Documentation](https://jax.readthedocs.io/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [ARS Paper](https://arxiv.org/abs/1803.07055)

## 💻 Git Commands Reference

### Initial Setup (First Time)
```bash
# Clone the repository
git clone https://github.com/seoneum/rl_ars.git
cd rl_ars

# Check available branches
git branch -a

# Switch to standing-improvement branch
git checkout standing-improvement

# Set up git credentials (if not set)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Daily Workflow
```bash
# 1. Update your local repository (pull latest changes)
git fetch origin
git pull origin standing-improvement

# 2. Check current status
git status

# 3. Make your changes, then save them
git add .                          # Add all changes
git add <specific_file>            # Add specific file
git commit -m "Your message here"  # Commit with message

# 4. Push changes to GitHub
git push origin standing-improvement
```

### Working with Branches
```bash
# Create new branch from current
git checkout -b new-feature-branch

# Switch between branches
git checkout standing-improvement
git checkout main

# Merge changes from another branch
git checkout main
git merge standing-improvement

# Delete local branch
git branch -d branch-name
```

### Common Git Operations
```bash
# View commit history
git log --oneline -10              # Last 10 commits
git log --graph --all --oneline    # Visual branch history

# Undo changes (before commit)
git checkout -- <file>              # Discard changes in file
git reset HEAD <file>               # Unstage file

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Check differences
git diff                            # Unstaged changes
git diff --staged                   # Staged changes
git diff HEAD~1                     # Changes from last commit

# Stash changes temporarily
git stash                          # Save current changes
git stash pop                      # Restore stashed changes
git stash list                     # List all stashes
```

### Syncing Fork (if you forked the repo)
```bash
# Add upstream remote (original repo)
git remote add upstream https://github.com/seoneum/rl_ars.git

# Fetch upstream changes
git fetch upstream

# Merge upstream changes to your fork
git checkout standing-improvement
git merge upstream/standing-improvement

# Push updates to your fork
git push origin standing-improvement
```

### Troubleshooting Git Issues
```bash
# Fix merge conflicts
git status                         # See conflicted files
# Edit files to resolve conflicts
git add <resolved_files>
git commit -m "Resolved merge conflicts"

# Reset to remote state (CAREFUL: loses local changes)
git fetch origin
git reset --hard origin/standing-improvement

# Clean untracked files
git clean -n                       # Preview what will be deleted
git clean -f                       # Actually delete untracked files
```

---

**Note**: This branch (`standing-improvement`) focuses specifically on the standing behavior. For walking or other behaviors, check other branches.