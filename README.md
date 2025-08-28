# A100 Quadruped Training

## Quick Start

```bash
# 1. Setup (once)
./setup.sh
source venv/bin/activate

# 2. Train
python train_a100.py

# 3. Resume training
python train_a100.py --resume
```

## Files
- `train_a100.py` - All-in-one training script
- `quadruped.xml` - Robot model
- `setup.sh` - One-click setup
- `requirements.txt` - Python packages

That's it. Simple.