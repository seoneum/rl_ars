#!/bin/bash
# Convenient training script with environment setup and command shortcuts.
# Usage: ./run_training.sh [phase1|phase2|test|visualize|custom|benchmark|clean]

# Exit on error
set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🤖 Quadruped Standing Training Runner 🤖${NC}"
echo -e "${GREEN}========================================${NC}"

# Source environment variables if the file exists
if [ -f "env.sh" ]; then
  source env.sh
else
  echo -e "${YELLOW}Warning: env.sh not found. Continuing with system defaults.${NC}"
  echo -e "${YELLOW}It's recommended to have env.sh for optimal performance.${NC}"
fi

# Check if the Python virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
  echo -e "${RED}Error: Python virtual environment not activated!${NC}"
  echo "Please activate it first by running: source .venv/bin/activate"
  exit 1
fi

# Function to check for GPU availability
check_gpu() {
  # Hide stderr to avoid clutter from JAX messages
  python -c "import jax; gpus=jax.devices('gpu'); print(f'Found {len(gpus)} GPU(s)'); exit(0 if gpus else 1)" 2>/dev/null
  if [ $? -ne 0 ]; then
    echo -e "${YELLOW}⚠️  No GPU detected by JAX. Training will be very slow.${NC}"
    read -p "Continue anyway? (y/n) " -r response
    if [[ ! "$response" =~ ^[yY]$ ]]; then
        exit 1
    fi
  fi
}

# Function to display training guidance
monitor_training() {
  echo -e "${GREEN}Training started. Monitor the reward progress...${NC}"
  echo "Success criteria:"
  echo "  - Phase 1: Aim for a reward of 35-45"
  echo "  - Phase 2: Aim for a reward of 65-75+"
  echo ""
}

# Main script logic based on the first argument
case "${1:-phase1}" in
  phase1)
    echo -e "${GREEN}Starting Phase 1: Learning to stand from a crouch${NC}"
    check_gpu
    monitor_training
    # Recommend the A100 script if available
    echo -e "${YELLOW}Note: For A100 GPUs, it's better to run: python train_a100_optimized.py phase1${NC}"
    python train_standing.py phase1
    echo -e "${GREEN}✓ Phase 1 complete!${NC}"
    echo "Next step: ./run_training.sh phase2"
    ;;
  phase2)
    echo -e "${GREEN}Starting Phase 2: Stabilization training${NC}"
    check_gpu
    if [ ! -f "ars_standing_phase1.npz" ]; then
      echo -e "${YELLOW}Warning: Phase 1 checkpoint not found. Phase 2 will start from scratch.${NC}"
    fi
    monitor_training
    python train_standing.py phase2
    echo -e "${GREEN}✓ Phase 2 complete!${NC}"
    echo "Next step: ./run_training.sh test"
    ;;
  test)
    echo -e "${GREEN}Testing the latest trained policy${NC}"
    check_gpu
    POLICY_FILE=""
    if [ -f "ars_standing_phase2.npz" ]; then
      POLICY_FILE="ars_standing_phase2.npz"
    elif [ -f "ars_standing_phase1.npz" ]; then
      POLICY_FILE="ars_standing_phase1.npz"
    elif [ -f "ars_policy1.npz" ]; then
      POLICY_FILE="ars_policy1.npz"
    fi
    
    if [ -n "$POLICY_FILE" ]; then
        echo "Using policy: $POLICY_FILE"
        python train_standing.py test --resume-path "$POLICY_FILE"
    else
        echo -e "${RED}Error: No trained policy found! Train a model first.${NC}"
        exit 1
    fi
    ;;
  visualize)
    echo -e "${GREEN}Visualizing trained policy${NC}"
    if [ -z "$DISPLAY" ] && [ "$MUJOCO_GL" != "egl" ] && [ "$MUJOCO_GL" != "osmesa" ]; then
      echo -e "${YELLOW}No display detected. Forcing MUJOCO_GL=egl for headless rendering.${NC}"
      export MUJOCO_GL=egl
    fi
    python visualize_standing.py --duration 30 --slow 2.0
    ;;
  benchmark)
    echo -e "${GREEN}Running performance benchmark${NC}"
    check_gpu
    echo "This will test training speed with different batch sizes."
    python train_a100_optimized.py benchmark
    ;;
  clean)
    echo -e "${YELLOW}Cleaning up training artifacts...${NC}"
    read -p "This will remove all .npz checkpoint files and the JAX cache. Are you sure? (y/n) " -r response
    if [[ "$response" =~ ^[yY]$ ]]; then
      echo "Removing checkpoints..."
      rm -f ars_*.npz custom_policy.npz benchmark_*.npz
      if [ -n "$JAX_COMPILATION_CACHE_DIR" ] && [ -d "$JAX_COMPILATION_CACHE_DIR" ]; then
        echo "Clearing JAX cache at $JAX_COMPILATION_CACHE_DIR..."
        rm -rf "$JAX_COMPILATION_CACHE_DIR"/*
      fi
      echo -e "${GREEN}✓ Clean-up complete.${NC}"
    else
      echo "Clean-up cancelled."
    fi
    ;;
  *)
    echo "Usage: $0 [phase1|phase2|test|visualize|benchmark|clean]"
    exit 1
    ;;
esac
