#!/bin/bash
# Convenient training script with environment setup
# Usage: ./run_training.sh [phase1|phase2|test|custom]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}🤖 Quadruped Standing Training Runner${NC}"
echo -e "${GREEN}========================================${NC}"

# Source environment variables
if [ -f "env.sh" ]; then
    source env.sh
else
    echo -e "${RED}Error: env.sh not found!${NC}"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}Error: Virtual environment not activated!${NC}"
    echo "Run: source .venv/bin/activate"
    exit 1
fi

# Function to check GPU availability
check_gpu() {
    python -c "import jax; gpus=jax.devices('gpu'); print(f'Found {len(gpus)} GPU(s)'); exit(0 if gpus else 1)" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}⚠️  No GPU detected. Training will be slower.${NC}"
        echo -e "${YELLOW}Continue anyway? (y/n)${NC}"
        read -r response
        if [ "$response" != "y" ]; then
            exit 1
        fi
    fi
}

# Function to monitor training
monitor_training() {
    echo -e "${GREEN}Training started. Monitor the progress...${NC}"
    echo "Success criteria:"
    echo "  Phase 1: Reach 35-45 reward points"
    echo "  Phase 2: Reach 65-75+ reward points"
    echo ""
}

# Check command line argument
case "${1:-phase1}" in
    phase1)
        echo -e "${GREEN}Starting Phase 1: Learning to stand from sitting${NC}"
        check_gpu
        monitor_training
        python train_standing.py phase1
        echo -e "${GREEN}✓ Phase 1 complete!${NC}"
        echo "Next step: ./run_training.sh phase2"
        ;;
        
    phase2)
        echo -e "${GREEN}Starting Phase 2: Stabilization training${NC}"
        check_gpu
        
        # Check if phase1 checkpoint exists
        if [ ! -f "ars_standing_phase1.npz" ]; then
            echo -e "${YELLOW}Warning: Phase 1 checkpoint not found.${NC}"
            echo "Starting fresh training instead."
        fi
        
        monitor_training
        python train_standing.py phase2
        echo -e "${GREEN}✓ Phase 2 complete!${NC}"
        echo "Next step: ./run_training.sh test"
        ;;
        
    test)
        echo -e "${GREEN}Testing trained policy${NC}"
        check_gpu
        
        # Find the best available checkpoint
        if [ -f "ars_standing_phase2.npz" ]; then
            echo "Using Phase 2 policy"
        elif [ -f "ars_standing_phase1.npz" ]; then
            echo "Using Phase 1 policy"
        elif [ -f "ars_policy1.npz" ]; then
            echo "Using default policy"
        else
            echo -e "${RED}No trained policy found!${NC}"
            exit 1
        fi
        
        python train_standing.py test
        ;;
        
    visualize)
        echo -e "${GREEN}Visualizing trained policy${NC}"
        
        # Check for display
        if [ -z "$DISPLAY" ] && [ "$MUJOCO_GL" != "egl" ] && [ "$MUJOCO_GL" != "osmesa" ]; then
            echo -e "${YELLOW}No display detected. Setting MUJOCO_GL=egl${NC}"
            export MUJOCO_GL=egl
        fi
        
        python visualize_standing.py --duration 30 --slow 2.0
        ;;
        
    custom)
        echo -e "${GREEN}Custom training with your parameters${NC}"
        echo "Edit this script to add your custom parameters"
        check_gpu
        
        # Example custom training
        python mjx_ars_train.py \
            --xml quadruped.xml \
            --save-path custom_policy.npz \
            --iterations 500 \
            --num-envs 256 \
            --crouch-init-ratio 0.75 \
            --knee-band-low 0.50 \
            --knee-band-high 0.70 \
            --target-speed 0.0
        ;;
        
    benchmark)
        echo -e "${GREEN}Running performance benchmark${NC}"
        check_gpu
        
        echo "Testing with different batch sizes..."
        for batch_size in 64 128 256 512; do
            echo -e "${YELLOW}Batch size: $batch_size${NC}"
            python mjx_ars_train.py \
                --xml quadruped.xml \
                --save-path benchmark_test.npz \
                --iterations 5 \
                --num-envs $batch_size \
                --episode-length 100
        done
        rm -f benchmark_test.npz
        ;;
        
    clean)
        echo -e "${YELLOW}Cleaning training artifacts${NC}"
        echo "This will remove checkpoint files. Continue? (y/n)"
        read -r response
        if [ "$response" = "y" ]; then
            rm -f ars_*.npz
            rm -rf $JAX_COMPILATION_CACHE_DIR/*
            echo -e "${GREEN}✓ Cleaned${NC}"
        fi
        ;;
        
    *)
        echo "Usage: $0 [phase1|phase2|test|visualize|custom|benchmark|clean]"
        echo ""
        echo "Commands:"
        echo "  phase1    - Train standing from sitting (300 iterations)"
        echo "  phase2    - Stabilization training (400 iterations)"  
        echo "  test      - Test the trained policy"
        echo "  visualize - Watch robot in MuJoCo viewer"
        echo "  custom    - Run with custom parameters"
        echo "  benchmark - Test performance with different batch sizes"
        echo "  clean     - Remove checkpoint files"
        exit 1
        ;;
esac