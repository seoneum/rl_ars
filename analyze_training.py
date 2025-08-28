#!/usr/bin/env python3
"""
훈련 결과 분석 도구
체크포인트를 로드해서 성능 분석 및 시각화
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

def load_all_checkpoints(checkpoint_dir="checkpoints"):
    """모든 체크포인트 로드"""
    checkpoints = []
    
    for ckpt_file in Path(checkpoint_dir).glob("*.ckpt"):
        try:
            data = np.load(ckpt_file, allow_pickle=True)
            info = {
                'file': str(ckpt_file),
                'iteration': int(data['iter']) if 'iter' in data else 0,
                'obs_dim': int(data['obs_dim']),
                'act_dim': int(data['act_dim']),
                'timestamp': datetime.fromtimestamp(os.path.getmtime(ckpt_file))
            }
            
            # 메트릭 정보가 있으면 추가
            if 'metrics' in data:
                metrics = data['metrics'].item()
                info.update(metrics)
            
            checkpoints.append(info)
        except Exception as e:
            print(f"Failed to load {ckpt_file}: {e}")
    
    return sorted(checkpoints, key=lambda x: x['iteration'])

def analyze_progress(checkpoints):
    """훈련 진행 상황 분석"""
    print("="*60)
    print(" Training Progress Analysis")
    print("="*60)
    
    if not checkpoints:
        print("No checkpoints found!")
        return
    
    # 최신 체크포인트
    latest = checkpoints[-1]
    print(f"\nLatest checkpoint:")
    print(f"  File: {latest['file']}")
    print(f"  Iteration: {latest['iteration']}")
    print(f"  Time: {latest['timestamp']}")
    
    if 'mean_reward' in latest:
        print(f"  Mean Reward: {latest['mean_reward']:.2f}")
    if 'best_reward' in latest:
        print(f"  Best Reward: {latest['best_reward']:.2f}")
    if 'curriculum_stage' in latest:
        print(f"  Curriculum Stage: {latest['curriculum_stage']}")
    if 'learning_rate' in latest:
        print(f"  Learning Rate: {latest['learning_rate']:.6f}")
    
    # 진행 상황
    print(f"\nTotal checkpoints: {len(checkpoints)}")
    
    # 보상 추이
    rewards = [c.get('mean_reward', 0) for c in checkpoints if 'mean_reward' in c]
    if rewards:
        print(f"\nReward progression:")
        print(f"  Initial: {rewards[0]:.2f}")
        print(f"  Current: {rewards[-1]:.2f}")
        print(f"  Best: {max(rewards):.2f}")
        print(f"  Improvement: {rewards[-1] - rewards[0]:.2f}")
    
    # Stage progression
    stages = [c.get('curriculum_stage', 0) for c in checkpoints if 'curriculum_stage' in c]
    if stages:
        print(f"\nCurriculum progression:")
        print(f"  Current stage: {stages[-1]}")
        print(f"  Max stage reached: {max(stages)}")

def plot_training_curves(checkpoints, save_path="training_curves.png"):
    """훈련 곡선 시각화"""
    if not checkpoints:
        print("No data to plot!")
        return
    
    # Extract data
    iterations = [c['iteration'] for c in checkpoints]
    mean_rewards = [c.get('mean_reward', 0) for c in checkpoints if 'mean_reward' in c]
    best_rewards = [c.get('best_reward', 0) for c in checkpoints if 'best_reward' in c]
    stages = [c.get('curriculum_stage', 0) for c in checkpoints if 'curriculum_stage' in c]
    learning_rates = [c.get('learning_rate', 0) for c in checkpoints if 'learning_rate' in c]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Analysis', fontsize=16)
    
    # Plot 1: Rewards
    if mean_rewards:
        ax = axes[0, 0]
        iters = iterations[:len(mean_rewards)]
        ax.plot(iters, mean_rewards, 'b-', label='Mean Reward', linewidth=2)
        if best_rewards:
            ax.plot(iters[:len(best_rewards)], best_rewards, 'r--', 
                   label='Best Reward', linewidth=2, alpha=0.7)
        ax.axhline(y=900, color='g', linestyle=':', label='Target (900)', linewidth=1)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Progression')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Curriculum Stages
    if stages:
        ax = axes[0, 1]
        iters = iterations[:len(stages)]
        ax.plot(iters, stages, 'g-', linewidth=2, marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Stage')
        ax.set_title('Curriculum Stage Progression')
        ax.set_ylim(-0.5, max(stages) + 0.5)
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate
    if learning_rates:
        ax = axes[1, 0]
        iters = iterations[:len(learning_rates)]
        ax.plot(iters, learning_rates, 'm-', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Improvement Rate
    if len(mean_rewards) > 10:
        ax = axes[1, 1]
        improvement = []
        window = 10
        for i in range(window, len(mean_rewards)):
            imp = mean_rewards[i] - mean_rewards[i-window]
            improvement.append(imp)
        
        iters = iterations[window:len(mean_rewards)]
        ax.plot(iters, improvement, 'c-', linewidth=2)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(f'Improvement (per {window} iters)')
        ax.set_title('Training Improvement Rate')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()

def find_best_checkpoint(checkpoints):
    """최고 성능 체크포인트 찾기"""
    if not checkpoints:
        return None
    
    best = None
    best_reward = -float('inf')
    
    for ckpt in checkpoints:
        reward = ckpt.get('best_reward', ckpt.get('mean_reward', -float('inf')))
        if reward > best_reward:
            best_reward = reward
            best = ckpt
    
    return best, best_reward

def generate_report(checkpoint_dir="checkpoints", output_file="training_report.txt"):
    """종합 보고서 생성"""
    checkpoints = load_all_checkpoints(checkpoint_dir)
    
    with open(output_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(" Quadruped Training Report\n")
        f.write("="*60 + "\n\n")
        
        if not checkpoints:
            f.write("No checkpoints found!\n")
            return
        
        # Summary
        f.write("Summary:\n")
        f.write(f"  Total checkpoints: {len(checkpoints)}\n")
        f.write(f"  Training iterations: {checkpoints[-1]['iteration']}\n")
        f.write(f"  Training period: {checkpoints[0]['timestamp']} to {checkpoints[-1]['timestamp']}\n\n")
        
        # Best performance
        best, best_reward = find_best_checkpoint(checkpoints)
        if best:
            f.write("Best Performance:\n")
            f.write(f"  Checkpoint: {best['file']}\n")
            f.write(f"  Iteration: {best['iteration']}\n")
            f.write(f"  Reward: {best_reward:.2f}\n")
            if 'curriculum_stage' in best:
                f.write(f"  Stage: {best['curriculum_stage']}\n")
            f.write("\n")
        
        # Recent checkpoints
        f.write("Recent Checkpoints:\n")
        for ckpt in checkpoints[-5:]:
            f.write(f"  Iter {ckpt['iteration']:5d}: ")
            if 'mean_reward' in ckpt:
                f.write(f"Reward={ckpt['mean_reward']:7.2f}")
            if 'curriculum_stage' in ckpt:
                f.write(f", Stage={ckpt['curriculum_stage']}")
            f.write(f" ({ckpt['timestamp'].strftime('%Y-%m-%d %H:%M')})\n")
        
        # Recommendations
        f.write("\nRecommendations:\n")
        
        latest_reward = checkpoints[-1].get('mean_reward', 0)
        if latest_reward < 500:
            f.write("  - Training is in early stages. Continue training.\n")
        elif latest_reward < 900:
            f.write("  - Making progress. Consider:\n")
            f.write("    * Increasing batch size for stability\n")
            f.write("    * Adjusting learning rate if plateaued\n")
            f.write("    * Using train_advanced.py for curriculum learning\n")
        else:
            f.write("  - Excellent performance achieved!\n")
            f.write("    * Consider fine-tuning for specific behaviors\n")
            f.write("    * Test with visualize_robot.py\n")
        
        # Check for plateau
        if len(checkpoints) > 20:
            recent_rewards = [c.get('mean_reward', 0) for c in checkpoints[-20:]]
            if max(recent_rewards) - min(recent_rewards) < 50:
                f.write("  - ⚠️ Training appears to be plateauing\n")
                f.write("    * Try curriculum learning (train_advanced.py)\n")
                f.write("    * Adjust exploration parameters\n")
    
    print(f"\nReport saved to {output_file}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze training progress")
    parser.add_argument("--checkpoint_dir", default="checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--plot", action="store_true",
                       help="Generate plots")
    parser.add_argument("--report", action="store_true", 
                       help="Generate text report")
    
    args = parser.parse_args()
    
    # Load checkpoints
    checkpoints = load_all_checkpoints(args.checkpoint_dir)
    
    # Analyze
    analyze_progress(checkpoints)
    
    # Find best
    best, best_reward = find_best_checkpoint(checkpoints)
    if best:
        print(f"\n🏆 Best checkpoint: {best['file']} (Reward: {best_reward:.2f})")
    
    # Generate outputs
    if args.plot:
        plot_training_curves(checkpoints)
    
    if args.report:
        generate_report(args.checkpoint_dir)

if __name__ == "__main__":
    main()