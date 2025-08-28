#!/usr/bin/env python3
"""
체크포인트(.npz) 파일만으로 로봇 제어하기
"""

import numpy as np
import jax.numpy as jnp

def load_and_use_checkpoint(checkpoint_path):
    """npz 파일만으로 정책 사용하기"""
    
    # 1. npz 파일 로드
    data = np.load(checkpoint_path, allow_pickle=True)
    
    # 2. 저장된 데이터 확인
    print("="*50)
    print(f"Checkpoint: {checkpoint_path}")
    print("="*50)
    print(f"Files in npz: {data.files}")
    print(f"Training iteration: {data['iter']}")
    print(f"Observation dim: {data['obs_dim']}")  
    print(f"Action dim: {data['act_dim']}")
    
    # 3. 정책 파라미터 추출
    theta = data['theta']
    obs_dim = int(data['obs_dim'])
    act_dim = int(data['act_dim'])
    
    print(f"Policy parameters shape: {theta.shape}")
    print(f"Total parameters: {len(theta)}")
    
    # 4. 정책 함수 정의 (npz만 있으면 이렇게 사용 가능)
    def policy(observation):
        """훈련된 정책으로 행동 계산"""
        # 파라미터를 Weight와 Bias로 분리
        W = theta[:obs_dim * act_dim].reshape(act_dim, obs_dim)
        b = theta[obs_dim * act_dim:obs_dim * act_dim + act_dim]
        
        # 선형 정책 + tanh 활성화
        action = np.tanh(W @ observation + b)
        return action
    
    # 5. 테스트 - 랜덤 관측값으로 행동 생성
    test_obs = np.random.randn(obs_dim)
    action = policy(test_obs)
    
    print(f"\nTest run:")
    print(f"  Input observation shape: {test_obs.shape}")
    print(f"  Output action shape: {action.shape}")
    print(f"  Action values: {action}")
    
    return policy

def export_checkpoint_info(checkpoint_path, output_txt="checkpoint_info.txt"):
    """체크포인트 정보를 텍스트 파일로 저장"""
    
    data = np.load(checkpoint_path, allow_pickle=True)
    
    with open(output_txt, 'w') as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write("="*50 + "\n\n")
        
        f.write("Basic Info:\n")
        f.write(f"  Training iterations: {data['iter']}\n")
        f.write(f"  Observation dimension: {data['obs_dim']}\n")
        f.write(f"  Action dimension: {data['act_dim']}\n")
        f.write(f"  Parameter count: {len(data['theta'])}\n\n")
        
        # 메트릭이 있으면 추가
        if 'metrics' in data.files:
            metrics = data['metrics'].item()
            f.write("Performance Metrics:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\nHow to use:\n")
        f.write("1. Load npz file: data = np.load('checkpoint.npz')\n")
        f.write("2. Extract theta: theta = data['theta']\n")
        f.write("3. Build policy with theta parameters\n")
        f.write("4. Run policy: action = policy(observation)\n")
    
    print(f"Info saved to {output_txt}")

def standalone_policy_runner():
    """npz 파일만으로 독립 실행 가능한 정책"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .npz checkpoint file")
    parser.add_argument("--export", action="store_true", help="Export checkpoint info")
    args = parser.parse_args()
    
    if args.export:
        export_checkpoint_info(args.checkpoint)
    
    # 정책 로드 및 테스트
    policy = load_and_use_checkpoint(args.checkpoint)
    
    print("\n" + "="*50)
    print("Policy loaded successfully!")
    print("You can now use this policy with just the npz file.")
    print("="*50)

if __name__ == "__main__":
    # 예시: 직접 실행
    import sys
    if len(sys.argv) > 1:
        standalone_policy_runner()
    else:
        # 기본 테스트
        print("Usage: python test_checkpoint.py <checkpoint.npz>")
        print("\nExample with existing checkpoint:")
        
        from pathlib import Path
        checkpoints = list(Path("checkpoints").glob("*.npz"))
        if checkpoints:
            latest = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
            print(f"  python test_checkpoint.py {latest}")
            load_and_use_checkpoint(str(latest))