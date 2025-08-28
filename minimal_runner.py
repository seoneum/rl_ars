#!/usr/bin/env python3
"""
최소 실행 코드 - npz 파일만 있으면 실행 가능
외부 의존성 최소화 버전
"""

import numpy as np

class MinimalQuadrupedPolicy:
    """npz 파일만으로 실행 가능한 최소 정책 클래스"""
    
    def __init__(self, npz_path):
        """
        Args:
            npz_path: 체크포인트 npz 파일 경로
        """
        # npz 로드
        self.data = np.load(npz_path, allow_pickle=True)
        self.theta = self.data['theta']
        self.obs_dim = int(self.data['obs_dim'])
        self.act_dim = int(self.data['act_dim'])
        
        # Weight와 Bias 분리
        param_size = self.obs_dim * self.act_dim
        self.W = self.theta[:param_size].reshape(self.act_dim, self.obs_dim)
        self.b = self.theta[param_size:param_size + self.act_dim]
        
        print(f"Policy loaded: obs_dim={self.obs_dim}, act_dim={self.act_dim}")
    
    def get_action(self, observation):
        """
        관측값을 받아 행동 반환
        
        Args:
            observation: numpy array of shape (obs_dim,)
        
        Returns:
            action: numpy array of shape (act_dim,)
        """
        # 선형 변환 + tanh 활성화
        action = np.tanh(self.W @ observation + self.b)
        return action
    
    def get_info(self):
        """체크포인트 정보 반환"""
        info = {
            'iterations': int(self.data['iter']),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
            'param_count': len(self.theta)
        }
        
        # 메트릭이 있으면 추가
        if 'metrics' in self.data.files:
            info['metrics'] = self.data['metrics'].item()
        
        return info


def demo_usage():
    """사용 예시"""
    print("="*60)
    print(" Minimal Quadruped Policy Runner")
    print("="*60)
    
    # 1. 체크포인트 찾기
    from pathlib import Path
    checkpoints = list(Path("checkpoints").glob("*.npz"))
    
    if not checkpoints:
        print("No checkpoints found in checkpoints/ directory")
        print("Train first with: python train_a100.py")
        return
    
    # 가장 최근 체크포인트 사용
    latest = sorted(checkpoints, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\nUsing checkpoint: {latest}")
    
    # 2. 정책 로드
    policy = MinimalQuadrupedPolicy(str(latest))
    
    # 3. 정보 출력
    info = policy.get_info()
    print(f"\nCheckpoint info:")
    for key, value in info.items():
        if key != 'metrics':
            print(f"  {key}: {value}")
    
    if 'metrics' in info:
        print("\nPerformance metrics:")
        for key, value in info['metrics'].items():
            print(f"  {key}: {value}")
    
    # 4. 테스트 실행
    print("\n" + "-"*40)
    print("Testing policy with random observations:")
    print("-"*40)
    
    for i in range(3):
        # 랜덤 관측값 생성
        obs = np.random.randn(policy.obs_dim) * 0.1
        
        # 행동 계산
        action = policy.get_action(obs)
        
        print(f"\nTest {i+1}:")
        print(f"  Observation (first 5): {obs[:5]}")
        print(f"  Action: {action}")
    
    print("\n" + "="*60)
    print("✅ Success! Policy is working with just the npz file.")
    print("="*60)


def main():
    """명령줄 실행"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Run quadruped policy from npz checkpoint only"
    )
    parser.add_argument(
        "checkpoint", 
        nargs='?',
        help="Path to .npz checkpoint file"
    )
    parser.add_argument(
        "--test-steps", 
        type=int, 
        default=10,
        help="Number of test steps to run"
    )
    
    args = parser.parse_args()
    
    if args.checkpoint:
        # 특정 체크포인트 사용
        policy = MinimalQuadrupedPolicy(args.checkpoint)
        
        print(f"\nRunning {args.test_steps} test steps...")
        for step in range(args.test_steps):
            obs = np.random.randn(policy.obs_dim) * 0.1
            action = policy.get_action(obs)
            if step % 5 == 0:
                print(f"  Step {step}: action = {action[:3]}...")
        
        print("\n✅ Policy test complete!")
    else:
        # 데모 실행
        demo_usage()


if __name__ == "__main__":
    main()