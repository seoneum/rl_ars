#!/usr/bin/env python3
"""
MuJoCo 시각화 도구 - 로봇 모델과 훈련된 정책 확인 (호환성 수정 버전)
"""

import os
import numpy as np
import mujoco
import mujoco.viewer
import time
from pathlib import Path

class QuadrupedViewer:
    def __init__(self, xml_path="quadruped.xml", checkpoint_path=None):
        """
        Args:
            xml_path: 로봇 모델 XML 파일 경로
            checkpoint_path: 훈련된 체크포인트 파일 경로 (옵션)
        """
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 제어 관련 설정
        self.nu = self.model.nu
        self.nv = self.model.nv
        self.nq = self.model.nq
        
        # 체크포인트 로드
        self.policy = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_policy(checkpoint_path)
            print(f"✅ Loaded policy from {checkpoint_path}")
        
        # 초기 자세 설정 (앉은 자세)
        self.set_sitting_pose()
        
        # 상태 변수
        self.use_policy = False
        self.paused = False
        
    def load_policy(self, checkpoint_path):
        """훈련된 정책 로드"""
        try:
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            self.policy = ckpt['theta']
            self.obs_dim = int(ckpt['obs_dim'])
            self.act_dim = int(ckpt['act_dim'])
            print(f"Policy shape: {self.policy.shape}")
            print(f"Obs dim: {self.obs_dim}, Act dim: {self.act_dim}")
            
            # 메트릭 정보 출력
            if 'metrics' in ckpt.files:
                metrics = ckpt['metrics'].item()
                print("\nTraining metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
            
            print(f"Training iterations: {ckpt['iter']}")
            
        except Exception as e:
            print(f"Failed to load policy: {e}")
            self.policy = None
    
    def set_sitting_pose(self):
        """앉은 자세로 초기화 (무릎 80% 굽힘)"""
        # 무릎 관절 찾기
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and 'knee' in joint_name:
                qpos_addr = self.model.jnt_qposadr[i]
                joint_range = self.model.jnt_range[i]
                # 80% 굽힘 설정
                self.data.qpos[qpos_addr] = joint_range[0] + 0.8 * (joint_range[1] - joint_range[0])
        
        # Z 위치 낮추기 (앉은 높이)
        if self.nq >= 3:
            self.data.qpos[2] = 0.25  # 낮은 높이
        
        mujoco.mj_forward(self.model, self.data)
    
    def set_standing_pose(self):
        """선 자세로 설정 (무릎 50-70% 펴짐)"""
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and 'knee' in joint_name:
                qpos_addr = self.model.jnt_qposadr[i]
                joint_range = self.model.jnt_range[i]
                # 60% 펴짐 (40% 굽힘)
                self.data.qpos[qpos_addr] = joint_range[0] + 0.4 * (joint_range[1] - joint_range[0])
        
        # Z 위치 높이기
        if self.nq >= 3:
            self.data.qpos[2] = 0.5  # 선 자세 높이
        
        mujoco.mj_forward(self.model, self.data)
    
    def get_observation(self):
        """현재 상태에서 관측값 얻기"""
        obs = []
        
        # 위치와 방향
        obs.extend(self.data.qpos[:3])  # x, y, z
        obs.extend(self.data.qpos[3:7])  # quaternion
        
        # 관절 각도
        obs.extend(self.data.qpos[7:])
        
        # 속도
        obs.extend(self.data.qvel[:6])  # linear & angular velocity
        obs.extend(self.data.qvel[6:])  # joint velocities
        
        return np.array(obs, dtype=np.float32)
    
    def apply_policy(self, obs):
        """정책을 사용해 행동 계산"""
        if self.policy is None:
            return np.zeros(self.nu)
        
        # 관측값 크기 조정
        if len(obs) > self.obs_dim:
            obs = obs[:self.obs_dim]
        elif len(obs) < self.obs_dim:
            # 부족한 부분을 0으로 채움
            obs = np.pad(obs, (0, self.obs_dim - len(obs)), 'constant')
        
        # 선형 정책
        W = self.policy[:self.obs_dim * self.act_dim].reshape(self.act_dim, self.obs_dim)
        b = self.policy[self.obs_dim * self.act_dim:self.obs_dim * self.act_dim + self.act_dim]
        
        action = np.tanh(W @ obs + b)
        return action
    
    def key_callback(self, key):
        """키보드 콜백 함수"""
        if key == ord(' '):
            self.paused = not self.paused
            print(f"Simulation {'paused' if self.paused else 'resumed'}")
        elif key == ord('r') or key == ord('R'):
            self.set_sitting_pose()
            print("Reset to sitting pose")
        elif key == ord('s') or key == ord('S'):
            self.set_standing_pose()
            print("Changed to standing pose")
        elif key == ord('p') or key == ord('P'):
            self.use_policy = not self.use_policy
            status = "ON" if self.use_policy else "OFF"
            print(f"Policy control: {status}")
    
    def run_interactive(self):
        """인터랙티브 뷰어 실행"""
        print("\n" + "="*50)
        print(" MuJoCo Quadruped Viewer")
        print("="*50)
        print("\n조작법:")
        print("  Space : 시뮬레이션 일시정지/재개")
        print("  R     : 앉은 자세로 리셋")
        print("  S     : 선 자세로 변경")
        print("  P     : 정책 실행 토글")
        print("  Q/ESC : 종료")
        print("  마우스 : 카메라 회전/줌")
        print("\n")
        
        if self.policy is not None:
            print("✅ Policy loaded and ready to use (Press 'P' to activate)")
        else:
            print("⚠️  No policy loaded (visualization only)")
        
        with mujoco.viewer.launch_passive(self.model, self.data, 
                                         key_callback=self.key_callback) as viewer:
            # 초기 카메라 설정
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
            
            start = time.time()
            
            while viewer.is_running():
                step_start = time.time()
                
                if not self.paused:
                    # 정책 적용
                    if self.use_policy and self.policy is not None:
                        obs = self.get_observation()
                        action = self.apply_policy(obs)
                        # 액션 스케일링
                        self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
                    
                    # 시뮬레이션 스텝
                    mujoco.mj_step(self.model, self.data)
                
                # 뷰어 동기화
                viewer.sync()
                
                # 타이밍 조절 (60 FPS)
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
    
    def run_evaluation(self, num_episodes=5, max_steps=1000):
        """훈련된 정책 평가 (렌더링 없이)"""
        if self.policy is None:
            print("No policy loaded!")
            return
        
        print(f"\nEvaluating policy for {num_episodes} episodes...")
        print(f"Max steps per episode: {max_steps}")
        
        rewards = []
        
        for ep in range(num_episodes):
            print(f"\nEpisode {ep+1}/{num_episodes}")
            self.set_sitting_pose()
            episode_reward = 0
            
            for step in range(max_steps):
                obs = self.get_observation()
                action = self.apply_policy(obs)
                self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
                
                mujoco.mj_step(self.model, self.data)
                
                # 보상 계산 (간단한 버전)
                z_height = self.data.qpos[2] if self.nq >= 3 else 0
                
                # 무릎 각도 체크
                knee_angles = []
                for i in range(self.model.njnt):
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if joint_name and 'knee' in joint_name:
                        qpos_addr = self.model.jnt_qposadr[i]
                        knee_angles.append(self.data.qpos[qpos_addr])
                
                # 보상: 높이 + 무릎 펴짐 정도
                height_reward = z_height * 10
                knee_reward = np.mean(knee_angles) * 2 if knee_angles else 0
                reward = height_reward + knee_reward
                
                episode_reward += reward
                
                # 진행 상황 출력 (100 스텝마다)
                if (step + 1) % 100 == 0:
                    print(f"  Step {step+1}: height={z_height:.3f}, reward={reward:.2f}")
            
            rewards.append(episode_reward / max_steps)
            print(f"  Episode reward: {episode_reward:.2f} (avg: {episode_reward/max_steps:.2f})")
        
        print("\n" + "="*50)
        print(f"Evaluation complete!")
        print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Best episode: {np.max(rewards):.2f}")
        print(f"Worst episode: {np.min(rewards):.2f}")
        print("="*50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MuJoCo Quadruped Viewer")
    parser.add_argument("--xml", default="quadruped.xml", help="Robot model XML file")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint file (.npz)")
    parser.add_argument("--mode", choices=["interactive", "evaluate"], default="interactive",
                       help="Viewer mode: interactive (with visualization) or evaluate (no rendering)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # XML 파일 확인
    if not os.path.exists(args.xml):
        print(f"Error: Robot model file '{args.xml}' not found!")
        return
    
    # 체크포인트 자동 찾기
    if args.checkpoint is None:
        # npz 파일 찾기
        npz_files = list(Path(".").glob("*.npz"))
        npz_files.extend(list(Path("checkpoints").glob("*.npz")) if os.path.exists("checkpoints") else [])
        
        if npz_files:
            # 가장 최근 파일 선택
            args.checkpoint = str(sorted(npz_files, key=os.path.getmtime)[-1])
            print(f"Auto-detected checkpoint: {args.checkpoint}")
    
    # 체크포인트 확인
    if args.checkpoint and not os.path.exists(args.checkpoint):
        print(f"Warning: Checkpoint file '{args.checkpoint}' not found!")
        args.checkpoint = None
    
    # 뷰어 생성
    viewer = QuadrupedViewer(args.xml, args.checkpoint)
    
    # 모드에 따라 실행
    if args.mode == "interactive":
        viewer.run_interactive()
    else:
        viewer.run_evaluation(num_episodes=args.episodes)


if __name__ == "__main__":
    main()