#!/usr/bin/env python3
"""
MuJoCo 시각화 도구 - 로봇 모델과 훈련된 정책 확인
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
        
    def load_policy(self, checkpoint_path):
        """훈련된 정책 로드"""
        try:
            ckpt = np.load(checkpoint_path, allow_pickle=True)
            self.policy = ckpt['theta']
            self.obs_dim = int(ckpt['obs_dim'])
            self.act_dim = int(ckpt['act_dim'])
            print(f"Policy shape: {self.policy.shape}")
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
        
        # 간단한 선형 정책 (실제 훈련 코드와 동일하게)
        obs_flat = obs[:self.obs_dim] if len(obs) > self.obs_dim else obs
        W = self.policy[:self.obs_dim * self.act_dim].reshape(self.act_dim, self.obs_dim)
        b = self.policy[self.obs_dim * self.act_dim:self.obs_dim * self.act_dim + self.act_dim]
        
        action = np.tanh(W @ obs_flat + b)
        return action
    
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
        
        use_policy = False
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 초기 카메라 설정
            viewer.cam.distance = 2.0
            viewer.cam.elevation = -20
            viewer.cam.azimuth = 45
            
            start = time.time()
            
            while viewer.is_running():
                step_start = time.time()
                
                # 키보드 입력 처리
                if viewer._key_pressed.get(ord('r')) or viewer._key_pressed.get(ord('R')):
                    self.set_sitting_pose()
                    print("Reset to sitting pose")
                    viewer._key_pressed[ord('r')] = False
                    viewer._key_pressed[ord('R')] = False
                
                if viewer._key_pressed.get(ord('s')) or viewer._key_pressed.get(ord('S')):
                    self.set_standing_pose()
                    print("Changed to standing pose")
                    viewer._key_pressed[ord('s')] = False
                    viewer._key_pressed[ord('S')] = False
                
                if viewer._key_pressed.get(ord('p')) or viewer._key_pressed.get(ord('P')):
                    use_policy = not use_policy
                    status = "ON" if use_policy else "OFF"
                    print(f"Policy control: {status}")
                    viewer._key_pressed[ord('p')] = False
                    viewer._key_pressed[ord('P')] = False
                
                # 정책 적용
                if use_policy and self.policy is not None:
                    obs = self.get_observation()
                    action = self.apply_policy(obs)
                    self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
                
                # 시뮬레이션 스텝
                mujoco.mj_step(self.model, self.data)
                
                # 뷰어 동기화
                viewer.sync()
                
                # 타이밍 조절 (60 FPS)
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
    
    def run_evaluation(self, num_episodes=5, render=True):
        """훈련된 정책 평가"""
        if self.policy is None:
            print("No policy loaded!")
            return
        
        print(f"\nEvaluating policy for {num_episodes} episodes...")
        
        rewards = []
        
        for ep in range(num_episodes):
            self.set_sitting_pose()
            episode_reward = 0
            
            if render:
                with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                    viewer.cam.distance = 2.0
                    viewer.cam.elevation = -20
                    
                    for step in range(1000):  # 1000 스텝
                        obs = self.get_observation()
                        action = self.apply_policy(obs)
                        self.data.ctrl[:] = action * self.model.actuator_ctrlrange[:, 1]
                        
                        mujoco.mj_step(self.model, self.data)
                        
                        # 보상 계산 (간단한 버전)
                        z_height = self.data.qpos[2] if self.nq >= 3 else 0
                        reward = z_height  # 높이에 비례
                        episode_reward += reward
                        
                        viewer.sync()
                        time.sleep(0.002)
                        
                        if not viewer.is_running():
                            break
            
            rewards.append(episode_reward)
            print(f"Episode {ep+1}: Reward = {episode_reward:.2f}")
        
        print(f"\nAverage reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MuJoCo Quadruped Viewer")
    parser.add_argument("--xml", default="quadruped.xml", help="Robot model XML file")
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint file")
    parser.add_argument("--mode", choices=["interactive", "evaluate"], default="interactive",
                       help="Viewer mode: interactive or evaluate")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    
    args = parser.parse_args()
    
    # 체크포인트 자동 찾기
    if args.checkpoint is None:
        checkpoints = list(Path(".").glob("checkpoints/*.ckpt"))
        if checkpoints:
            args.checkpoint = str(sorted(checkpoints, key=os.path.getmtime)[-1])
            print(f"Auto-detected checkpoint: {args.checkpoint}")
    
    viewer = QuadrupedViewer(args.xml, args.checkpoint)
    
    if args.mode == "interactive":
        viewer.run_interactive()
    else:
        viewer.run_evaluation(num_episodes=args.episodes)


if __name__ == "__main__":
    main()