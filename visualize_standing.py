#!/usr/bin/env python3
"""
서기 학습된 정책 시각화 스크립트
MuJoCo viewer를 사용하여 로봇의 동작을 실시간으로 확인
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import os
import argparse

def load_policy(ckpt_path):
    """학습된 정책 로드"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"정책 파일을 찾을 수 없습니다: {ckpt_path}")
    
    ckpt = np.load(ckpt_path, allow_pickle=True)
    theta = ckpt["theta"]
    obs_dim = int(ckpt["obs_dim"])
    act_dim = int(ckpt["act_dim"])
    
    # 정책 파라미터 언팩
    W_size = obs_dim * act_dim
    W = theta[:W_size].reshape(obs_dim, act_dim)
    b = theta[W_size:W_size + act_dim]
    
    def policy(obs):
        """정책 실행 함수"""
        return np.tanh(obs @ W + b)
    
    return policy, obs_dim, act_dim

def visualize_policy(xml_path, ckpt_path, duration=30, slow_motion=1.0):
    """정책 시각화"""
    
    # MuJoCo 모델 로드
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 정책 로드
    policy, obs_dim, act_dim = load_policy(ckpt_path)
    print(f"정책 로드 완료: obs_dim={obs_dim}, act_dim={act_dim}")
    
    # 초기 자세 설정 (앉은 자세)
    # 무릎 관절 찾기
    knee_joints = []
    for i in range(model.njnt):
        joint_name = model.joint(i).name
        if 'knee' in joint_name:
            knee_joints.append(i)
    
    print(f"무릎 관절 발견: {[model.joint(i).name for i in knee_joints]}")
    
    # 뷰어 생성
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # 초기화
        mujoco.mj_resetData(model, data)
        
        # 앉은 자세로 초기화
        if knee_joints:
            for knee_id in knee_joints:
                qpos_addr = model.jnt_qposadr[knee_id]
                joint_range = model.jnt_range[knee_id]
                # 80% 굴곡 (앉은 자세)
                target_angle = joint_range[0] + 0.8 * (joint_range[1] - joint_range[0])
                data.qpos[qpos_addr] = target_angle
        
        # 몸통 pitch 설정 (약간 앞으로 기울임)
        pitch = -0.08
        half_pitch = 0.5 * pitch
        q_w = np.cos(half_pitch)
        q_y = np.sin(half_pitch)
        data.qpos[3:7] = [q_w, 0, q_y, 0]  # quaternion (w,x,y,z)
        
        # Forward kinematics
        mujoco.mj_forward(model, data)
        
        start_time = time.time()
        step_count = 0
        
        print(f"\n시뮬레이션 시작 ({duration}초간 실행)")
        print("로봇이 앉은 자세에서 일어서는 것을 관찰하세요...")
        print("-" * 50)
        
        # 시뮬레이션 루프
        while viewer.is_running() and (time.time() - start_time) < duration:
            
            # 관측값 생성
            obs = np.concatenate([
                data.qpos[7:],  # joint positions (excluding root)
                data.qvel        # joint velocities
            ])
            
            # 정책 실행
            action = policy(obs)
            
            # 액션을 컨트롤 범위로 변환
            ctrl_range = model.actuator_ctrlrange
            ctrl_center = (ctrl_range[:, 0] + ctrl_range[:, 1]) / 2.0
            ctrl_half = (ctrl_range[:, 1] - ctrl_range[:, 0]) / 2.0
            data.ctrl[:] = ctrl_center + ctrl_half * np.tanh(action)
            
            # 물리 시뮬레이션 스텝
            mujoco.mj_step(model, data)
            
            # 뷰어 동기화
            viewer.sync()
            
            # 속도 조절
            if slow_motion > 1.0:
                time.sleep(model.opt.timestep * (slow_motion - 1))
            
            step_count += 1
            
            # 상태 정보 출력 (1초마다)
            if step_count % int(1.0 / model.opt.timestep) == 0:
                torso_height = data.qpos[2]
                torso_quat = data.qpos[3:7]
                upright = 1.0 - 2.0 * (torso_quat[1]**2 + torso_quat[2]**2)
                
                # 무릎 각도 평균
                if knee_joints:
                    knee_angles = []
                    for knee_id in knee_joints:
                        qpos_addr = model.jnt_qposadr[knee_id]
                        joint_range = model.jnt_range[knee_id]
                        angle = data.qpos[qpos_addr]
                        ratio = (angle - joint_range[0]) / (joint_range[1] - joint_range[0])
                        knee_angles.append(ratio)
                    knee_avg = np.mean(knee_angles)
                else:
                    knee_avg = 0.0
                
                elapsed = time.time() - start_time
                print(f"[{elapsed:5.1f}s] 높이: {torso_height:.3f}m, "
                      f"수직도: {upright:.3f}, 무릎: {knee_avg:.2%}")
        
        print("-" * 50)
        print("시뮬레이션 종료")

def main():
    parser = argparse.ArgumentParser(description="서기 학습 정책 시각화")
    parser.add_argument("--xml", type=str, default="quadruped.xml",
                       help="로봇 모델 XML 파일")
    parser.add_argument("--policy", type=str, default=None,
                       help="정책 파일 경로 (.npz)")
    parser.add_argument("--duration", type=int, default=30,
                       help="시뮬레이션 시간 (초)")
    parser.add_argument("--slow", type=float, default=1.0,
                       help="슬로우모션 배율 (1.0 = 정상속도)")
    
    args = parser.parse_args()
    
    # 정책 파일 자동 선택
    if args.policy is None:
        policy_files = [
            "ars_standing_phase2.npz",
            "ars_standing_phase1.npz",
            "ars_policy1.npz"
        ]
        
        for f in policy_files:
            if os.path.exists(f):
                args.policy = f
                print(f"자동 선택된 정책 파일: {f}")
                break
    
    if args.policy is None:
        print("정책 파일을 찾을 수 없습니다.")
        print("먼저 학습을 실행하세요: python train_standing.py")
        return
    
    visualize_policy(args.xml, args.policy, args.duration, args.slow)

if __name__ == "__main__":
    main()