#!/usr/bin/env python3
"""
간단한 MuJoCo 뷰어 - npz 체크포인트 시각화
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import sys
import os

def load_policy(npz_path):
    """npz 파일에서 정책 로드"""
    if not os.path.exists(npz_path):
        print(f"Error: {npz_path} not found!")
        return None, None, None
    
    data = np.load(npz_path, allow_pickle=True)
    theta = data['theta']
    obs_dim = int(data['obs_dim'])
    act_dim = int(data['act_dim'])
    
    print(f"\n✅ Loaded checkpoint: {npz_path}")
    print(f"   Training iterations: {data['iter']}")
    print(f"   Obs dim: {obs_dim}, Act dim: {act_dim}")
    
    # 메트릭 정보가 있으면 출력
    if 'metrics' in data.files:
        try:
            metrics = data['metrics'].item()
            print("   Performance metrics:")
            for key, value in metrics.items():
                print(f"     - {key}: {value}")
        except:
            pass
    
    return theta, obs_dim, act_dim

def get_observation(model, data):
    """현재 상태에서 관측값 추출"""
    obs = []
    
    # 위치와 방향
    if model.nq >= 7:
        obs.extend(data.qpos[:3])  # position
        obs.extend(data.qpos[3:7])  # quaternion
        obs.extend(data.qpos[7:])   # joint angles
    else:
        obs.extend(data.qpos)
    
    # 속도
    if model.nv >= 6:
        obs.extend(data.qvel[:6])   # base velocity
        obs.extend(data.qvel[6:])   # joint velocities
    else:
        obs.extend(data.qvel)
    
    return np.array(obs, dtype=np.float32)

def apply_policy(theta, obs, obs_dim, act_dim):
    """정책 적용"""
    # 관측값 크기 맞추기
    if len(obs) > obs_dim:
        obs = obs[:obs_dim]
    elif len(obs) < obs_dim:
        obs = np.pad(obs, (0, obs_dim - len(obs)), 'constant')
    
    # Weight와 Bias 추출
    W = theta[:obs_dim * act_dim].reshape(act_dim, obs_dim)
    b = theta[obs_dim * act_dim:obs_dim * act_dim + act_dim]
    
    # 행동 계산
    action = np.tanh(W @ obs + b)
    return action

def main():
    # 인자 파싱
    if len(sys.argv) < 2:
        print("Usage: python simple_viewer.py <checkpoint.npz> [robot.xml]")
        print("\nExample:")
        print("  python simple_viewer.py checkpoint.npz")
        print("  python simple_viewer.py checkpoint.npz quadruped.xml")
        
        # 현재 디렉토리의 npz 파일 찾기
        import glob
        npz_files = glob.glob("*.npz") + glob.glob("checkpoints/*.npz")
        if npz_files:
            print(f"\nFound npz files:")
            for f in npz_files[:5]:
                print(f"  - {f}")
        return
    
    npz_path = sys.argv[1]
    xml_path = sys.argv[2] if len(sys.argv) > 2 else "quadruped.xml"
    
    # XML 파일 확인
    if not os.path.exists(xml_path):
        print(f"Error: Robot model '{xml_path}' not found!")
        print("Make sure quadruped.xml is in the current directory")
        return
    
    # 정책 로드
    theta, obs_dim, act_dim = load_policy(npz_path)
    if theta is None:
        return
    
    # MuJoCo 모델 로드
    print(f"\n📦 Loading robot model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    print("\n" + "="*50)
    print(" Simple MuJoCo Viewer")
    print("="*50)
    print("\n🎮 Controls:")
    print("  Mouse: Rotate/zoom camera")
    print("  Space: Pause/resume")
    print("  ESC/Q: Exit")
    print("\n🤖 Robot will execute the trained policy automatically")
    print("="*50 + "\n")
    
    # 초기 자세 설정 (앉은 자세)
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name and 'knee' in joint_name:
            qpos_addr = model.jnt_qposadr[i]
            joint_range = model.jnt_range[i]
            # 80% 굽힘
            data.qpos[qpos_addr] = joint_range[0] + 0.8 * (joint_range[1] - joint_range[0])
    
    if model.nq >= 3:
        data.qpos[2] = 0.25  # 낮은 시작 높이
    
    mujoco.mj_forward(model, data)
    
    # 뷰어 실행
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 카메라 설정
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 45
        
        # 시뮬레이션 루프
        start_time = time.time()
        step_count = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # 관측값 얻기
            obs = get_observation(model, data)
            
            # 정책 적용
            action = apply_policy(theta, obs, obs_dim, act_dim)
            
            # 액션을 제어 입력으로 변환
            data.ctrl[:] = action * model.actuator_ctrlrange[:, 1]
            
            # 물리 시뮬레이션 스텝
            mujoco.mj_step(model, data)
            
            # 뷰어 업데이트
            viewer.sync()
            
            # FPS 제한 (60 FPS)
            time_until_next = model.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
            
            step_count += 1
            
            # 상태 출력 (1초마다)
            if step_count % 60 == 0:
                elapsed = time.time() - start_time
                height = data.qpos[2] if model.nq >= 3 else 0
                print(f"[{elapsed:6.1f}s] Height: {height:.3f}m", end='\r')
    
    print("\n\n✅ Viewer closed")

if __name__ == "__main__":
    main()