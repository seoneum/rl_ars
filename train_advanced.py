#!/usr/bin/env python3
"""
Advanced Quadruped Training - 900점 돌파를 위한 개선 버전
주요 개선사항:
1. Curriculum Learning - 점진적 난이도 증가
2. Improved Reward Shaping - 더 세밀한 보상 설계
3. Adaptive Learning Rate - 학습률 자동 조절
4. Better Exploration - 향상된 탐색 전략
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
from dataclasses import replace
from typing import Tuple, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax, vmap, pmap
from jax.flatten_util import ravel_pytree
from tqdm import trange
import mujoco
from mujoco import mjx
from pathlib import Path
import json

# ==============================================================================
# 1. Enhanced Checkpoint System
# ==============================================================================
def save_checkpoint(path, theta, key, it, obs_dim, act_dim, metrics=None, compressed=True):
    """향상된 체크포인트 저장"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp_path = path + ".tmp"
    saver = np.savez_compressed if compressed else np.savez
    
    save_dict = {
        "theta": np.asarray(theta, dtype=np.float32),
        "key": np.asarray(key, dtype=np.uint32),
        "iter": np.int64(it),
        "obs_dim": np.int32(obs_dim),
        "act_dim": np.int32(act_dim)
    }
    
    if metrics:
        save_dict["metrics"] = metrics
    
    with open(tmp_path, "wb") as f:
        saver(f, **save_dict)
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved at iteration {it}")

def load_checkpoint(path):
    """체크포인트 로드"""
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        result = {
            "theta": jnp.array(d["theta"]),
            "obs_dim": int(d["obs_dim"]),
            "act_dim": int(d["act_dim"]),
            "iter": int(d["iter"]) if "iter" in d.files else 0,
            "key": jnp.array(d["key"]) if "key" in d.files else None
        }
        if "metrics" in d.files:
            result["metrics"] = d["metrics"].item()
        return result
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None

# ==============================================================================
# 2. Advanced MJX Environment with Curriculum Learning
# ==============================================================================
def make_advanced_mjx_env(xml_path: str, curriculum_stage: int = 0):
    """
    Curriculum Learning을 적용한 환경 생성
    
    Stage 0: 앉기 -> 일어서기 (기본)
    Stage 1: 일어서서 균형 유지
    Stage 2: 일어서서 걷기 준비
    Stage 3: 실제 걷기
    """
    m = mujoco.MjModel.from_xml_path(xml_path)
    mm = mjx.put_model(m)
    
    nu, nv, nq = int(m.nu), int(m.nv), int(m.nq)
    
    # Control limits
    ctrl_low = jnp.array(m.actuator_ctrlrange[:, 0], dtype=np.float32)
    ctrl_high = jnp.array(m.actuator_ctrlrange[:, 1], dtype=np.float32)
    
    # Find knee joints
    knee_jids = []
    hip_jids = []
    for jid in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        nm = nm if isinstance(nm, str) else (nm.decode() if nm else "")
        if 'knee' in nm:
            knee_jids.append(jid)
        elif 'hip' in nm:
            hip_jids.append(jid)
    
    import numpy as _np
    knee_jids = _np.array(knee_jids, dtype=_np.int32)
    hip_jids = _np.array(hip_jids, dtype=_np.int32)
    
    knee_qadr = jnp.array(m.jnt_qposadr[knee_jids], dtype=jnp.int32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.int32)
    knee_min = jnp.array(m.jnt_range[knee_jids, 0], dtype=jnp.float32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.float32)
    knee_max = jnp.array(m.jnt_range[knee_jids, 1], dtype=jnp.float32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.float32)
    
    hip_qadr = jnp.array(m.jnt_qposadr[hip_jids], dtype=jnp.int32) if hip_jids.size > 0 else jnp.zeros((0,), dtype=jnp.int32)
    
    # Curriculum-specific parameters
    if curriculum_stage == 0:
        # Stage 0: 앉기 -> 서기
        action_repeat = 3
        z_target = 0.5
        knee_target = 0.6  # 60% extension
        balance_weight = 0.5
        velocity_penalty = 0.1
        walk_reward = 0.0
    elif curriculum_stage == 1:
        # Stage 1: 균형 유지
        action_repeat = 2
        z_target = 0.5
        knee_target = 0.65
        balance_weight = 1.0
        velocity_penalty = 0.05
        walk_reward = 0.0
    elif curriculum_stage == 2:
        # Stage 2: 걷기 준비
        action_repeat = 2
        z_target = 0.52
        knee_target = 0.65
        balance_weight = 0.8
        velocity_penalty = 0.02
        walk_reward = 0.1
    else:
        # Stage 3: 실제 걷기
        action_repeat = 1
        z_target = 0.52
        knee_target = 0.65
        balance_weight = 0.5
        velocity_penalty = 0.01
        walk_reward = 0.5
    
    # JIT compiled functions
    @jax.jit
    def reset(key):
        """환경 리셋 - Curriculum에 따라 다른 초기 자세"""
        mdata = mjx.make_data(mm)
        
        # 무릎 초기 각도 설정 (stage에 따라 다름)
        if curriculum_stage == 0:
            knee_init = 0.8  # 80% flexion (앉은 자세)
            z_init = 0.25
        elif curriculum_stage == 1:
            knee_init = 0.5  # 50% flexion (중간 자세)
            z_init = 0.35
        else:
            knee_init = 0.4  # 40% flexion (거의 선 자세)
            z_init = 0.4
        
        qpos = mdata.qpos
        qpos = qpos.at[knee_qadr].set(knee_min + knee_init * (knee_max - knee_min))
        if nq >= 3:
            qpos = qpos.at[2].set(z_init)
        
        # 약간의 랜덤성 추가
        key, subkey = random.split(key)
        noise = random.normal(subkey, shape=qpos.shape) * 0.01
        qpos = qpos + noise
        
        mdata = mdata.replace(qpos=qpos, qvel=jnp.zeros_like(mdata.qvel))
        mdata = mjx.forward(mm, mdata)
        
        obs = get_obs(mdata)
        return mdata, obs, key
    
    @jax.jit
    def get_obs(data):
        """향상된 관측값 - 더 많은 정보 포함"""
        obs_components = []
        
        # 기본 상태
        if nq >= 7:
            obs_components.append(data.qpos[:3])  # position
            obs_components.append(data.qpos[3:7])  # quaternion
            obs_components.append(data.qpos[7:])   # joint angles
        else:
            obs_components.append(data.qpos)
        
        # 속도
        if nv >= 6:
            obs_components.append(data.qvel[:6])   # base velocity
            obs_components.append(data.qvel[6:])   # joint velocities
        else:
            obs_components.append(data.qvel)
        
        # 추가 정보 (curriculum stage에 따라)
        if curriculum_stage >= 1:
            # 가속도 정보 추가
            obs_components.append(data.qacc[:min(6, data.qacc.shape[0])])
        
        if curriculum_stage >= 2:
            # Contact forces
            if hasattr(data, 'cfrc_ext'):
                obs_components.append(data.cfrc_ext[:4].flatten())
        
        return jnp.concatenate(obs_components)
    
    @jax.jit
    def compute_reward(data, prev_data, action):
        """Curriculum-aware reward function"""
        z = data.qpos[2] if nq >= 3 else 0.5
        
        # 1. Height reward (stage별 다른 목표)
        height_reward = jnp.exp(-2.0 * jnp.square(z - z_target))
        
        # 2. Knee extension reward
        knee_angles = data.qpos[knee_qadr] if knee_qadr.size > 0 else jnp.zeros(1)
        knee_norm = (knee_angles - knee_min) / (knee_max - knee_min + 1e-6)
        knee_reward = jnp.exp(-2.0 * jnp.square(knee_norm - knee_target).mean())
        
        # 3. Upright bonus (orientation)
        if nq >= 7:
            quat = data.qpos[3:7]
            up_vec = jnp.array([2*(quat[1]*quat[3] - quat[0]*quat[2]),
                               2*(quat[2]*quat[3] + quat[0]*quat[1]),
                               1 - 2*(quat[1]**2 + quat[2]**2)])
            upright = jnp.maximum(up_vec[2], 0.0)
        else:
            upright = 1.0
        
        # 4. Balance penalty (낮을수록 좋음)
        if nv >= 6:
            angular_vel = data.qvel[3:6]
            balance_penalty = -balance_weight * jnp.sum(jnp.square(angular_vel))
        else:
            balance_penalty = 0.0
        
        # 5. Velocity penalty (너무 빠르게 움직이지 않도록)
        velocity = jnp.linalg.norm(data.qvel[:3]) if nv >= 3 else 0.0
        vel_penalty = -velocity_penalty * jnp.square(velocity)
        
        # 6. Walking reward (stage 2, 3에서만)
        if walk_reward > 0 and nv >= 3:
            forward_vel = data.qvel[0]  # x방향 속도
            walk_bonus = walk_reward * jnp.maximum(forward_vel, 0.0)
        else:
            walk_bonus = 0.0
        
        # 7. Action smoothness
        action_penalty = -0.01 * jnp.sum(jnp.square(action))
        
        # 8. Foot contact reward (stage 2, 3)
        if curriculum_stage >= 2:
            # 발이 번갈아 지면에 닿도록
            contact_reward = 0.0  # TODO: implement alternating foot contact
        else:
            contact_reward = 0.0
        
        # Total reward
        reward = (
            height_reward * 5.0 +
            knee_reward * 3.0 +
            upright * 2.0 +
            balance_penalty +
            vel_penalty +
            walk_bonus +
            action_penalty +
            contact_reward
        )
        
        # Success bonus
        success = (z > z_target - 0.05) & (upright > 0.9) & (knee_norm.mean() > knee_target - 0.1)
        reward = jnp.where(success, reward + 10.0, reward)
        
        return reward
    
    @jax.jit 
    def step(data, action):
        """환경 스텝"""
        # Rescale action
        ctrl = ctrl_low + (action + 1.0) * 0.5 * (ctrl_high - ctrl_low)
        
        # Multiple physics steps
        def body_fn(carry, _):
            d = carry
            d = d.replace(ctrl=ctrl)
            d = mjx.step(mm, d)
            return d, None
        
        prev_data = data
        data, _ = lax.scan(body_fn, data, None, length=action_repeat)
        
        obs = get_obs(data)
        reward = compute_reward(data, prev_data, action)
        
        # Termination conditions (stage별로 다름)
        if curriculum_stage == 0:
            # Stage 0: 넘어지면 종료
            done = (data.qpos[2] < 0.1) if nq >= 3 else False
        else:
            # Later stages: 더 엄격한 조건
            done = (data.qpos[2] < 0.15) if nq >= 3 else False
            if nq >= 7:
                # 너무 기울어지면 종료
                quat = data.qpos[3:7]
                up_vec_z = 1 - 2*(quat[1]**2 + quat[2]**2)
                done = done | (up_vec_z < 0.5)
        
        return data, obs, reward, done
    
    # Observation and action dimensions
    sample_data = mjx.make_data(mm)
    sample_obs = get_obs(sample_data)
    obs_dim = sample_obs.shape[0]
    act_dim = nu
    
    return {
        'reset': reset,
        'step': step,
        'obs_dim': obs_dim,
        'act_dim': act_dim,
        'batch_size': None,
        'curriculum_stage': curriculum_stage
    }

# ==============================================================================
# 3. Advanced ARS Algorithm with Adaptive Learning
# ==============================================================================
def train_advanced_ars(
    env_fn: Callable,
    key: jax.random.PRNGKey,
    num_iters: int = 2000,
    batch_size: int = 1024,
    step_size: float = 0.03,
    num_directions: int = 32,
    top_directions: int = 16,
    horizon: int = 1000,
    seed: int = 0,
    checkpoint_path: str = "checkpoints/advanced.ckpt",
    checkpoint_interval: int = 50,
    curriculum_stages: int = 4,
    stage_threshold: float = 850.0
):
    """
    Advanced ARS with:
    - Curriculum learning
    - Adaptive learning rate
    - Better exploration
    """
    
    # Initialize
    current_stage = 0
    env = env_fn(curriculum_stage=current_stage)
    obs_dim = env['obs_dim']
    act_dim = env['act_dim']
    
    print(f"Starting Advanced Training")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"Batch size: {batch_size}, Horizon: {horizon}")
    print(f"Curriculum stages: {curriculum_stages}")
    
    # Load checkpoint if exists
    start_iter = 0
    ckpt = load_checkpoint(checkpoint_path)
    if ckpt is not None:
        theta = ckpt['theta']
        key = ckpt['key'] if ckpt['key'] is not None else key
        start_iter = ckpt['iter']
        if 'metrics' in ckpt and 'curriculum_stage' in ckpt['metrics']:
            current_stage = ckpt['metrics']['curriculum_stage']
        print(f"Resuming from iteration {start_iter}, stage {current_stage}")
    else:
        # Initialize policy with better initialization
        key, init_key = random.split(key)
        W = random.normal(init_key, (act_dim, obs_dim)) * 0.01
        b = jnp.zeros(act_dim)
        theta, _ = ravel_pytree((W, b))
    
    # Linear policy
    def policy(theta, obs):
        W = theta[:obs_dim * act_dim].reshape(act_dim, obs_dim)
        b = theta[obs_dim * act_dim:obs_dim * act_dim + act_dim]
        return jnp.tanh(W @ obs + b)
    
    # Vectorized rollout
    def rollout_batch(env_keys, theta_batch):
        def single_rollout(env_key, theta_single):
            data, obs, _ = env['reset'](env_key)
            
            def body_fn(carry, _):
                data, obs, total_r = carry
                action = policy(theta_single, obs)
                data, obs, reward, done = env['step'](data, action)
                total_r = total_r + reward
                return (data, obs, total_r), None
            
            (_, _, total_reward), _ = lax.scan(body_fn, (data, obs, 0.0), None, length=horizon)
            return total_reward
        
        return vmap(single_rollout)(env_keys, theta_batch)
    
    # Adaptive learning rate
    best_reward_history = []
    learning_rate = step_size
    
    # Training loop
    pbar = trange(start_iter, num_iters, desc="Training", unit="iter")
    for iteration in pbar:
        # Generate random directions
        key, *subkeys = random.split(key, num=2*num_directions+2)
        deltas = random.normal(subkeys[0], (num_directions, theta.shape[0])) * 0.1
        
        # Create perturbed parameters
        theta_plus = theta[None, :] + deltas
        theta_minus = theta[None, :] - deltas
        all_thetas = jnp.concatenate([theta_plus, theta_minus])
        
        # Rollout in batches
        all_rewards = []
        for i in range(0, 2*num_directions, batch_size // horizon):
            batch_thetas = all_thetas[i:i+batch_size//horizon]
            key, *env_keys = random.split(key, num=len(batch_thetas)+1)
            env_keys = jnp.array(env_keys)
            rewards = rollout_batch(env_keys, batch_thetas)
            all_rewards.append(rewards)
        
        all_rewards = jnp.concatenate(all_rewards)
        rewards_plus = all_rewards[:num_directions]
        rewards_minus = all_rewards[num_directions:]
        
        # Select top directions
        scores = jnp.maximum(rewards_plus, rewards_minus)
        top_indices = jnp.argsort(scores)[-top_directions:]
        
        # Compute update with adaptive learning rate
        gradient = jnp.zeros_like(theta)
        for idx in top_indices:
            gradient += (rewards_plus[idx] - rewards_minus[idx]) * deltas[idx]
        gradient /= (top_directions * jnp.std(all_rewards) + 1e-6)
        
        # Adaptive learning rate
        current_best = jnp.max(scores)
        if len(best_reward_history) >= 10:
            recent_improvement = current_best - np.mean(best_reward_history[-10:])
            if recent_improvement < 10:  # Plateau detected
                learning_rate *= 0.95
            elif recent_improvement > 50:  # Good progress
                learning_rate *= 1.05
            learning_rate = np.clip(learning_rate, step_size * 0.1, step_size * 2.0)
        best_reward_history.append(current_best)
        
        # Update parameters
        theta = theta + learning_rate * gradient
        
        # Curriculum progression
        mean_reward = jnp.mean(scores[top_indices])
        if mean_reward > stage_threshold and current_stage < curriculum_stages - 1:
            current_stage += 1
            env = env_fn(curriculum_stage=current_stage)
            stage_threshold *= 1.1  # Increase threshold for next stage
            print(f"\n🎯 Advanced to Stage {current_stage}! Mean reward: {mean_reward:.2f}")
        
        # Update progress bar
        pbar.set_postfix({
            'reward': f"{mean_reward:.2f}",
            'best': f"{current_best:.2f}",
            'lr': f"{learning_rate:.4f}",
            'stage': current_stage
        })
        
        # Save checkpoint
        if (iteration + 1) % checkpoint_interval == 0:
            metrics = {
                'mean_reward': float(mean_reward),
                'best_reward': float(current_best),
                'learning_rate': float(learning_rate),
                'curriculum_stage': int(current_stage)
            }
            save_checkpoint(checkpoint_path, theta, key, iteration + 1, 
                          obs_dim, act_dim, metrics)
            
            # Save best model separately
            if current_best > 900:
                best_path = checkpoint_path.replace('.ckpt', f'_best_{current_best:.0f}.ckpt')
                save_checkpoint(best_path, theta, key, iteration + 1,
                              obs_dim, act_dim, metrics)
    
    print(f"\nTraining complete! Final best reward: {current_best:.2f}")
    return theta, key


def main():
    parser = argparse.ArgumentParser(description="Advanced Quadruped Training")
    parser.add_argument("--checkpoint_path", type=str, 
                       default="checkpoints/advanced.ckpt",
                       help="Checkpoint file path")
    parser.add_argument("--num_iters", type=int, default=2000,
                       help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=1024,
                       help="Batch size for parallel rollouts")
    parser.add_argument("--step_size", type=float, default=0.03,
                       help="Initial learning rate")
    parser.add_argument("--horizon", type=int, default=1000,
                       help="Episode length")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    parser.add_argument("--curriculum_stages", type=int, default=4,
                       help="Number of curriculum stages")
    
    args = parser.parse_args()
    
    # Check device
    print(f"JAX devices: {jax.devices()}")
    
    # Create environment function
    def env_fn(curriculum_stage=0):
        return make_advanced_mjx_env("quadruped.xml", curriculum_stage=curriculum_stage)
    
    # Train
    key = jax.random.PRNGKey(args.seed)
    theta, key = train_advanced_ars(
        env_fn=env_fn,
        key=key,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
        step_size=args.step_size,
        horizon=args.horizon,
        checkpoint_path=args.checkpoint_path,
        curriculum_stages=args.curriculum_stages
    )


if __name__ == "__main__":
    main()