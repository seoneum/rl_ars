#!/usr/bin/env python3
"""
A100 Optimized Quadruped Standing Training
All-in-one training script for NVIDIA A100 GPUs
- No rendering/EGL dependencies (pure headless training)
- JAX CUDA12 compatible environment
"""

import os

# ==============================================================================
# 0. Environment Setup for A100 (before any JAX/MuJoCo import)
# ==============================================================================

# JAX: force CUDA backend with safe defaults
os.environ['JAX_PLATFORMS'] = 'cuda'  # Force CUDA backend
os.environ['JAX_ENABLE_X64'] = 'false'  # Use float32 for speed
os.environ['JAX_ENABLE_COMPILATION_CACHE'] = '1'
os.environ['JAX_COMPILATION_CACHE_DIR'] = os.path.expanduser('~/.cache/jax_a100')
os.makedirs(os.environ['JAX_COMPILATION_CACHE_DIR'], exist_ok=True)

# VRAM allocation policy (A100 optimized)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'  # Preallocate for performance
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.92'  # Use 92% of GPU memory

# Remove legacy XLA_FLAGS if present
os.environ.pop('XLA_FLAGS', None)

# CUDA settings
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async kernel launches
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Reduce TF logging

# ==============================================================================
# Imports
# ==============================================================================
from jax import config
config.update("jax_enable_x64", False)

import argparse
from dataclasses import replace
from typing import Tuple, Any, Callable
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, lax
from jax.flatten_util import ravel_pytree
from tqdm import trange
import mujoco
from mujoco import mjx

# ==============================================================================
# 1. Checkpoint Functions
# ==============================================================================
def save_checkpoint(path, theta, key, it, obs_dim, act_dim, compressed=True):
    """Save training state"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    tmp_path = path + ".tmp"
    saver = np.savez_compressed if compressed else np.savez
    with open(tmp_path, "wb") as f:
        saver(f,
            theta=np.asarray(theta, dtype=np.float32),
            key=np.asarray(key, dtype=np.uint32),
            iter=np.int64(it),
            obs_dim=np.int32(obs_dim),
            act_dim=np.int32(act_dim))
    os.replace(tmp_path, path)
    print(f"[Checkpoint] Saved at iteration {it}")

def load_checkpoint(path):
    """Load checkpoint if exists"""
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        return {
            "theta": jnp.array(d["theta"]),
            "obs_dim": int(d["obs_dim"]),
            "act_dim": int(d["act_dim"]),
            "iter": int(d["iter"]) if "iter" in d.files else 0,
            "key": jnp.array(d["key"]) if "key" in d.files else None
        }
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None

# ==============================================================================
# 2. MJX Environment
# ==============================================================================
def make_mjx_env(xml_path: str):
    """Create MJX environment optimized for standing task"""
    m = mujoco.MjModel.from_xml_path(xml_path)
    mm = mjx.put_model(m)
    
    nu, nv, nq = int(m.nu), int(m.nv), int(m.nq)
    
    # Control limits
    ctrl_low = jnp.array(m.actuator_ctrlrange[:, 0], dtype=np.float32)
    ctrl_high = jnp.array(m.actuator_ctrlrange[:, 1], dtype=np.float32)
    ctrl_center = (ctrl_low + ctrl_high) / 2.0
    ctrl_half = (ctrl_high - ctrl_low) / 2.0
    
    # Find knee joints
    knee_jids = []
    for jid in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        nm = nm if isinstance(nm, str) else (nm.decode() if nm else "")
        if 'knee' in nm:
            knee_jids.append(jid)
    
    import numpy as _np
    knee_jids = _np.array(knee_jids, dtype=_np.int32)
    knee_qadr = jnp.array(m.jnt_qposadr[knee_jids], dtype=jnp.int32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.int32)
    knee_min = jnp.array(m.jnt_range[knee_jids, 0], dtype=jnp.float32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.float32)
    knee_max = jnp.array(m.jnt_range[knee_jids, 1], dtype=jnp.float32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.float32)
    knee_span = jnp.maximum(knee_max - knee_min, 1e-6)
    
    # A100 Optimized Parameters
    action_repeat = 3
    z_th = jnp.float32(0.25)
    z_lo = jnp.float32(0.45)
    z_hi = jnp.float32(0.55)
    up_min = jnp.float32(0.85)
    crouch_r = jnp.float32(0.80)  # Start sitting
    knee_lo = jnp.float32(0.50)
    knee_hi = jnp.float32(0.70)
    
    qpos0 = jnp.array(m.qpos0, dtype=jnp.float32)
    zero_qvel = jnp.zeros((nv,), dtype=jnp.float32)
    zero_ctrl = jnp.zeros((nu,), dtype=jnp.float32)
    
    def scale_action(a_unit):
        return ctrl_center + ctrl_half * jnp.tanh(a_unit)
    
    def obs_from_dd(dd):
        return jnp.concatenate([dd.qpos[7:], dd.qvel], axis=0).astype(jnp.float32)
    
    def reset_single(key):
        dd = mjx.make_data(mm)
        k1, k2 = random.split(key)
        noise = 0.01 * random.normal(k1, (qpos0.shape[0],), dtype=jnp.float32)
        qpos = qpos0 + noise.at[:7].set(0.0)
        
        # Start in sitting position
        if knee_qadr.shape[0] > 0:
            knee_target = knee_min + crouch_r * knee_span
            qpos = qpos.at[knee_qadr].set(knee_target)
        
        # Initial pitch
        pitch = -0.08
        q_w, q_y = jnp.cos(pitch/2), jnp.sin(pitch/2)
        qpos = qpos.at[3:7].set(jnp.array([q_w, 0.0, q_y, 0.0], dtype=jnp.float32))
        
        dd = replace(dd, qpos=qpos, qvel=zero_qvel, ctrl=zero_ctrl)
        obs = obs_from_dd(dd)
        return (dd, obs, jnp.array(False), zero_ctrl, jnp.array(0.0)), obs
    
    def step_single(state, a_unit):
        dd, _, done_prev, prev_ctrl, streak = state
        ctrl = scale_action(a_unit)
        
        def integrate(dd_in):
            def body(carry, _):
                return mjx.step(mm, replace(carry, ctrl=ctrl)), None
            dd_out, _ = lax.scan(body, dd_in, xs=None, length=action_repeat)
            return dd_out
        
        dd = lax.cond(done_prev, lambda d: d, integrate, dd)
        obs = obs_from_dd(dd)
        
        # Calculate rewards
        base_z = dd.qpos[2].astype(jnp.float32)
        q = dd.qpos[3:7]
        up = jnp.clip(1.0 - 2.0 * (q[1]**2 + q[2]**2), 0.0, 1.0)
        
        # Standing rewards
        z_ok = (base_z > z_lo) & (base_z < z_hi)
        up_ok = up > up_min
        standing = z_ok & up_ok
        
        # Knee rewards
        knee_reward = jnp.float32(0.0)
        if knee_qadr.shape[0] > 0:
            kq = dd.qpos[knee_qadr]
            kr = jnp.clip((kq - knee_min) / knee_span, 0.0, 1.0)
            knee_ok = (kr > knee_lo) & (kr < knee_hi)
            knee_reward = 2.0 * jnp.mean(knee_ok.astype(jnp.float32))
        
        # Streak bonus
        streak_next = jnp.where(standing & (~done_prev), streak + 1.0, 0.0)
        streak_reward = 0.1 * jnp.tanh(streak_next / 30.0)
        
        # Penalties
        ctrl_cost = 1e-4 * jnp.sum(jnp.square(ctrl))
        vel_pen = 0.1 * jnp.sum(jnp.square(dd.qvel[:6].astype(jnp.float32)))
        
        # Total reward
        reward = (
            0.8 * standing.astype(jnp.float32) +  # Main standing reward
            1.5 * z_ok.astype(jnp.float32) +       # Height reward
            1.0 * up_ok.astype(jnp.float32) +      # Upright reward
            knee_reward +                           # Knee configuration
            streak_reward -                         # Continuous standing
            ctrl_cost -                             # Control penalty
            vel_pen                                 # Velocity penalty
        )
        
        # Termination
        done_now = (base_z < z_th) | (up < 0.15) | jnp.any(jnp.isnan(dd.qpos))
        done = done_prev | done_now
        
        new_state = (dd, obs, done, ctrl, streak_next)
        return new_state, (obs, reward, done)
    
    reset_batch = jax.jit(jax.vmap(reset_single))
    step_batch = jax.jit(jax.vmap(step_single))
    
    return reset_batch, step_batch, {"obs_dim": int(nq - 7 + nv), "act_dim": int(nu)}

# ==============================================================================
# 3. Policy Network
# ==============================================================================
def make_policy_fns(obs_dim: int, act_dim: int):
    """Create policy functions"""
    params_shape = (jnp.zeros((obs_dim, act_dim)), jnp.zeros((act_dim,)))
    _, unravel_fn = ravel_pytree(params_shape)
    
    @jax.jit
    def policy_apply(theta, obs):
        W, b = unravel_fn(theta)
        return jnp.tanh(obs @ W + b)
    
    def ravel_pytree_partial(pytree):
        return ravel_pytree(pytree)[0]
    
    return ravel_pytree_partial, policy_apply

# ==============================================================================
# 4. ARS Training Loop
# ==============================================================================
def train(xml_path="quadruped.xml", save_path="a100_policy.npz", iterations=200, resume=False):
    """Main training function optimized for A100"""
    
    # A100 Optimal Hyperparameters
    num_envs = 1024      # Large batch for A100
    num_dirs = 64        # More exploration directions
    top_dirs = 16        # Top directions to use
    episode_length = 200
    step_size = 0.010
    noise_std = 0.015
    eval_every = 10
    ckpt_every = 10
    
    print("=" * 60)
    print("A100 Optimized Quadruped Standing Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Environments: {num_envs}")
    print(f"  Directions: {num_dirs}")
    print(f"  Episode Length: {episode_length}")
    print(f"  Iterations: {iterations}")
    print(f"  JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS')}")
    print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION: {os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')}")
    print("=" * 60)
    
    # Create environment
    reset_batch, step_batch, info = make_mjx_env(xml_path)
    obs_dim, act_dim = info["obs_dim"], info["act_dim"]
    
    # Initialize policy
    key = random.PRNGKey(0)
    ravel, policy_apply = make_policy_fns(obs_dim, act_dim)
    key, kW = random.split(key)
    W0 = 0.01 * random.normal(kW, (obs_dim, act_dim))
    b0 = jnp.zeros((act_dim,))
    theta = ravel((W0, b0))
    
    # Resume if requested
    start_it = 0
    if resume and os.path.exists(save_path):
        ckpt = load_checkpoint(save_path)
        if ckpt:
            theta = ckpt["theta"]
            start_it = ckpt["iter"]
            if ckpt["key"] is not None:
                key = ckpt["key"]
            print(f"[Resume] Starting from iteration {start_it}")
    
    @jax.jit
    def rollout(theta_, keys):
        """Single rollout with given policy"""
        state, obs = reset_batch(keys)
        def body(carry, _):
            st, ob, ret = carry
            act = policy_apply(theta_, ob)
            st, (ob_next, r, done) = step_batch(st, act)
            return (st, ob_next, ret + r * (1.0 - done)), None
        (_, _, returns), _ = lax.scan(body, (state, obs, jnp.zeros(num_envs)), None, length=episode_length)
        return returns
    
    @jax.jit
    def eval_directions(theta_, deltas, keys):
        """Evaluate all directions in parallel"""
        def eval_one(delta, k):
            k_plus, k_minus = random.split(k)
            keys_plus = random.split(k_plus, num_envs)
            keys_minus = random.split(k_minus, num_envs)
            r_plus = rollout(theta_ + noise_std * delta, keys_plus).mean()
            r_minus = rollout(theta_ - noise_std * delta, keys_minus).mean()
            return r_plus, r_minus
        return jax.vmap(eval_one)(deltas, keys)
    
    # Training loop
    pbar = trange(iterations, desc="Training")
    for it_local in pbar:
        it = start_it + it_local
        
        # Generate random directions
        key, k_delta, k_eval = random.split(key, 3)
        deltas = random.normal(k_delta, (num_dirs, theta.shape[0]))
        eval_keys = random.split(k_eval, num_dirs)
        
        # Evaluate all directions (no chunking for A100)
        R_plus, R_minus = eval_directions(theta, deltas, eval_keys)
        
        # Select top directions
        scores = jnp.maximum(R_plus, R_minus)
        topk_idx = jnp.argsort(scores)[-top_dirs:]
        R_plus_top = R_plus[topk_idx]
        R_minus_top = R_minus[topk_idx]
        deltas_top = deltas[topk_idx]
        
        # Compute gradient estimate
        reward_std = jnp.std(jnp.concatenate([R_plus_top, R_minus_top])) + 1e-8
        grad = jnp.mean((R_plus_top - R_minus_top)[:, None] * deltas_top, axis=0) / reward_std
        
        # Update policy
        theta += step_size * grad
        
        # Display progress
        pbar.set_postfix({
            "mean_reward": f"{scores.mean():.2f}",
            "best": f"{scores.max():.2f}",
            "std": f"{scores.std():.2f}"
        })
        
        # Evaluation
        if (it + 1) % eval_every == 0:
            key, k_test = random.split(key)
            test_keys = random.split(k_test, num_envs)
            returns = rollout(theta, test_keys)
            print(f"\n[Iter {it+1}] Eval Return: {returns.mean():.2f} (±{returns.std():.2f})")
        
        # Save checkpoint
        if (it + 1) % ckpt_every == 0 or it_local == iterations - 1:
            save_checkpoint(save_path, theta, key, it + 1, obs_dim, act_dim)
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Final policy saved to: {save_path}")
    print("=" * 60)

# ==============================================================================
# 5. Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="A100 Optimized Quadruped Training")
    parser.add_argument("--xml", type=str, default="quadruped.xml", help="Robot model XML")
    parser.add_argument("--save", type=str, default="a100_policy.npz", help="Save path")
    parser.add_argument("--iterations", type=int, default=200, help="Training iterations")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Check GPU
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    
    # Check if GPU is available
    has_gpu = False
    for d in devices:
        if hasattr(d, 'platform'):
            if d.platform in ('gpu', 'cuda'):
                has_gpu = True
        elif hasattr(d, 'device_kind'):
            if d.device_kind == 'gpu':
                has_gpu = True
    
    if not has_gpu:
        print("WARNING: No GPU detected! Training will be slow.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Run training
    train(
        xml_path=args.xml,
        save_path=args.save,
        iterations=args.iterations,
        resume=args.resume
    )

if __name__ == "__main__":
    main()