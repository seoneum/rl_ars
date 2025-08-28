#!/usr/bin/env python3
"""
Visualize quadruped standing policy (interactive or headless)
- If --policy is provided, tries to load linear ARS policy from npz.
- Otherwise falls back to simple PD standing controller.
- Interactive viewer if DISPLAY available, else offscreen; optional --video.

Usage:
  python visualize_standing.py --duration 30 --slow 1.0
  python visualize_standing.py --policy ars_standing_phase2.npz --duration 20
  python visualize_standing.py --video out.mp4 --duration 10 --slow 1.0
"""

import os
import sys
import time
import argparse
import numpy as np

# Prefer EGL in headless envs
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import mujoco.viewer

def get_obs(model, data):
    # Drop free root (7 qpos for free joint, 6 qvel root dofs)
    nq = model.nq
    nv = model.nv
    # If model has a free root, first body is free; common for this quadruped
    # qpos: [x, y, z, qw, qx, qy, qz, joint...]
    # qvel: [vx, vy, vz, wx, wy, wz, joint_vel...]
    has_free = (model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE) if model.njnt > 0 else False
    if has_free:
        q = data.qpos[7:].copy()
        dq = data.qvel[6:].copy()
    else:
        q = data.qpos.copy()
        dq = data.qvel.copy()
    return np.concatenate([q, dq]).astype(np.float32)

def try_load_linear_policy(npz_path):
    """
    Tries to construct a linear policy u = W @ (normed_obs) + b
    Expects keys like: W, b, or theta, and possibly obs_mean/std in 'meta' or top-level.
    Returns callable(obs)->ctrl or None if not possible.
    """
    try:
        ckpt = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"✗ Failed to load policy: {e}")
        return None

    meta = {}
    if 'meta' in ckpt:
        try:
            meta = ckpt['meta'].item()
        except Exception:
            meta = {}
    obs_dim = int(ckpt.get('obs_dim', meta.get('obs_dim', 0)))
    act_dim = int(ckpt.get('act_dim', meta.get('act_dim', 0)))

    # Try to find normalization
    obs_mean = (ckpt.get('obs_mean', None) or meta.get('obs_mean', None))
    obs_std = (ckpt.get('obs_std', None) or meta.get('obs_std', None) or meta.get('obs_scale', None))
    if obs_mean is not None: obs_mean = np.array(obs_mean, dtype=np.float32)
    if obs_std is not None: obs_std = np.array(obs_std, dtype=np.float32)

    W = ckpt.get('W', None)
    b = ckpt.get('b', None)
    theta = ckpt.get('theta', None)

    if W is not None:
        W = np.array(W)
        if b is None:
            if 'bias' in ckpt:
                b = np.array(ckpt['bias'])
            else:
                b = np.zeros(W.shape[0], dtype=np.float32)
        if act_dim and W.shape[0] != act_dim:
            print("ℹ️  Policy W act_dim mismatch; proceeding anyway.")
        if obs_dim and W.shape[1] != obs_dim:
            print("ℹ️  Policy W obs_dim mismatch; proceeding anyway.")
    elif theta is not None:
        th = np.array(theta)
        if th.ndim == 2:
            W = th
            b = np.zeros(W.shape[0], dtype=np.float32)
        elif th.ndim == 1 and obs_dim and act_dim and th.size == act_dim * obs_dim + act_dim:
            W = th[:act_dim * obs_dim].reshape(act_dim, obs_dim)
            b = th[act_dim * obs_dim:]
        else:
            print("✗ Unrecognized theta format")
            return None
    else:
        print("✗ No recognizable policy keys (W/b or theta)")
        return None

    # Build policy function
    def policy_fn(obs, ctrl_low=None, ctrl_high=None):
        x = obs
        if obs_mean is not None and obs_std is not None:
            x = (x - obs_mean) / (obs_std + 1e-8)
        u = W @ x + b
        if ctrl_low is not None and ctrl_high is not None:
            u = np.clip(u, ctrl_low, ctrl_high)
        return u.astype(np.float32)

    print(f"✓ Loaded linear policy: W {W.shape}, b {b.shape}, obs_dim={obs_dim}, act_dim={act_dim}")
    return policy_fn

def build_pd_controller(model):
    # Simple PD stand: hips -> 0 rad, knees -> ~0.6 rad
    hip_targets = {}
    knee_targets = {}
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if 'hip' in name:
            hip_targets[j] = 0.0
        if 'knee' in name:
            knee_targets[j] = 0.6

    Kp = 40.0
    Kd = 1.5

    def pd_policy(model, data, ctrl_low=None, ctrl_high=None):
        u = np.zeros(model.nu, dtype=np.float32)
        for i in range(model.nu):
            # Map actuator -> joint
            j_id = model.actuator_trnid[i, 0]
            if j_id < 0:
                continue
            qadr = model.jnt_qposadr[j_id]
            dadr = model.jnt_dofadr[j_id]
            q = float(data.qpos[qadr])
            dq = float(data.qvel[dadr])
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or ""

            if j_id in knee_targets:
                q_ref = knee_targets[j_id]
            elif j_id in hip_targets:
                q_ref = hip_targets[j_id]
            else:
                q_ref = 0.0

            tau = Kp * (q_ref - q) - Kd * dq
            gear = float(model.actuator_gear[i, 0]) if model.actuator_gear.shape[1] > 0 else 1.0
            ctrl = tau / max(gear, 1e-6)
            if ctrl_low is not None and ctrl_high is not None:
                ctrl = np.clip(ctrl, ctrl_low[i], ctrl_high[i])
            u[i] = ctrl
        return u
    print("ℹ️  Using PD standing controller (policy not provided or not recognized)")
    return pd_policy

def run(args):
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    ctrl_low = model.actuator_ctrlrange[:, 0].copy()
    ctrl_high = model.actuator_ctrlrange[:, 1].copy()

    policy_fn = None
    if args.policy and os.path.exists(args.policy):
        policy_fn = try_load_linear_policy(args.policy)

    if policy_fn is None:
        pd_fn = build_pd_controller(model)
        def policy_apply(obs):
            return pd_fn(model, data, ctrl_low, ctrl_high)
    else:
        def policy_apply(obs):
            return policy_fn(obs, ctrl_low, ctrl_high)

    # Reset pose: small crouch init to avoid instant fall
    mujoco.mj_resetData(model, data)
    # Small knee bend
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if 'knee' in name:
            adr = model.jnt_qposadr[j]
            data.qpos[adr] = 0.4

    sim_dt = model.opt.timestep
    slow = max(args.slow, 1e-3)
    steps = int(args.duration / sim_dt)

    # Try interactive viewer if DISPLAY available and glfw works
    use_viewer = (os.environ.get("DISPLAY") is not None and not args.headless and args.video is None)

    if use_viewer:
        try:
            with mujoco.viewer.launch_passive(model, data) as v:
                v.cam.lookat[:] = [0.0, 0.0, 0.5]
                print("Viewer running. Press ESC to quit.")
                for _ in range(steps):
                    if not v.is_running():
                        break
                    obs = get_obs(model, data)
                    u = policy_apply(obs)
                    data.ctrl[:] = u
                    mujoco.mj_step(model, data)
                    v.sync()
                    time.sleep(sim_dt * slow)
            return
        except Exception as e:
            print(f"Viewer unavailable ({e}), falling back to offscreen.")
            use_viewer = False

    # Offscreen render (EGL/OSMesa)
    H, W = args.height, args.width
    renderer = mujoco.Renderer(model, H, W)
    # Optional video writer
    writer = None
    if args.video is not None:
        try:
            import imageio
            writer = imageio.get_writer(args.video, fps=int(1.0/(sim_dt*slow)))
            print(f"Recording to {args.video} ...")
        except Exception as e:
            print(f"✗ imageio not available for video writing: {e}")
            writer = None

    for _ in range(steps):
        obs = get_obs(model, data)
        u = policy_apply(obs)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=args.camera)
        frame = renderer.render()
        if writer is not None:
            writer.append_data(frame)
        time.sleep(sim_dt * slow)

    if writer is not None:
        writer.close()
        print(f"✓ Saved video: {args.video}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--xml', type=str, default='quadruped.xml', help='MuJoCo model xml')
    p.add_argument('--policy', type=str, default=None, help='Path to npz policy (optional)')
    p.add_argument('--duration', type=float, default=20.0, help='Seconds to simulate')
    p.add_argument('--slow', type=float, default=1.0, help='Slowdown factor (>=1.0 is slower)')
    p.add_argument('--camera', type=str, default='track', help='Camera name for rendering')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--video', type=str, default=None, help='Output mp4 path (headless recording)')
    p.add_argument('--headless', action='store_true', help='Force headless offscreen')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)
