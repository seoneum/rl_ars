#!/usr/bin/env python3
# ==============================================================================
# 0. JAX 환경 설정 (가장 먼저 실행)
# ==============================================================================
from jax import config
config.update("jax_enable_x64", False)

import os
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
# 체크포인트 유틸리티 함수
# ==============================================================================
def save_checkpoint(path, theta, key, it, obs_dim, act_dim, meta=None, compressed=False):
    """학습 상태(정책 파라미터, 키, 이터레이션 등)를 안전하게 저장합니다."""
    dirpath = os.path.dirname(os.path.abspath(path))
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_path = path + ".tmp"
    saver = np.savez_compressed if compressed else np.savez
    with open(tmp_path, "wb") as f:
        saver(
            f,
            theta=np.asarray(theta, dtype=np.float32),
            key=np.asarray(key, dtype=np.uint32),
            iter=np.int64(it),
            obs_dim=np.int32(obs_dim),
            act_dim=np.int32(act_dim),
            meta=np.asarray(meta or {}, dtype=object),
            version=np.int32(1),
        )
    os.replace(tmp_path, path)

def load_checkpoint(path):
    """저장된 체크포인트를 불러옵니다."""
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=True)
        return {
            "theta": jnp.array(d["theta"]),
            "obs_dim": int(d["obs_dim"]),
            "act_dim": int(d["act_dim"]),
            "iter": int(d["iter"]) if "iter" in d.files else 0,
            "key": jnp.array(d["key"]) if "key" in d.files else None,
            "meta": d["meta"].item() if "meta" in d.files else {},
        }
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None

# ==============================================================================
# 1. MJX 기반 병렬 시뮬레이션 환경 생성
# ==============================================================================
def make_mjx_env(xml_path: str, **kwargs):
    """주어진 XML과 하이퍼파라미터로 MJX 시뮬레이션 환경을 구성합니다."""
    m = mujoco.MjModel.from_xml_path(xml_path)
    mm = mjx.put_model(m)

    nu, nv, nq = int(m.nu), int(m.nv), int(m.nq)

    ctrl_low = jnp.array(m.actuator_ctrlrange[:, 0], dtype=np.float32)
    ctrl_high = jnp.array(m.actuator_ctrlrange[:, 1], dtype=np.float32)
    ctrl_center = (ctrl_low + ctrl_high) / 2.0
    ctrl_half = (ctrl_high - ctrl_low) / 2.0

    # --- 무릎 관절 자동 탐색 및 정보 수집 ---
    knee_patterns = kwargs.get('knee_patterns', 'knee')
    pat_list = [s.strip() for s in knee_patterns.split(",") if s.strip()]
    knee_jids = []
    for jid in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        nm = nm if isinstance(nm, str) else (nm.decode() if nm else "")
        if any(p in nm for p in pat_list):
            if m.jnt_type[jid] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                knee_jids.append(jid)

    import numpy as _np
    knee_jids = _np.array(knee_jids, dtype=_np.int32)
    knee_qadr = jnp.array(m.jnt_qposadr[knee_jids], dtype=jnp.int32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.int32)
    knee_min = jnp.array(m.jnt_range[knee_jids, 0], dtype=jnp.float32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.float32)
    knee_max = jnp.array(m.jnt_range[knee_jids, 1], dtype=jnp.float32) if knee_jids.size > 0 else jnp.zeros((0,), dtype=jnp.float32)
    knee_span = jnp.maximum(knee_max - knee_min, 1e-6)

    # --- JIT 컴파일을 위한 파라미터 추출 및 캐스팅 ---
    action_repeat = int(kwargs.get('action_repeat', 3))
    z_th = jnp.float32(kwargs.get('z_threshold', 0.32))
    ctrl_w = jnp.float32(kwargs.get('ctrl_cost_weight', 2e-4))
    tilt_w = jnp.float32(kwargs.get('tilt_penalty_weight', 3e-2))
    fwd_idx = {"x": 0, "y": 1, "z": 2}[kwargs.get('forward_axis', 'x')]
    fsign = jnp.float32(kwargs.get('forward_sign', 1))
    tgt_speed = jnp.float32(kwargs.get('target_speed', 0.35))
    over_w = jnp.float32(kwargs.get('overspeed_weight', 1.2))
    sp_w = jnp.float32(kwargs.get('speed_weight', 0.0))
    stand_b = jnp.float32(kwargs.get('stand_bonus', 0.20))
    stand_w = jnp.float32(kwargs.get('stand_shape_weight', 1.20))
    z_lo = jnp.float32(kwargs.get('target_z_low', 0.50))
    z_hi = jnp.float32(kwargs.get('target_z_high', 0.60))
    up_min = jnp.float32(kwargs.get('upright_min', 0.75))
    ang_w = jnp.float32(kwargs.get('angvel_penalty_weight', 0.01))
    actdw = jnp.float32(kwargs.get('act_delta_weight', 1e-4))
    base_w = jnp.float32(kwargs.get('base_vel_penalty_weight', 0.02))
    st_w = jnp.float32(kwargs.get('streak_weight', 0.01))
    st_scale = jnp.float32(kwargs.get('streak_scale', 60.0))
    z_pen_w = jnp.float32(kwargs.get('z_over_penalty', 0.06))

    # 초기 자세 및 무릎 관련 파라미터
    crouch_r = jnp.float32(kwargs.get('crouch_init_ratio', 0.20))
    crouch_n = jnp.float32(kwargs.get('crouch_init_noise', 0.03))
    pitch0   = jnp.float32(kwargs.get('init_pitch', -0.12))
    knee_lo  = jnp.float32(kwargs.get('knee_band_low', 0.50))
    knee_hi  = jnp.float32(kwargs.get('knee_band_high', 0.70))
    knee_w   = jnp.float32(kwargs.get('knee_band_weight', 0.8))
    knee_pen = jnp.float32(kwargs.get('knee_over_penalty', 0.05))
    # [신규] 과굴곡 패널티 및 중심 타겟 파라미터
    knee_under_pen = jnp.float32(kwargs.get('knee_under_penalty', 0.18))
    kcenter = jnp.float32(kwargs.get('knee_center', 0.60))
    kcenter_w = jnp.float32(kwargs.get('knee_center_weight', 0.40))


    def scale_action(a_unit: jnp.ndarray) -> jnp.ndarray:
        return ctrl_center + ctrl_half * jnp.tanh(a_unit)

    def obs_from_dd(dd: Any) -> jnp.ndarray:
        return jnp.concatenate([dd.qpos[7:], dd.qvel], axis=0).astype(jnp.float32)

    qpos0 = jnp.array(m.qpos0, dtype=jnp.float32)
    zero_qvel = jnp.zeros((nv,), dtype=jnp.float32)
    zero_ctrl = jnp.zeros((nu,), dtype=jnp.float32)

    def reset_single(key: jax.Array) -> Tuple[Tuple[Any, ...], jnp.ndarray]:
        dd = mjx.make_data(mm)
        k1, k2 = random.split(key)
        noise_all = 0.01 * random.normal(k1, (qpos0.shape[0],), dtype=jnp.float32)
        qpos = qpos0 + noise_all.at[:7].set(0.0)

        if knee_qadr.shape[0] > 0:
            knee_noise = crouch_n * random.normal(k2, (knee_qadr.shape[0],), dtype=jnp.float32)
            knee_target = knee_min + crouch_r * knee_span + knee_noise * knee_span
            qpos = qpos.at[knee_qadr].set(knee_target)

        half = 0.5 * pitch0
        q_w, q_y = jnp.cos(half), jnp.sin(half)
        qpos = qpos.at[3:7].set(jnp.array([q_w, 0.0, q_y, 0.0], dtype=jnp.float32))

        dd = replace(dd, qpos=qpos, qvel=zero_qvel, ctrl=zero_ctrl)
        obs = obs_from_dd(dd)
        streak0 = jnp.array(0.0, dtype=jnp.float32)
        return (dd, obs, jnp.array(False, dtype=jnp.bool_), zero_ctrl, streak0), obs

    def upright_from_quat(q: jnp.ndarray) -> jnp.ndarray:
        return 1.0 - 2.0 * (q[1]**2 + q[2]**2)

    def sgate(x, t, k=30.0):
        return 1.0 / (1.0 + jnp.exp(-k * (x - t)))

    def band_parabola(x, lo, hi):
        span = jnp.maximum(hi - lo, 1e-6)
        xn = jnp.clip((x - lo) / span, 0.0, 1.0)
        return jnp.clip(1.0 - 4.0 * (xn - 0.5)**2, 0.0, 1.0)

    def between_gate(x, lo, hi, k=30.0):
        return sgate(x, lo, k) * sgate(hi, x, k)

    def step_single(state, a_unit: jnp.ndarray):
        dd, _, done_prev, prev_ctrl, streak = state
        ctrl = scale_action(a_unit)

        def integrate_if_alive(dd_in):
            def body_fn(carry, _):
                return mjx.step(mm, replace(carry, ctrl=ctrl)), None
            dd_out, _ = lax.scan(body_fn, dd_in, xs=None, length=action_repeat)
            return dd_out

        dd = lax.cond(done_prev, lambda d: d, integrate_if_alive, dd)
        obs = obs_from_dd(dd)

        base_z = dd.qpos[2].astype(jnp.float32)
        up = jnp.clip(upright_from_quat(dd.qpos[3:7]), 0.0, 1.0)
        forward_speed = (dd.qvel[fwd_idx] * fsign).astype(jnp.float32)

        z_band = band_parabola(base_z, z_lo, z_hi)
        up_band = band_parabola(up, up_min, 0.98)
        stand_shape = 0.5 * (z_band + up_band)
        stand_gate_soft = between_gate(base_z, z_lo, z_hi) * sgate(up, up_min)

        reward_forward = jnp.minimum(forward_speed, tgt_speed)
        overspeed = jnp.maximum(forward_speed - tgt_speed, 0.0)
        speed_term = sp_w * stand_gate_soft * (reward_forward - over_w * (overspeed**2))

        is_standing_now = (stand_gate_soft > 0.6) & (~done_prev)
        streak_next = jnp.where(is_standing_now, streak + 1.0, 0.0)
        streak_reward = st_w * jnp.tanh(streak_next / st_scale)

        # [수정] 무릎 보상/패널티 계산 (과굴곡 패널티 강화 및 중심 타겟 추가)
        if knee_qadr.shape[0] > 0:
            kq = dd.qpos[knee_qadr]
            kr = jnp.clip((kq - knee_min) / knee_span, 0.0, 1.0)
            kband = band_parabola(kr, knee_lo, knee_hi)
            knee_band_mean = jnp.mean(kband)
            under = jnp.clip(knee_lo - kr, 0.0, 1.0)
            over  = jnp.clip(kr - knee_hi, 0.0, 1.0)
            knee_out_pen = knee_under_pen * jnp.sum(under*under) + knee_pen * jnp.sum(over*over)
            kcenter_term = -kcenter_w * jnp.mean((kr - kcenter)**2)
        else:
            knee_band_mean = jnp.float32(0.0)
            knee_out_pen   = jnp.float32(0.0)
            kcenter_term   = jnp.float32(0.0)

        ctrl_cost = ctrl_w * jnp.sum(jnp.square(ctrl))
        tilt_pen = tilt_w * (1.0 - up)
        ang_pen = ang_w * jnp.sum(jnp.square(dd.qvel[3:5].astype(jnp.float32)))
        act_pen = actdw * jnp.sum(jnp.square(ctrl - prev_ctrl))

        mask_xy = jnp.array([1.0, 1.0], dtype=jnp.float32).at[fwd_idx].set(0.0)
        lin_xy = dd.qvel[0:2].astype(jnp.float32) * mask_xy
        base_ang = dd.qvel[3:5].astype(jnp.float32)
        base_pen = base_w * (jnp.sum(jnp.square(lin_xy)) + jnp.sum(jnp.square(base_ang)))

        under_z = jnp.clip(z_lo - base_z, 0.0, 1e9)
        over_z  = jnp.clip(base_z - z_hi, 0.0, 1e9)
        z_span = jnp.maximum(z_hi - z_lo, 1e-6)
        z_out_pen = (under_z / z_span)**2 + (over_z / z_span)**2

        # [수정] 최종 보상 결합
        reward = (stand_b + stand_w * stand_shape + streak_reward + speed_term
                  + knee_w * knee_band_mean + kcenter_term
                  - (tilt_pen + ang_pen + base_pen + act_pen + ctrl_cost + knee_out_pen + z_pen_w * z_out_pen))

        isnan = jnp.any(jnp.isnan(dd.qpos)) | jnp.any(jnp.isnan(dd.qvel))
        done_now = jnp.logical_or(isnan, jnp.logical_or(base_z < z_th, up < 0.2))
        done = jnp.logical_or(done_prev, done_now)

        new_state = (dd, obs, done, ctrl, streak_next)
        return new_state, (obs, reward, done)

    reset_batch = jax.jit(jax.vmap(reset_single))
    step_batch = jax.jit(jax.vmap(step_single))
    env_info = {
        "obs_dim": int(nq - 7 + nv), "act_dim": int(nu),
        "knee_qadr": knee_qadr, "knee_min": knee_min, "knee_span": knee_span
    }
    return reset_batch, step_batch, env_info

# ==============================================================================
# 2. 정책 정의 및 ARS 학습 루프
# ==============================================================================
def make_policy_fns(obs_dim: int, act_dim: int) -> Tuple[Callable, Callable]:
    params_pytree_shape = (jnp.zeros((obs_dim, act_dim)), jnp.zeros((act_dim,)))
    def ravel_pytree_partial(pytree): return ravel_pytree(pytree)[0]
    _, unravel_fn = ravel_pytree(params_pytree_shape)
    @jax.jit
    def policy_apply_fn(theta: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        W, b = unravel_fn(theta)
        return jnp.tanh(obs @ W + b)
    return ravel_pytree_partial, policy_apply_fn

def ars_train(xml_path: str, **kwargs):
    seed, num_envs, episode_length = kwargs['seed'], kwargs['num_envs'], kwargs['episode_length']
    steps_per_iter = (2 * kwargs["num_dirs"] * num_envs * episode_length * kwargs["action_repeat"])
    print(f"[Config] steps/iter ≈ {steps_per_iter:,}  | dir_chunk={kwargs.get('dir_chunk')}")

    env_params_keys = [
        'action_repeat','z_threshold','ctrl_cost_weight','tilt_penalty_weight',
        'forward_axis','forward_sign','target_speed','overspeed_weight',
        'stand_bonus','stand_shape_weight','speed_weight',
        'target_z_low','target_z_high','upright_min',
        'angvel_penalty_weight','act_delta_weight','base_vel_penalty_weight',
        'streak_weight','streak_scale','z_over_penalty',
        'knee_patterns','crouch_init_ratio','crouch_init_noise','init_pitch',
        'knee_band_low','knee_band_high','knee_band_weight','knee_over_penalty',
        'knee_under_penalty', 'knee_center', 'knee_center_weight', # [신규]
    ]
    reset_batch, step_batch, info = make_mjx_env(xml_path=xml_path, **{k: kwargs[k] for k in env_params_keys})
    obs_dim, act_dim = info["obs_dim"], info["act_dim"]
    knee_qadr, knee_min, knee_span = info["knee_qadr"], info["knee_min"], info["knee_span"]

    key = random.PRNGKey(seed)
    ravel, policy_apply = make_policy_fns(obs_dim, act_dim)
    key, kW = random.split(key)
    W0 = 0.01 * random.normal(kW, (obs_dim, act_dim))
    b0 = jnp.zeros((act_dim,))
    theta = ravel((W0, b0))

    start_it = 0
    if kwargs.get("resume", False):
        ckpt = load_checkpoint(kwargs["save_path"])
        if ckpt is not None:
            if ckpt["obs_dim"] != obs_dim or ckpt["act_dim"] != act_dim:
                raise ValueError("Checkpoint dim mismatch")
            theta, start_it = ckpt["theta"], ckpt["iter"]
            if ckpt["key"] is not None: key = ckpt["key"]
            print(f"[Resume] Loaded checkpoint at iter {start_it}")

    @jax.jit
    def rollout_return(theta_: jnp.ndarray, keys: jnp.ndarray) -> jnp.ndarray:
        state, obs = reset_batch(keys)
        def body(carry, _):
            st, ob, ret = carry
            act = policy_apply(theta_, ob)
            st, (ob_next, r, done) = step_batch(st, act)
            return (st, ob_next, ret + r * (1.0 - done)), None
        (_, _, returns), _ = lax.scan(body, (state, obs, jnp.zeros(num_envs, dtype=jnp.float32)), None, length=episode_length)
        return returns

    @jax.jit
    def eval_stats(theta_: jnp.ndarray, keys_: jnp.ndarray):
        state, obs = reset_batch(keys_)
        def body(carry, _):
            st, ob, ret, sum_up, sum_z, sum_knee, alive_steps = carry
            dd = st[0]
            qpos = dd.qpos
            z = qpos[:, 2].astype(jnp.float32)
            q = qpos[:, 3:7]
            up = jnp.clip(1.0 - 2.0 * (q[:, 1]**2 + q[:, 2]**2), 0.0, 1.0)
            
            knee_ratio_mean = jnp.float32(0.0)
            if knee_qadr.shape[0] > 0:
                kq = qpos[:, knee_qadr]
                kr = jnp.clip((kq - knee_min) / knee_span, 0.0, 1.0)
                knee_ratio_mean = jnp.mean(kr, axis=-1)

            act = policy_apply(theta_, ob)
            st2, (ob_next, r, done) = step_batch(st, act)
            is_alive = (1.0 - done)
            return (st2, ob_next, ret + r*is_alive, sum_up + up*is_alive, sum_z + z*is_alive, sum_knee + knee_ratio_mean*is_alive, alive_steps + is_alive), None

        init = (state, obs, *[jnp.zeros(num_envs, jnp.float32) for _ in range(5)])
        (_, _, returns, sum_up, sum_z, sum_knee, alive_steps), _ = lax.scan(body, init, None, length=episode_length)
        total_alive_steps = jnp.maximum(jnp.sum(alive_steps), 1.0)
        mean_up = jnp.sum(sum_up) / total_alive_steps
        mean_z  = jnp.sum(sum_z)  / total_alive_steps
        mean_knee = jnp.sum(sum_knee) / total_alive_steps
        return returns, mean_up, mean_z, mean_knee

    def make_eval_chunk_fn(rollout_return_fn, noise_std):
        @jax.jit
        def eval_chunk(theta_, deltas_chunk, keys_chunk):
            def eval_one(delta, keys):
                r_plus = rollout_return_fn(theta_ + noise_std * delta, keys).mean()
                r_minus = rollout_return_fn(theta_ - noise_std * delta, keys).mean()
                return r_plus, r_minus
            return jax.vmap(eval_one)(deltas_chunk, keys_chunk)
        return eval_chunk

    eval_chunk = make_eval_chunk_fn(rollout_return, kwargs["noise_std"])

    pbar = trange(kwargs["iterations"], desc=f"ARS training (start iter: {start_it})")
    for it_local in pbar:
        global_it = start_it + it_local
        key, k_delta, k_dirs = random.split(key, 3)
        deltas = random.normal(k_delta, (kwargs["num_dirs"], theta.shape[0]))
        base_keys = random.split(k_dirs, kwargs["num_dirs"] * num_envs).reshape(kwargs["num_dirs"], num_envs, 2)

        dir_chunk = kwargs.get("dir_chunk")
        if dir_chunk and dir_chunk < kwargs["num_dirs"]:
            R_plus_list, R_minus_list = [], []
            for s in range(0, kwargs["num_dirs"], dir_chunk):
                e = min(s + dir_chunk, kwargs["num_dirs"])
                r_p, r_m = eval_chunk(theta, deltas[s:e], base_keys[s:e])
                R_plus_list.append(r_p); R_minus_list.append(r_m)
            R_plus, R_minus = jnp.concatenate(R_plus_list), jnp.concatenate(R_minus_list)
        else:
            R_plus, R_minus = eval_chunk(theta, deltas, base_keys)

        scores = jnp.maximum(R_plus, R_minus)
        topk_idx = jnp.argsort(scores)[-kwargs["top_dirs"]:]
        R_plus_top, R_minus_top, deltas_top = R_plus[topk_idx], R_minus[topk_idx], deltas[topk_idx]
        reward_std = jnp.std(jnp.concatenate([R_plus_top, R_minus_top])) + 1e-8
        grad_est = jnp.mean((R_plus_top - R_minus_top)[:, None] * deltas_top, axis=0) / reward_std
        theta += kwargs["step_size"] * grad_est

        pbar.set_postfix({"mean+": f"{R_plus.mean():.2f}", "mean-": f"{R_minus.mean():.2f}", "best": f"{scores.max():.2f}"})

        if (global_it + 1) % kwargs["eval_every"] == 0 or it_local == kwargs["iterations"] - 1:
            key, k_eval = random.split(key)
            eval_keys = random.split(k_eval, num_envs)
            returns, mean_up, mean_z, mean_knee = eval_stats(theta, eval_keys)
            print(f"\n[Iter {global_it+1}] Eval Return: {returns.mean():.2f} | up: {mean_up:.3f} | z: {mean_z:.3f} | knee: {mean_knee:.3f}")

        if (global_it + 1) % kwargs["ckpt_every"] == 0 or it_local == kwargs["iterations"] - 1:
            save_checkpoint(kwargs["save_path"], theta, key, global_it + 1, obs_dim, act_dim, meta={k: kwargs.get(k) for k in env_params_keys + ["xml_path"]})

# ==============================================================================
# 3. 추론
# ==============================================================================
def run_inference(xml_path: str, ckpt_path: str, **kwargs):
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None: raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    theta, obs_dim, act_dim = ckpt["theta"], ckpt["obs_dim"], ckpt["act_dim"]
    _, policy_apply = make_policy_fns(obs_dim, act_dim)

    if ckpt.get("meta"):
        for k, v in ckpt["meta"].items(): kwargs.setdefault(k, v)

    env_params_keys = [
        'action_repeat','z_threshold','ctrl_cost_weight','tilt_penalty_weight',
        'forward_axis','forward_sign','target_speed','overspeed_weight',
        'stand_bonus','stand_shape_weight','speed_weight',
        'target_z_low','target_z_high','upright_min',
        'angvel_penalty_weight','act_delta_weight','base_vel_penalty_weight',
        'streak_weight','streak_scale','z_over_penalty',
        'knee_patterns','crouch_init_ratio','crouch_init_noise','init_pitch',
        'knee_band_low','knee_band_high','knee_band_weight','knee_over_penalty',
        'knee_under_penalty', 'knee_center', 'knee_center_weight', # [신규]
    ]
    reset_batch, step_batch, info = make_mjx_env(xml_path=xml_path, **{k: kwargs[k] for k in env_params_keys})
    knee_qadr, knee_min, knee_span = info["knee_qadr"], info["knee_min"], info["knee_span"]

    keys = random.split(random.PRNGKey(kwargs['seed']), kwargs['num_envs'])

    @jax.jit
    def eval_stats_once(theta_, keys_):
        state, obs = reset_batch(keys_)
        def body(carry, _):
            st, ob, ret, sum_up, sum_z, sum_knee, alive_steps = carry
            dd = st[0]
            qpos = dd.qpos
            z = qpos[:, 2].astype(jnp.float32)
            q = qpos[:, 3:7]
            up = jnp.clip(1.0 - 2.0 * (q[:, 1]**2 + q[:, 2]**2), 0.0, 1.0)
            
            knee_ratio_mean = jnp.float32(0.0)
            if knee_qadr.shape[0] > 0:
                kq = qpos[:, knee_qadr]
                kr = jnp.clip((kq - knee_min) / knee_span, 0.0, 1.0)
                knee_ratio_mean = jnp.mean(kr, axis=-1)

            act = policy_apply(theta_, ob)
            st2, (ob_next, r, done) = step_batch(st, act)
            is_alive = (1.0 - done)
            return (st2, ob_next, ret + r*is_alive, sum_up + up*is_alive, sum_z + z*is_alive, sum_knee + knee_ratio_mean*is_alive, alive_steps + is_alive), None

        init = (state, obs, *[jnp.zeros(kwargs['num_envs'], jnp.float32) for _ in range(5)])
        (_, _, returns, sum_up, sum_z, sum_knee, alive_steps), _ = lax.scan(body, init, None, length=kwargs['episode_length'])
        total_alive_steps = jnp.maximum(jnp.sum(alive_steps), 1.0)
        mean_up = jnp.sum(sum_up) / total_alive_steps
        mean_z  = jnp.sum(sum_z)  / total_alive_steps
        mean_knee = jnp.sum(sum_knee) / total_alive_steps
        return returns.mean(), mean_up, mean_z, mean_knee

    ret, up, z, knee = eval_stats_once(theta, keys)
    print(f"Inference mean return: {ret:.2f} | up: {up:.3f} | z: {z:.3f} | knee: {knee:.3f}")

# ==============================================================================
# 4. 스크립트 실행 메인 블록
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Headless-only MJX + ARS trainer for quadruped (crouch-stand first)")
    p.add_argument("--xml", type=str, required=True, dest="xml_path")
    p.add_argument("--infer", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-dirs", type=int, default=16)
    p.add_argument("--top-dirs", type=int, default=4)
    p.add_argument("--episode-length", type=int, default=250)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--dir-chunk", type=int, default=8)
    p.add_argument("--step-size", type=float, default=0.010)
    p.add_argument("--noise-std", type=float, default=0.015)
    p.add_argument("--z-threshold", type=float, default=0.32)
    p.add_argument("--ctrl-cost-weight", type=float, default=2e-4)
    p.add_argument("--tilt-penalty-weight", type=float, default=3e-2)
    p.add_argument("--stand-bonus", type=float, default=0.20)
    p.add_argument("--stand-shape-weight", type=float, default=1.20)
    p.add_argument("--speed-weight", type=float, default=0.0)
    p.add_argument("--target-z-low", type=float, default=0.50)
    p.add_argument("--target-z-high", type=float, default=0.60)
    p.add_argument("--upright-min", type=float, default=0.75)
    p.add_argument("--angvel-penalty-weight", type=float, default=0.01)
    p.add_argument("--act-delta-weight", type=float, default=1e-4)
    p.add_argument("--base-vel-penalty-weight", type=float, default=0.02)
    p.add_argument("--streak-weight", type=float, default=0.01)
    p.add_argument("--streak-scale", type=float, default=60.0)
    p.add_argument("--forward-axis", choices=["x", "y", "z"], default="x")
    p.add_argument("--forward-sign", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--target-speed", type=float, default=0.35)
    p.add_argument("--overspeed-weight", type=float, default=1.2)
    p.add_argument("--z-over-penalty", type=float, default=0.06, help="z 밴드(target_z_low/high) 밖 패널티 가중치")

    # --- 초기 앉기 및 무릎 밴드 제어 인자 ---
    p.add_argument("--knee-patterns", type=str, default="knee", help="무릎 관절 이름 부분문자열(쉼표 구분). 예: knee,kneer")
    p.add_argument("--crouch-init-ratio", type=float, default=0.20, help="무릎 신장 비율 초기값(0=최소각,1=최대각). 깊게 앉기면 0.15~0.25")
    p.add_argument("--crouch-init-noise", type=float, default=0.03, help="초기 무릎 각도에 더할 잡음(관절 span 비율)")
    p.add_argument("--init-pitch", type=float, default=-0.12, help="초기 몸통 pitch(rad, y축 회전). 앞쪽으로 약간 숙임")
    p.add_argument("--knee-band-low", type=float, default=0.50, help="무릎 신장 비율 하한(50%)")
    p.add_argument("--knee-band-high", type=float, default=0.70, help="무릎 신장 비율 상한(70%)")
    p.add_argument("--knee-band-weight", type=float, default=0.8, help="무릎 밴드 보상 가중치")
    p.add_argument("--knee-over-penalty", type=float, default=0.05, help="무릎 밴드 밖(과신전) 패널티 가중치")
    p.add_argument("--knee-under-penalty", type=float, default=0.18, help="무릎 밴드 하한(과굴곡) 패널티 가중치")
    p.add_argument("--knee-center", type=float, default=0.60, help="무릎 비율 타겟(밴드 중심)")
    p.add_argument("--knee-center-weight", type=float, default=0.40, help="타겟 중심으로 모으는 2차 셰이프 가중치")

    # --- 로깅 및 저장 ---
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--save-path", type=str, default="ars_policy.npz")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.infer:
        kwargs = vars(args).copy()
        xml = kwargs.pop("xml_path")
        ckpt = kwargs.pop("save_path")
        kwargs.pop("infer", None); kwargs.pop("resume", None)
        run_inference(xml_path=xml, ckpt_path=ckpt, **kwargs)
    else:
        ars_train(**vars(args))
