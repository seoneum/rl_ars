# ==============================================================================
# 0. JAX 환경 설정 (가장 먼저 실행)
# ==============================================================================
# float64(배정밀도)를 비활성화하여 GPU에서 float32로 연산 속도 향상
from jax import config
config.update("jax_enable_x64", False)

#!/usr/bin/env python3
import os
import argparse
import time
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
    """학습 상태를 안전하게 저장합니다 (임시 파일 및 디렉토리 생성)."""
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
        ckpt = {
            "theta": jnp.array(d["theta"]),
            "obs_dim": int(d["obs_dim"]),
            "act_dim": int(d["act_dim"]),
            "iter": int(d["iter"]) if "iter" in d.files else 0,
            "key": jnp.array(d["key"]) if "key" in d.files else None,
            "meta": d["meta"].item() if "meta" in d.files else {},
        }
        return ckpt
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {path}. Error: {e}")
        return None

# ==============================================================================
# 1. MJX 기반 병렬 시뮬레이션 환경 생성
# ==============================================================================
def make_mjx_env(xml_path: str,
                 action_repeat: int = 4,
                 z_threshold: float = 0.45,
                 ctrl_cost_weight: float = 1e-3,
                 tilt_penalty_weight: float = 1e-3,
                 forward_axis: str = "x",
                 forward_sign: int = -1,
                 target_speed: float = 0.1,
                 overspeed_weight: float = 0.5,
                 stand_bonus: float = 0.1,
                 stand_shape_weight: float = 0.3,
                 # [수정 4] stand_gate 임계값을 인자로 받도록 추가
                 stand_gate_z: float = 0.27,
                 stand_gate_up: float = 0.80
                 ):
    m = mujoco.MjModel.from_xml_path(xml_path)
    mm = mjx.put_model(m)

    ctrl_low = jnp.array(m.actuator_ctrlrange[:, 0], dtype=jnp.float32)
    ctrl_high = jnp.array(m.actuator_ctrlrange[:, 1], dtype=jnp.float32)
    ctrl_center = (ctrl_low + ctrl_high) / 2.0
    ctrl_half = (ctrl_high - ctrl_low) / 2.0

    def scale_action(a_unit: jnp.ndarray) -> jnp.ndarray:
        """액션 값을 [-1, 1]에서 실제 컨트롤 범위로 스케일링합니다."""
        return ctrl_center + ctrl_half * jnp.tanh(a_unit)

    def obs_from_dd(dd: Any) -> jnp.ndarray:
        """MJX 데이터에서 관측 벡터를 추출합니다."""
        return jnp.concatenate([dd.qpos[7:], dd.qvel], axis=0).astype(jnp.float32)

    qpos0 = jnp.array(m.qpos0, dtype=jnp.float32)
    zero_qvel = jnp.zeros((m.nv,), dtype=jnp.float32)
    zero_ctrl = jnp.zeros((m.nu,), dtype=jnp.float32)

    axis2idx = {"x": 0, "y": 1, "z": 2}
    fwd_idx = axis2idx[forward_axis]
    fwd_sign = jnp.float32(forward_sign)
    tgt_speed = jnp.float32(target_speed)
    over_w = jnp.float32(overspeed_weight)

    def reset_single(key: jax.Array) -> Tuple[Tuple[Any, ...], jnp.ndarray]:
        """단일 환경을 리셋합니다."""
        dd = mjx.make_data(mm)
        noise = 0.01 * random.normal(key, (qpos0.shape[0],), dtype=jnp.float32)
        qpos = qpos0 + noise.at[:7].set(0.0)
        dd = replace(dd, qpos=qpos, qvel=zero_qvel, ctrl=zero_ctrl)
        obs = obs_from_dd(dd)
        return (dd, obs, jnp.array(False), zero_ctrl), obs

    def upright_from_quat(q: jnp.ndarray) -> jnp.ndarray:
        """쿼터니언으로부터 몸체가 얼마나 수직으로 서 있는지를 계산합니다."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        return 1.0 - 2.0 * (x*x + y*y)

    def step_single(state, a_unit: jnp.ndarray):
        """단일 환경의 한 스텝을 진행하고 보상과 종료 여부를 계산합니다."""
        dd, _, done_prev, _ = state
        ctrl = scale_action(a_unit)

        def integrate_if_alive(dd_in):
            def body_fn(carry, _):
                return mjx.step(mm, replace(carry, ctrl=ctrl)), None
            dd_out, _ = lax.scan(body_fn, dd_in, xs=None, length=action_repeat)
            return dd_out
        
        dd = lax.cond(done_prev, lambda d: d, integrate_if_alive, dd)

        obs = obs_from_dd(dd)
        
        forward_speed = (dd.qvel[fwd_idx] * fwd_sign).astype(jnp.float32)
        base_z = dd.qpos[2].astype(jnp.float32)
        up = jnp.clip(upright_from_quat(dd.qpos[3:7]), 0.0, 1.0)

        # [수정 4] 하드코딩된 값을 인자로 받은 stand_gate_z, stand_gate_up으로 교체
        stand_gate = jnp.logical_and(base_z > stand_gate_z, up > stand_gate_up)

        reward_forward = jnp.minimum(forward_speed, tgt_speed)
        overspeed = jnp.maximum(forward_speed - tgt_speed, 0.0)
        overspeed_pen = over_w * (overspeed ** 2)
        speed_term = jnp.where(stand_gate, reward_forward - overspeed_pen, 0.0)

        height_term = jnp.clip((base_z - 0.25) / 0.07, 0.0, 1.0)
        upright_term = jnp.clip((up - 0.6) / 0.2, 0.0, 1.0)
        stand_shape = 0.5 * (height_term + upright_term)

        ctrl_cost = ctrl_cost_weight * jnp.sum(jnp.square(ctrl))
        tilt_pen = tilt_penalty_weight * (1.0 - up)

        reward = (
            stand_bonus
            + stand_shape_weight * stand_shape
            + speed_term
            - tilt_pen
            - ctrl_cost
        )

        isnan = jnp.any(jnp.isnan(dd.qpos)) | jnp.any(jnp.isnan(dd.qvel))
        done_now = jnp.logical_or(
            isnan,
            jnp.logical_or(base_z < z_threshold, up < 0.2)
        )
        done = jnp.logical_or(done_prev, done_now)

        new_state = (dd, obs, done, ctrl)
        return new_state, (obs, reward, done)

    reset_batch = jax.jit(jax.vmap(reset_single))
    step_batch = jax.jit(jax.vmap(step_single))

    env_info = { "obs_dim": int(m.nq - 7 + m.nv), "act_dim": int(m.nu) }
    return reset_batch, step_batch, env_info

# ==============================================================================
# 2. 정책 정의 및 ARS 학습 루프
# ==============================================================================
def make_policy_fns(obs_dim: int, act_dim: int) -> Tuple[Callable, Callable]:
    """선형 정책 함수를 생성합니다."""
    params_pytree_shape = (jnp.zeros((obs_dim, act_dim)), jnp.zeros((act_dim,)))
    def ravel_pytree_partial(pytree): return ravel_pytree(pytree)[0]
    _, unravel_fn = ravel_pytree(params_pytree_shape)
    @jax.jit
    def policy_apply_fn(theta: jnp.ndarray, obs: jnp.ndarray) -> jnp.ndarray:
        W, b = unravel_fn(theta)
        return jnp.tanh(obs @ W + b)
    return ravel_pytree_partial, policy_apply_fn

def ars_train(xml_path: str, **kwargs):
    """ARS 알고리즘으로 정책을 학습합니다."""
    seed, num_envs, episode_length = kwargs['seed'], kwargs['num_envs'], kwargs['episode_length']
    
    steps_per_iter = (2 * kwargs["num_dirs"] * kwargs["num_envs"] * kwargs["episode_length"] * kwargs["action_repeat"])
    print(f"[Config] steps/iter ≈ {steps_per_iter:,}  | dir_chunk={kwargs.get('dir_chunk')}")

    # [수정 4] make_mjx_env에 전달할 인자 목록에 stand_gate_z, stand_gate_up 추가
    env_params_keys = ['action_repeat', 'z_threshold', 'ctrl_cost_weight', 
                       'tilt_penalty_weight', 'forward_axis', 'forward_sign', 
                       'target_speed', 'overspeed_weight', 'stand_bonus', 
                       'stand_shape_weight', 'stand_gate_z', 'stand_gate_up']
    reset_batch, step_batch, info = make_mjx_env(
        xml_path=xml_path, 
        **{k: kwargs[k] for k in env_params_keys}
    )
    obs_dim, act_dim = info["obs_dim"], info["act_dim"]
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
                raise ValueError(f"Checkpoint dim mismatch: ckpt({ckpt['obs_dim']},{ckpt['act_dim']}) vs model({obs_dim},{act_dim})")
            theta, start_it = ckpt["theta"], ckpt["iter"]
            if ckpt["key"] is not None: key = ckpt["key"]
            print(f"[Resume] Loaded checkpoint from {kwargs['save_path']} at iter {start_it}")

    @jax.jit
    def rollout_return(theta_: jnp.ndarray, keys: jnp.ndarray) -> jnp.ndarray:
        """주어진 정책 파라미터로 롤아웃을 실행하고 누적 보상을 반환합니다."""
        state, obs = reset_batch(keys)
        def body(carry, _):
            st, ob, ret = carry
            act = policy_apply(theta_, ob)
            st, (ob_next, r, done) = step_batch(st, act)
            return (st, ob_next, ret + r * (1.0 - done)), None
        # [개선 2] XLA의 안정적인 연산을 위해 lax.scan 초기값에 dtype 명시
        (_, _, returns), _ = lax.scan(body, (state, obs, jnp.zeros(num_envs, dtype=jnp.float32)), None, length=episode_length)
        return returns

    def make_eval_chunk_fn(rollout_return_fn, noise_std):
        """메모리 관리를 위해 델타(탐색 방향)를 청크 단위로 평가하는 함수를 생성합니다."""
        @jax.jit
        def eval_chunk(theta_, deltas_chunk, keys_chunk):
            def eval_one(delta, keys):
                r_plus = rollout_return_fn(theta_ + noise_std * delta, keys).mean()
                r_minus = rollout_return_fn(theta_ - noise_std * delta, keys).mean()
                return r_plus, r_minus
            return jax.vmap(eval_one)(deltas_chunk, keys_chunk)
        return eval_chunk

    eval_chunk = make_eval_chunk_fn(rollout_return, kwargs["noise_std"])

    total_new_iters = kwargs["iterations"]
    ckpt_every = kwargs["ckpt_every"] or kwargs["eval_every"]
    pbar = trange(total_new_iters, desc=f"ARS training (start iter: {start_it})")
    for it_local in pbar:
        global_it = start_it + it_local
        key, k_delta, k_dirs = random.split(key, 3)
        deltas = random.normal(k_delta, (kwargs["num_dirs"], theta.shape[0]))
        base_keys = random.split(k_dirs, kwargs["num_dirs"] * num_envs).reshape(kwargs["num_dirs"], num_envs, 2)
        
        dir_chunk = kwargs.get("dir_chunk")
        if dir_chunk:
            R_plus_list, R_minus_list = [], []
            for s in range(0, kwargs["num_dirs"], dir_chunk):
                e = min(s + dir_chunk, kwargs["num_dirs"])
                r_p, r_m = eval_chunk(theta, deltas[s:e], base_keys[s:e])
                R_plus_list.append(r_p); R_minus_list.append(r_m)
            R_plus = jnp.concatenate(R_plus_list, axis=0)
            R_minus = jnp.concatenate(R_minus_list, axis=0)
        else:
            R_plus, R_minus = eval_chunk(theta, deltas, base_keys)
        
        scores = jnp.maximum(R_plus, R_minus)
        topk_idx = jnp.argsort(scores)[-kwargs["top_dirs"]:]
        R_plus_top, R_minus_top, deltas_top = R_plus[topk_idx], R_minus[topk_idx], deltas[topk_idx]
        reward_std = jnp.std(jnp.concatenate([R_plus_top, R_minus_top])) + 1e-8
        weighted = (R_plus_top - R_minus_top)[:, None] * deltas_top
        grad_est = jnp.mean(weighted, axis=0) / reward_std
        theta += kwargs["step_size"] * grad_est
        
        # [수정 2] tqdm 진행률 표시줄에 JAX 배열 포맷팅 오류 수정 (float()으로 감싸기)
        pbar.set_postfix({
            "mean+": f"{float(jnp.mean(R_plus)):.2f}",
            "mean-": f"{float(jnp.mean(R_minus)):.2f}",
            "best": f"{float(jnp.max(scores)):.2f}",
        })

        if (global_it + 1) % kwargs["eval_every"] == 0 or it_local == total_new_iters - 1:
            key, k_eval = random.split(key)
            eval_keys = random.split(k_eval, num_envs)
            ret = rollout_return(theta, eval_keys).mean()
            print(f"\n[Iter {global_it+1}] Eval Return: {float(ret):.2f}")
        
        if (global_it + 1) % ckpt_every == 0 or it_local == total_new_iters - 1:
            # [수정 4] 체크포인트 메타데이터에 새 인자 추가
            meta_keys = ["xml_path", "forward_axis", "forward_sign", "target_speed", 
                         "overspeed_weight", "stand_bonus", "stand_shape_weight",
                         "stand_gate_z", "stand_gate_up"]
            save_checkpoint(
                kwargs["save_path"], theta=theta, key=key, it=global_it + 1,
                obs_dim=obs_dim, act_dim=act_dim, 
                meta={k: kwargs.get(k) for k in meta_keys},
            )

# ==============================================================================
# 3. 추론
# ==============================================================================
def run_inference(xml_path: str, ckpt_path: str, **kwargs):
    """학습된 정책으로 추론을 실행합니다."""
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None: raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    theta, obs_dim, act_dim = ckpt["theta"], ckpt["obs_dim"], ckpt["act_dim"]
    _, policy_apply = make_policy_fns(obs_dim, act_dim)
    
    # [수정 4] make_mjx_env에 전달할 인자 목록에 stand_gate_z, stand_gate_up 추가
    env_params_keys = ['action_repeat', 'z_threshold', 'ctrl_cost_weight', 
                       'tilt_penalty_weight', 'forward_axis', 'forward_sign', 
                       'target_speed', 'overspeed_weight', 'stand_bonus', 
                       'stand_shape_weight', 'stand_gate_z', 'stand_gate_up']
    env_params = {k: kwargs[k] for k in env_params_keys}
    reset_batch, step_batch, _ = make_mjx_env(xml_path=xml_path, **env_params)
    
    keys = random.split(random.PRNGKey(kwargs['seed']), kwargs['num_envs'])
    state, obs = reset_batch(keys)
    def body(carry, _):
        st, ob, ret = carry
        act = policy_apply(theta, ob)
        st, (ob_next, r, done) = step_batch(st, act)
        return (st, ob_next, ret + r * (1.0 - done)), None
    # [개선 2] 추론 시에도 lax.scan 초기값에 dtype 명시
    (_, _, returns), _ = lax.scan(body, (state, obs, jnp.zeros(kwargs['num_envs'], dtype=jnp.float32)), None, length=kwargs['episode_length'])
    print(f"Inference mean return: {float(jnp.mean(returns)):.2f}")


# ==============================================================================
# 4. 스크립트 실행 메인 블록
# ==============================================================================
def parse_args():
    """스크립트 실행을 위한 인자들을 파싱합니다."""
    p = argparse.ArgumentParser(description="Headless-only MJX + ARS trainer for quadruped")
    p.add_argument("--xml", type=str, required=True, dest="xml_path")
    p.add_argument("--infer", action="store_true", help="추론 모드로 실행")
    p.add_argument("--resume", action="store_true", help="저장된 체크포인트에서 학습 재개")
    p.add_argument("--seed", type=int, default=0)
    # 학습 파라미터
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--num-dirs", type=int, default=8)
    p.add_argument("--top-dirs", type=int, default=4)
    p.add_argument("--episode-length", type=int, default=300)
    p.add_argument("--action-repeat", type=int, default=2)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--dir-chunk", type=int, default=16, help="num_dirs를 나눠 평가할 chunk 크기")
    p.add_argument("--step-size", type=float, default=0.015)
    p.add_argument("--noise-std", type=float, default=0.025)
    # 환경 및 보상 파라미터
    p.add_argument("--z-threshold", type=float, default=0.35, help="에피소드 종료 높이 임계값")
    p.add_argument("--ctrl-cost-weight", type=float, default=1e-4)
    p.add_argument("--tilt-penalty-weight", type=float, default=1e-3, help="몸체 기울기 패널티 가중치")
    p.add_argument("--forward-axis", choices=["x", "y", "z"], default="x")
    # [개선 1] forward_sign 기본값을 -1 (후진)에서 1 (전진)으로 변경
    p.add_argument("--forward-sign", type=int, choices=[-1, 1], default=-1)
    p.add_argument("--target-speed", type=float, default=0.5)
    p.add_argument("--overspeed-weight", type=float, default=1.0)
    p.add_argument("--stand-bonus", type=float, default=0.05, help="서 있을 때 받는 기본 보너스")
    p.add_argument("--stand-shape-weight", type=float, default=0.3, help="일어서기 자세 유도 보상 가중치")
    # [수정 4] stand_gate 임계값을 인자로 추가
    p.add_argument("--stand-gate-z", type=float, default=0.27, help="속도 보상 활성화 z높이 임계값")
    p.add_argument("--stand-gate-up", type=float, default=0.80, help="속도 보상 활성화 수직 임계값")
    # 로깅 및 저장
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--save-path", type=str, default="ars_policy.npz")
    
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # [수정 3] jax import 이후에 호출되어 효과가 없는 환경 변수 설정 제거 (env.sh로 관리 권장)
    # os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    # [수정 1] run_inference 호출 시 중복 인자 전달 버그 수정
    if args.infer:
        kwargs = vars(args).copy()
        xml = kwargs.pop("xml_path")
        ckpt = kwargs.pop("save_path")
        kwargs.pop("infer", None)  # 사용하지 않는 플래그 제거
        kwargs.pop("resume", None) # 추론에 불필요한 플래그 제거
        run_inference(xml_path=xml, ckpt_path=ckpt, **kwargs)
    else:
        ars_train(**vars(args))
