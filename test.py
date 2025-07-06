"""
Run a trained PPO policy in MuJoCo viewer at 60 FPS.
"""

import time, re, threading
from pathlib import Path

import mujoco
from mujoco import viewer
from pynput import keyboard

import numpy as np
import jax, jax.numpy as jnp
from flax.training import checkpoints

XML_PATH = "env_ping_pong.xml"
# CKPT_DIR = Path("jax_mujoco_2").absolute() / "checkpoints"
# CKPT_PREFIX = "policy_"            # adjust if you used a different prefix
DT_TARGET = 1.0 / 60.0             # 60 FPS

# ── 0. MuJoCo sizes ────────────────────────────────────────────────────────
mj_model = mujoco.MjModel.from_xml_path(XML_PATH)
nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu

# policy          = Policy(19, 6)
# params_template = policy                      # no .init needed

# ckpt_path = checkpoints.latest_checkpoint(CKPT_DIR, prefix=CKPT_PREFIX)
# if ckpt_path:
#     params = checkpoints.restore_checkpoint(ckpt_path, target=params_template)
#     step   = int(re.search(r"_([0-9]+)$", ckpt_path).group(1))
#     print(f"✓ loaded step {step} from {ckpt_path}")
# else:
#     print("[WARN] no checkpoint found; using random weights.")
#     params = params_template

# @jax.jit
# def act_fn(model, obs, key):
#     ctrls, stds  = model(obs[None, :], key)     # remove batch dim
#     return ctrls[0], stds[0]                   # deterministic mean action

# from jax import random
# key = random.PRNGKey(0)
# ── 4. Keys for manual overrides (optional) ────────────────────────────────
pressed_keys = set()
def on_press(key):
    try:    pressed_keys.add(key.char)
    except AttributeError: pressed_keys.add(str(key))
def on_release(key):
    try:    pressed_keys.discard(key.char)
    except AttributeError: pressed_keys.discard(str(key))
keyboard.Listener(on_press=on_press, on_release=on_release).start()

# ── 5. Actuator IDs, camera, keyframe reset ────────────────────────────────
aid_x = mj_model.actuator("motor_paddle_x").id
aid_y = mj_model.actuator("motor_paddle_y").id
aid_z = mj_model.actuator("motor_paddle_z").id
cam_id = mj_model.camera("fixed_cam").id
kf_id  = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "serve")

paddle_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_face")
ball_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
table_id   = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
print(ball_id)
print(table_id)

mj_data = mujoco.MjData(mj_model)
mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)
ctrl = jnp.array([0, 0, 0])
# ── 6. Viewer loop ─────────────────────────────────────────────────────────
episode_start = time.time()
with viewer.launch_passive(mj_model, mj_data) as v:
    while v.is_running():
        frame_start = time.time()

        # ---- roll environment until DT_TARGET elapses ----
        sim_t0 = mj_data.time
        while (mj_data.time - sim_t0) < DT_TARGET:
            # 1. observation
            # qpos = jnp.asarray(np.copy(mj_data.qpos), dtype=jnp.float32)
            # qvel = jnp.asarray(np.copy(mj_data.qvel), dtype=jnp.float32)
            # obs  = jnp.concatenate([qpos, qvel])

            # # 2. policy → control
            # key, subkey = random.split(key)
            # subkeys = random.split(subkey, 1)
            # ctrls, stds = params.inference((obs[None, :]))
            # ctrls, stds = ctrls[0], stds[0]

            # dist_paddle_ball = jnp.exp(-2*jnp.linalg.norm(jnp.array([0, 0, 2]) - jnp.array(mj_data.qpos[7:])))
            # # print(dist_paddle_ball)
            
            # ncon = mj_data.ncon                          

            # c = mj_data.contact
            # # print(c[:ncon].dist)
            # # print(c.dist)

            # geom1 = c.geom1[:ncon] 
            # geom2 = c.geom2[:ncon]  
            # # print(geom1)         
            # # print(geom2)             

            # mask_pb = (geom1 == paddle_id) & (geom2 == ball_id) | (geom1 == ball_id)   & (geom2 == paddle_id)
            # hit_pb = jnp.any(mask_pb) 
            # mask_tb = (geom1 == table_id) & (geom2 == ball_id) | (geom1 == ball_id)   & (geom2 == table_id)
            # hit_tb = jnp.any(mask_tb) 
            # if (hit_pb):
            #     print(hit_pb)

            
            # mj_data.ctrl[:] = np.asarray(ctrls, dtype=np.float64)
            # d = jnp.linalg.norm(mj_data.qpos[7:] - mj_data.qpos[:3])
            # if d < .06:
            #     print(d)
            # # if(mj_data.ncon == 1):
            # #     print("contact")
            # #     print(mj_data.contact)
            # #     print("\n\n")
            # print("control")
            # print(mj_data.ctrl)
            # print("std")
            # print(stds)



            # 3. physics
            mujoco.mj_step(mj_model, mj_data)

        # ---- render frame ----
        v.sync()

        # ---- real‑time pacing ----
        sleep_t = DT_TARGET - (time.time() - frame_start)
        if sleep_t > 0:
            time.sleep(sleep_t)

        # ---- auto‑reset ---------------------------------------------------
        # if (mj_data.qpos[2] < 0.5) or (time.time() - episode_start > 5):
        #     mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)
        #     episode_start = time.time()
        if (time.time() - episode_start > 4):
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, kf_id)
            episode_start = time.time()
