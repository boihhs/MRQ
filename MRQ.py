import jax.numpy as jnp, jax
from jax import random
from jax import lax
from jax.tree_util import register_pytree_node_class
from functools import partial
from Sim import Sim, SimCfg, SimData
import optax
from flax.training import checkpoints
from pathlib import Path
import math

# qpos is [ballxyz, ballquat, paddelxyz]
class MRQ:
    def __init__(self, encoder, policy, value_1, value_2, Sim, key=random.PRNGKey(0)):
        
        self.encoder = encoder
        self.value_1 = value_1
        self.value_2 = value_2
        self.policy = policy

        self.target_encoder = encoder
        self.target_value_1 = value_1
        self.target_value_2 = value_2
        self.target_policy = policy

        self.Sim = Sim
        self.key = key

        self.gamma = .9
        self.llambda = .99
        self.replay_buffer_capacity = 1000000

        self.zs_dim = 512
        self.za_dim = 256
        self.zsa_dim = 512

        self.horizon_enc = 5
        self.horizon_Q = 3


        # In your PPO __init__
        initial_lr = 3e-5

        self.steps = 3000

        self.epochs=200
        self.target_sync_every = 10
        self.minibatch_size=256
        self.updates_per_epoch=1
        self.num_bins = 21


        self.capacity = math.ceil(1_000_000 / (self.steps * self.Sim.cfg.batch))
        print(self.capacity)
        self.buffer = [None] * self.capacity

        
        self.policy_opt = optax.chain(
            optax.clip_by_global_norm(20),
            optax.adamw(learning_rate=3e-4)
        )
        self.value_1_opt = optax.chain(
            optax.adamw(learning_rate=3e-4)
        )
        self.value_2_opt = optax.chain(
            optax.adamw(learning_rate=3e-4)
        )
        self.encoder_opt = optax.chain(
            optax.adamw(learning_rate=1e-4, weight_decay=1e-4),
            
        )
        self.policy_opt_state = self.policy_opt.init(self.policy)
        self.value_1_opt_state = self.value_1_opt.init(self.value_1)
        self.value_2_opt_state = self.value_2_opt.init(self.value_2)
        self.encoder_opt_state = self.encoder_opt.init(self.encoder)
        
    @staticmethod
    @jax.jit
    def reward(prevState: SimData, nextState: SimData, ctrls: jnp.ndarray, prev_ctrls: jnp.ndarray):
        qpos = nextState.qpos
        qvel = nextState.qvel

        prev_qpos = prevState.qpos
        prev_qvel = prevState.qvel

        @partial(jax.vmap, in_axes=(0,0,0,0,0,0), out_axes=(0,0))
        def _reward_and_done(qpos: jnp.ndarray, qvel: jnp.ndarray, prev_qpos: jnp.ndarray, prev_qvel: jnp.ndarray, ctrl: jnp.ndarray, prev_ctrl: jnp.ndarray):

            PADDLE_CENTER = jnp.array([-1.5, 0.0, 1.0])
            SPHERE_RADIUS = 2.0
            k_sphere      = 0.05


            ball_pos        = qpos[:3]
            ball_vel = qvel[:3]
            prev_ball_pos   = prev_qpos[:3]
            paddle_pos      = qpos[7:10]
            prev_paddle_pos = prev_qpos[7:10]
            paddle_vel      = qvel[6:9]
            prev_paddle_vel = prev_qvel[6:9]
            reward = -jnp.linalg.norm(paddle_pos - jnp.array([0, 0, 2]))
            done = 0

            return reward, done



        # Call the vmapped function on the batch of states
        return _reward_and_done(qpos, qvel, prev_qpos, prev_qvel, ctrls, prev_ctrls)

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, encoder_model, policy_model, key):

        state = self.Sim.reset() # (B, SimData)

        noise_qpos = jnp.zeros(state.qpos.shape)
        noise_qvel = jnp.zeros(state.qvel.shape)

        key, subkey = random.split(key)

        noise = random.normal(subkey, (self.Sim.cfg.batch, 3)) * .1

        key, subkey = random.split(key)

        noise_vel = random.normal(subkey, (self.Sim.cfg.batch, 3))

        noise_qpos = noise_qpos.at[:, 7:].set(noise)
        noise_qvel = noise_qvel.at[:, :3].set(noise_vel)

        state = SimData(noise_qpos + state.qpos, noise_qvel + state.qvel) # (B, SimData)

        def _rollout(carry, _):
            state, key, prev_ctrl = carry

            s = self.Sim.getObs(state)   # (B, 19)       
            
            zs  = encoder_model.get_zs(s)
            action, _ = policy_model(zs)
            
            key, subkey = random.split(key)
            action = random.normal(subkey, action.shape) * .1 + action

            next_state = self.Sim.step(state, action)     # (B, SimData), (B, 3)
            r, done = self.reward(state, next_state, action, prev_ctrl)  # (B,), (B,)


            next_state = self.Sim.reset_partial(next_state, done)

            key, subkey = random.split(key)
            noise_qpos = random.normal(subkey, (self.Sim.cfg.batch, 3)) * .1
            noise_qpos = jnp.zeros_like(next_state.qpos).at[:, 7:].set(noise_qpos) * done[:, None]
            key, subkey = random.split(key)
            noise_qvel = random.normal(subkey, (self.Sim.cfg.batch, 3))
            noise_qvel = jnp.zeros_like(next_state.qvel).at[:, :3].set(noise_qvel) * done[:, None]
            next_state = SimData(noise_qpos + next_state.qpos, noise_qvel + next_state.qvel)

            return (next_state, key, action), (s, action, r, done, next_state)
        
        prev_ctrl = jnp.zeros((state.qpos.shape[0], 3))
        
        (state, key, _), (s, action, r, done, next_state) = jax.lax.scan(_rollout, (state, key, prev_ctrl), None, length=self.steps)
        # log_probs_old (T, B)
        # states (T, B, 19)
        # rewards (T, B)
        # V_theta_ts (T, B)
        # ctrls (T, B, 7)

        T, B = self.steps, self.Sim.cfg.batch
        
        # flat_s     = jax.lax.stop_gradient(s.reshape((T * B,)))
        # flat_action   = jax.lax.stop_gradient(action.reshape((T * B, -1)))
        # flat_r  = jax.lax.stop_gradient(r.reshape((T * B,)))
        # flat_done    = jax.lax.stop_gradient(done.reshape((T * B,)))
        # flat_next_s     = jax.lax.stop_gradient(next_state.reshape((T * B,)))

        Batch = {
            "states": jax.lax.stop_gradient(s),
            "actions": jax.lax.stop_gradient(action),
            "rewards": jax.lax.stop_gradient(r),
            "dones": jax.lax.stop_gradient(done),
            "next_states": jax.lax.stop_gradient(next_state),
        }

        return Batch, key
    
    @staticmethod
    @jax.jit
    def loss_fn_encoder(target_encoder, encoder, sup_episode, bins): # Work in (T, B, ..)
        
        states, actions, rewards, dones, next_states = sup_episode["states"], sup_episode["actions"], sup_episode["rewards"], sup_episode["dones"], sup_episode["next_states"]

        cumprod = jnp.cumprod(1.0 - dones, axis=0, exclusive=True)
        mask    = cumprod[..., None]

        T = states.shape[0]

        llamba_reward = .1
        llamba_dyamics = 1
        llamaba_terminal = .1
        state_0 = states[0]

        zs_0 = encoder.get_zs(state_0)

        def _get_loss_values(carry, xs):
            zs = carry
            action = xs

            _, d, r, zs_prime, _ = encoder(zs, action)

            return zs_prime, (zs_prime, r, d)
        
        zs, (zs_primes, rs, ds) = jax.lax.scan(_get_loss_values, zs_0, actions, length=T)

        # Reward Loss
        arange     = jnp.arange(-bins//2, bins//2)
        bin_values = jnp.abs(arange)/arange * (jnp.exp(jnp.abs(arange)) - 1)
        bin_values = jnp.nan_to_num(bin_values, nan=0.0)
        idxs = jnp.searchsorted(bin_values, rewards)
        idxs = jnp.clip(idxs, 1, bins-1)
        lower_idxs = idxs - 1 
        upper_idxs = idxs       

        lower_values = bin_values[lower_idxs]
        upper_values = bin_values[upper_idxs]
        alpha = (upper_values - rewards) / (upper_values - lower_values)
        beta = (rewards - lower_values) / (upper_values - lower_values)

        batch_idx      = jnp.arange(rs.shape[0]) 
        log_probs_lower = rs[batch_idx, lower_idxs]
        log_probs_upper = rs[batch_idx, upper_idxs]

        reward_loss = jnp.mean(-(log_probs_lower * alpha + log_probs_upper * beta) * mask[:-1, :, 0])

        # Dynamics Loss
        states_no_state_0 = states[1:]
        _, B, data_len = states_no_state_0.shape
        flat_states_no_state_0 = states_no_state_0.reshape((T-1)*B, -1)
        flat_zs_target = jax.lax.stop_gradient(target_encoder.get_zs(flat_states_no_state_0))
        zs_target = flat_zs_target.reshape(T-1, B, -1)

        dynamics_loss = jnp.mean((zs_target - zs_primes)**2 * mask[1:, :, 0, None])

        # Terminal Loss
        terminal_loss = jnp.mean((ds - dones)**2 * mask[:, :, 0])

        loss = reward_loss * llamba_reward + dynamics_loss * llamba_dyamics + terminal_loss * llamaba_terminal
        return loss
        

    @staticmethod
    @jax.jit
    def loss_fn_value(value_Model_1, value_Model_2, target_value_Model_1, target_value_Model_2, target_encoder, target_policy, target_r_avg, r_avg, sup_episode, key): # Work in (T, B, ..)
        states, actions, rewards, dones, next_states = sup_episode["states"], sup_episode["actions"], sup_episode["rewards"], sup_episode["dones"], sup_episode["next_states"]

        gamma = .99
       
        T = states.shape[0]

        def _get_values(carry, xs):
            v, m = carry
            r, i, d = xs

            v = v + r * gamma**i * m
            m = m*(1 - d)

            return (v, m), None
        
        i_s = jnp.arange(T - 1)
        v_0 = jnp.zeros((states.shape[1]))
        m = jnp.ones_like(v_0)
        (v, m), _ = jax.lax.scan(_get_values, (v_0, m), (rewards[:-1, :], i_s, dones[:-1, :]), length= T - 1)

        state_end = states[-1]
        zs = target_encoder.get_zs(state_end)
        action, _ = target_policy(zs)
        key, subkey = random.split(key)
        action = random.normal(subkey, action.shape) * .1 + action
        _, _, _, _, zsa = target_encoder(state_end, action)

        target_value_1 = jax.lax.stop_gradient(target_value_Model_1(zsa))
        target_value_2 = jax.lax.stop_gradient(target_value_Model_2(zsa))
        target_value = jnp.minimum(target_value_1, target_value_2) * target_r_avg

        value_goal = (1 / r_avg) * (v + gamma**T * target_value * m)

        value_predict_1 = value_Model_1(zsa)
        value_predict_2 = value_Model_2(zsa)

        loss_1 = jnp.mean(jnp.where(jnp.abs(value_goal - value_predict_1) < 1, .5* (value_goal - value_predict_1)**2, jnp.abs(value_goal - value_predict_1) - .5))
        loss_2 = jnp.mean(jnp.where(jnp.abs(value_goal - value_predict_2) < 1, .5* (value_goal - value_predict_2)**2, jnp.abs(value_goal - value_predict_2) - .5))

        return (loss_1 + loss_2) / 2
    
    @staticmethod
    @jax.jit
    def loss_fn_policy(policy_Model, value_Model_1, value_Model_2, encoder, sup_episode, key): # Work in (T, B, ..)
        states, actions, rewards, dones, next_states = sup_episode["states"], sup_episode["actions"], sup_episode["rewards"], sup_episode["dones"], sup_episode["next_states"]
        
        llamba_pre_act = 1e-5

        mask = jnp.cumprod(1.0 - dones, axis=0, exclusive=True)
        mask = mask[..., None]

        T, B, data_len = mask.shape
        flat_masks = mask.reshape(T*B, -1)

        T, B, data_len = states.shape
        flat_states = states.reshape(T*B, -1)

        flat_zs = encoder.get_zs(flat_states)
        flat_action, flat_pre_act = policy_Model(flat_zs)

        # key, subkey = random.split(key)
        # action = random.normal(subkey, action.shape) * .1 + action

        _, _, _, _, flat_zsa = encoder(flat_states, flat_action)

        flat_value_1 = value_Model_1(flat_zsa)
        flat_value_2 = value_Model_2(flat_zsa)

        loss = jnp.mean((-0.5 * (flat_value_1 + flat_value_2) + llamba_pre_act * (flat_pre_act**2).mean(axis=-1, keepdims=True)) * flat_masks) 

        
        return loss
    
    
    def train(self):

        
        def _pick(bi, ti, ei, field):
            return self.buffer[bi][field][ti, ei]  

        def _window(bi, ti, ei, field, H):
            return self.buffer[bi][field][ti : ti+H, ei]    

        def _sample_step(rng, N):
            T, B = self.steps, self.Sim.cfg.batch
            rng, a = random.split(rng); buf = random.randint(a, (N,), 0, self.capacity)
            rng, b = random.split(rng); t   = random.randint(b, (N,), 0, T)
            rng, c = random.split(rng); env = random.randint(c, (N,), 0, B)
            g = jax.vmap(lambda bi, ti, ei, k: _pick(bi,ti,ei,k), in_axes=(0,0,0,None))
            return {k: g(buf,t,env,k) for k in ["states","actions","rewards","dones","next_states"]}, rng

        def _sample_window(rng, N, H):
            T, B = self.steps, self.Sim.cfg.batch
            rng, a = random.split(rng); buf = random.randint(a, (N,), 0, self.capacity)
            rng, b = random.split(rng); t   = random.randint(b, (N,), 0, T-H+1)
            rng, c = random.split(rng); env = random.randint(c, (N,), 0, B)
            g = jax.vmap(lambda bi, ti, ei, k: _window(bi,ti,ei,k,H), in_axes=(0,0,0,None))
            return {k: g(buf,t,env,k) for k in ["states","actions","rewards","dones","next_states"]}, rng

        def _mean_abs_reward():
            tot, n = 0.0, 0
            for slot in self.buffer:
                r = slot["rewards"]
                tot += jnp.abs(r).sum()
                n += r.size
            return float(tot/n)

        
        enc_grad = jax.jit(jax.value_and_grad(self.loss_fn_encoder))
        val_grad = jax.jit(jax.value_and_grad(self.loss_fn_value, argnums=(0,1)))
        pi_grad  = jax.jit(jax.value_and_grad(self.loss_fn_policy))

        
        ptr = 0
        while ptr < self.capacity:
            batch, self.key = self.rollout(self.encoder, self.policy, self.key)
            self.buffer[ptr] = batch
            ptr += 1

        target_r_avg = r_avg = _mean_abs_reward()

       
        for epoch in range(self.epochs):

            batch, self.key = self.rollout(self.encoder, self.policy, self.key)
            self.buffer[ptr] = batch
            ptr = (ptr + 1) % self.capacity

            for _ in range(self.updates_per_epoch):

                # Encoder
                enc_sup, self.key = _sample_window(self.key, self.minibatch_size, self.horizon_enc+1)
                enc_loss, enc_gr  = enc_grad(self.target_encoder, self.encoder, enc_sup, self.num_bins)
                upd, self.encoder_opt_state = self.encoder_opt.update(enc_gr, self.encoder_opt_state)
                self.encoder = optax.apply_updates(self.encoder, upd)

                # Critics
                val_sup, self.key = _sample_window(self.key, self.minibatch_size,
                                                self.horizon_Q+1)
                (vl1, vl2), (g1,g2) = val_grad(self.value_1, self.value_2, self.target_value_1, self.target_value_2, self.target_encoder, self.target_policy, target_r_avg, r_avg, val_sup, self.key)
                upd1, self.value_1_opt_state = self.value_1_opt.update(g1, self.value_1_opt_state)
                upd2, self.value_2_opt_state = self.value_2_opt.update(g2, self.value_2_opt_state)
                self.value_1 = optax.apply_updates(self.value_1, upd1)
                self.value_2 = optax.apply_updates(self.value_2, upd2)

                # Policy
                policy_sup, self.key = _sample_step(self.key, self.minibatch_size)
                policy_loss, policy_gr = pi_grad(self.policy, self.value_1, self.value_2, self.encoder, policy_sup, self.key)
                upd, self.policy_opt_state = self.policy_opt.update(policy_gr, self.policy_opt_state)
                self.policy = optax.apply_updates(self.policy, upd)

            # (c) Sync target nets every target_sync_every epochs
            if (epoch + 1) % self.target_sync_every == 0:
                self.target_encoder = self.encoder
                self.target_value_1 = self.value_1
                self.target_value_2 = self.value_2
                self.target_policy  = self.policy
                target_r_avg, r_avg = r_avg, _mean_abs_reward()

            # (d) Logging
            mean_r = batch["rewards"].mean().item()
            print(f"epoch {epoch:4d} | R̄={mean_r:6.3f}  "
                f"enc={enc_loss:6.4f} v={vl1:6.4f} pi={policy_loss:6.4f}")

        # ------------------------------------------------------------------
        # 2.  Save models at the end
        # ------------------------------------------------------------------
        ckpt_dir = Path("checkpoints_mrq").resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        for name, obj in [
            ("enc_",   self.encoder),
            ("val1_",  self.value_1),
            ("val2_",  self.value_2),
            ("pi_",    self.policy),
            ("tenc_",  self.target_encoder),
            ("tval1_", self.target_value_1),
            ("tval2_", self.target_value_2),
            ("tpi_",   self.target_policy),
        ]:
            checkpoints.save_checkpoint(ckpt_dir, obj, epoch+1, prefix=name)