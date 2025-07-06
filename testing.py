from Sim import Sim, SimCfg, SimData
import mujoco
from mujoco import mjx
from jax import lax
import jax
import jax.numpy as jnp
from dataclasses import dataclass, field
from jax.tree_util import register_pytree_node_class
import jax.tree_util

from jax import random
from MRQ import MRQ
from Models.policy import Policy
from Models.encoder import Encoder
from Models.value  import Value


zs_dim = 64
za_dim = 32
zsa_dim = 64
number_bins = 21

policy = Policy(zs_dim, 3)
value_1 = Value(zsa_dim)
value_2 = Value(zsa_dim)
encoder = Encoder(19, 3, zs_dim, za_dim, zsa_dim, number_bins + 1 + zs_dim, number_bins)

cfg = SimCfg(
        xml_path="/home/leo-benaharon/Desktop/ping_pong/env_ping_pong.xml",
        batch    = 64,
        model_freq = 100,
        init_pos = jnp.array([3.5, 0, 1.3, 1, 0, 0, 0,   -1.5, 0.0, 1.],
                             dtype=jnp.float32),
        init_vel = jnp.array([-10, 0, 0, 0, 0, 0, 0, 0, 0],
                             dtype=jnp.float32),
    )
ctrl = jnp.zeros((cfg.batch, 3))

sim = Sim(cfg)


mrq = MRQ(encoder, policy, value_1, value_2, sim)

mrq.train()



