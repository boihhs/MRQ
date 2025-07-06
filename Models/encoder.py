import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial
from Models.state_action_encoder import State_Action_Encoder
from Models.state_enocder import State_Enoder


@register_pytree_node_class
class Encoder:
    # Out dim is zs_dim + 1 + num_bins
    def __init__(self, state_dim, action_dim, zs_dim, za_dim, zsa_dim, out_dim, num_bins, key=random.PRNGKey(0)):
        self.state_dim, self.action_dim, self.zs_dim, self.za_dim, self.zsa_dim, self.out_dim, self.num_bins = state_dim, action_dim, zs_dim, za_dim, zsa_dim, out_dim, num_bins
        self.key = key

        self.state_action_encoder = State_Action_Encoder(action_dim, za_dim, zsa_dim, out_dim, num_bins)
        self.state_encoder = State_Enoder(state_dim, zs_dim)

    # ── forward ────────────────────────────────────────────────────────────
    @jax.jit
    def __call__(self, state, action):

        zs = self.state_encoder(state)
        d, r, zs_prime, zsa = self.state_action_encoder(zs, action)

        return zs, d, r, zs_prime, zsa
    
    @jax.jit
    def get_zs(self, state):

        zs = self.state_encoder(state)

        return zs

    # ── pytree interface (unchanged) ───────────────────────────────────────
    def tree_flatten(self):
        children = (self.state_action_encoder, self.state_encoder)
        aux = (self.state_dim, self.action_dim, self.zs_dim, self.za_dim, self.zsa_dim, self.out_dim, self.num_bins)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        
        obj = cls.__new__(cls)  # bypass __init__
        (obj.state_dim, obj.action_dim, obj.zs_dim, obj.za_dim, obj.zsa_dim, obj.out_dim, obj.num_bins) = aux
        (obj.state_action_encoder, obj.state_encoder) = children
  
        obj.key = None
        return obj
