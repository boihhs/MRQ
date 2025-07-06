import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class State_Action_Encoder:
    # Out dim is zs_dim + 1 + num_bins
    def __init__(self, action_dim, za_dim, zs_dim, zsa_dim, out_dim, num_bins, key=random.PRNGKey(0)):
        self.action_dim, self.za_dim, self.zs_dim, self.zsa_dim, self.out_dim, self.num_bins = action_dim, za_dim, zs_dim, zsa_dim, out_dim, num_bins
        self.key = key
        # three dense layers as before
        self.w1, self.b1, self.key = self._layer(action_dim, za_dim, self.key, bias_scale=0.0, use_normal=True)
        self.w2, self.b2, self.key = self._layer(za_dim + zs_dim,     512, self.key, bias_scale=0.0, use_normal=True)
        self.w3, self.b3, self.key = self._layer(512,     512, self.key, bias_scale=0.0, use_normal=True)
        self.w4, self.b4, self.key = self._layer(512,     zsa_dim, self.key, bias_scale=0.0, use_normal=True)
        self.w5, self.b5, self.key = self._layer(zsa_dim,     out_dim, self.key, bias_scale=0.0, use_normal=True)



    @staticmethod
    def _layer(m, n, key, *, bias_scale=0.0, use_normal=True):
        key, sub_w, sub_b = random.split(key, 3)

        if use_normal:
            std = jnp.sqrt(2.0 / (m + n))
            w = std * random.normal(sub_w, (n, m))
        else:
            limit = jnp.sqrt(6.0 / (m + n))
            w = random.uniform(sub_w, (n, m), minval=-limit, maxval=limit)

        b = jnp.zeros((n,))
        return w, b, key
    
    @staticmethod
    @jax.jit
    def _layer_norm(x_in):
        x_in = (x_in - jnp.mean(x_in, axis=-1, keepdims=True)) / jnp.sqrt(jnp.var(x_in, axis=-1, keepdims=True) + 1e-5)
        return x_in
    
    @staticmethod
    @jax.jit
    def _act(x_in):
        x_in = jnp.where(x_in > 0, x_in, jnp.exp(x_in) - 1)
        return x_in

    # ── forward ────────────────────────────────────────────────────────────
    @jax.jit
    @partial(jax.vmap, in_axes=(None, 0, 0), out_axes=(0, 0, 0, 0))
    def __call__(self, zs, action):

        za = self._act((self.w1 @ action + self.b1))
        zsa = jnp.concatenate([zs, za], axis= -1)
        zsa = self._act(self._layer_norm(self.w2 @ zsa + self.b2))
        zsa = self._act(self._layer_norm(self.w3 @ zsa + self.b3))
        zsa = self.w4 @ zsa + self.b4

        MDP = self.w5 @ zsa + self.b5
        d = 1/(1 + jnp.exp(-MDP[0:1]))
        r = jax.nn.log_softmax(MDP[1:self.num_bins + 1], axis=-1)
        zs_prime = MDP[self.num_bins + 1:]

        return d, r, zs_prime, zsa

    # ── pytree interface (unchanged) ───────────────────────────────────────
    def tree_flatten(self):
        children = (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4, self.w5, self.b5)
        aux = (self.action_dim, self.za_dim, self.zs_dim, self.zsa_dim, self.out_dim, self.num_bins)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        
        obj = cls.__new__(cls)  # bypass __init__
        (obj.action_dim, obj.za_dim, obj.zs_dim, obj.zsa_dim, obj.out_dim, obj.num_bins) = aux
        (obj.w1, obj.b1, obj.w2, obj.b2, obj.w3, obj.b3, obj.w4, obj.b4, obj.w5, obj.b5) = children
  
        obj.key = None
        return obj
