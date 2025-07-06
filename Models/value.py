import jax.numpy as jnp, jax
from jax import random
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class Value:
    def __init__(self, size_in, key=random.PRNGKey(0)):
        self.size_in = size_in
        self.key = key
        # three dense layers as before
        self.w1, self.b1, self.key = self._layer(size_in, 512, self.key, bias_scale=0.0, use_normal=True)
        self.w2, self.b2, self.key = self._layer(512,     512, self.key, bias_scale=0.0, use_normal=True)
        self.w3, self.b3, self.key = self._layer(512,     512, self.key, bias_scale=0.0, use_normal=True)
        self.w4, self.b4, self.key = self._layer(512,     1, self.key, bias_scale=0.0, use_normal=True)


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
    @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
    def __call__(self, zsa):

        x = self._act(self._layer_norm(self.w1 @ zsa + self.b1))
        x = self._act(self._layer_norm(self.w2 @ x + self.b2))
        x = self._act(self._layer_norm(self.w3 @ x + self.b3))
        value = self.w4 @ x + self.b4

        return value

    # ── pytree interface (unchanged) ───────────────────────────────────────
    def tree_flatten(self):
        children = (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4)
        aux = (self.size_in)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        size_in = aux
        obj = cls.__new__(cls)  # bypass __init__
        (obj.w1, obj.b1, obj.w2, obj.b2, obj.w3, obj.b3, obj.w4, obj.b4) = children
        obj.size_in = size_in
        obj.key = None
        return obj
