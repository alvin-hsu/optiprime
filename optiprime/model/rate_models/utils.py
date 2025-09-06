from jax import jit
from jax.lax import dynamic_update_slice
import jax.numpy as jnp
import flax.linen as nn


EPS = 1e-5
def jnp_stable_log(x, eps=EPS):
    return jnp.log((x + eps) / (1 + eps))


@jit
def make_attention_mask(q_in: jnp.ndarray, kv_in: jnp.ndarray, q_len: int, kv_len: int):
    mask = (jnp.expand_dims((jnp.arange(0, q_in.shape[-2]) < q_len), -1) &
            jnp.expand_dims((jnp.arange(0, kv_in.shape[-2]) < kv_len), -2))
    mask = jnp.expand_dims(mask, -3)
    return mask


@jit
def make_sequence_mask(x: jnp.ndarray, x_len: int):
    return jnp.expand_dims((jnp.arange(x.shape[-2]) < x_len), -1)


# noinspection PyAttributeOutsideInit
class PositionalEncodings(nn.Module):
    num_dim: int
    max_len: int

    def setup(self):
        pos_emb = jnp.zeros((self.max_len, self.num_dim), dtype=jnp.float32)
        pos = jnp.arange(0, self.max_len, dtype=jnp.float32).reshape(-1, 1)
        div = jnp.exp2(jnp.arange(0, self.num_dim, 2) * (-256.0 / self.num_dim))
        pos_emb = pos_emb.at[:, 0::2].set(jnp.sin(pos * div))
        pos_emb = pos_emb.at[:, 1::2].set(jnp.cos(pos * div))
        self.pos_emb = pos_emb

    def __call__(self, x: jnp.ndarray, start: int, mask: jnp.ndarray):
        pe = jnp.zeros((2 * self.max_len, self.num_dim), dtype=jnp.float32)
        pe = dynamic_update_slice(pe, self.pos_emb, (start, 0))
        pe = pe[0:self.max_len, :]
        pe = jnp.where(mask, pe, 0)
        return x + pe
