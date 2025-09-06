import flax.linen as nn
import flax.linen.initializers as init
import jax.numpy as jnp

from optiprime.constants import CELL_TYPES

# noinspection PyAttributeOutsideInit
class SynModel(nn.Module):
    def setup(self):
        self.dntp_param = self.param('dntp_factor', init.constant(1), (len(CELL_TYPES), 4))
        self.dense = nn.Dense(1, use_bias=True, kernel_init=init.constant(1))

    def __call__(self, cell_type: jnp.ndarray, rtt_len: int, base_counts: jnp.ndarray,
                 rtt_bp_prob: float, deterministic: bool):
        cell_type = jnp.expand_dims(cell_type, -2)
        base_counts = jnp.expand_dims(base_counts, -1)
        x = cell_type @ self.dntp_param @ base_counts
        x = jnp.squeeze(x)
        x = jnp.hstack([x, rtt_bp_prob])
        x = self.dense(x)
        x = jnp.squeeze(x)
        x = x / rtt_len
        return x
