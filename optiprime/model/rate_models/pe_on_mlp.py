import jax.numpy as jnp
import flax.linen as nn

from optiprime.constants import CAS9_PAMS


# noinspection PyAttributeOutsideInit
class PegRNAMLP(nn.Module):
    num_layers: int
    hidden_dim: int

    @nn.compact
    def __call__(self, pamda: float, pam_idx: int, *inputs, deterministic: bool):
        x = jnp.hstack(inputs)
        x = nn.LayerNorm()(x)
        for i in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.elu(x)
        x = nn.Dense(features=1)(x)
        x = jnp.squeeze(x)
        # "Multiply" by PAM factor from HT-PAMDA (in log space)
        pamda_w = self.param('pamda_w', nn.initializers.ones, (1,))
        pamda_b = self.param('pamda_b', nn.initializers.zeros, (len(CAS9_PAMS),))
        pamda_b = jnp.take(pamda_b, pam_idx)
        # PAM variant factor
        x = jnp.squeeze(x + pamda_w * pamda + pamda_b)
        return x
