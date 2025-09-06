import jax.numpy as jnp
import flax.linen as nn
import flax.linen.initializers as init


# noinspection PyAttributeOutsideInit
class MMRModel(nn.Module):
    @nn.compact
    def __call__(self, *inputs, deterministic: bool):
        x = jnp.hstack(inputs)
        x = nn.LayerNorm()(x)
        x = nn.Dense(features=1, use_bias=True, kernel_init=init.normal())(x)
        return jnp.squeeze(x)
