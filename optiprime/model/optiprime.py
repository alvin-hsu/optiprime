"""
This file implements the OptiPrime model. OptiPrime is a mechanistic machine learning model with
the following mechanistic structure:
    0:unedited*   --pe_on->    u_pbs_bound:1
    1:u_pbs_bound --pe_off->   unedited*  :0
    1:u_pbs_bound --syn{50}->  u_flap3    :2
    2:u_flap3     --u_fen1->   het_duplex :3
    3:het_duplex  --rep_u->    unedited*  :0
    3:het_duplex  --rep_e->    edited*    :4
    3:het_duplex  --muts_on->  muts_het   :5
    5:muts_het    --muts_off-> het_duplex :3
    5:muts_het    --mmr->      unedited*  :0

"""
from typing import Any, Callable, Dict

from jax import jit
from jax.lax import select
import jax.numpy as jnp
from jax.scipy.linalg import expm
from flax.core.frozen_dict import FrozenDict
import flax.linen as nn

from rate_models import PegRNAMLP, SynModel, MMRModel, HetFormer


# OptiPrime simplified mechanism of PE
EDGES = [('pe_on', 0, 1),
         ('pe_off', 1, 0),
         ('syn', 1, 2),
         ('u_fen1', 2, 3),
         ('rep_u', 3, 0),
         ('rep_e', 3, 4),
         ('muts_on', 3, 5),
         ('muts_off', 5, 3),
         ('mmr', 5, 0)]


def _make_repeat_filler(a_idx: int, b_idx: int, offset: int, repeat_n: int):
    def fill_repeats(g: jnp.ndarray, r: float, k: int):
        k = jnp.minimum(k, repeat_n)
        g = g.at[a_idx, b_idx ].set(select(k == 1, r, 0.))
        g = g.at[a_idx, offset].set(select(k > 1, r, 0.))
        for i in range(0, repeat_n - 2):
            i_idx = offset + i
            g = g.at[i_idx, b_idx    ].set(select(i + 2 == k, r, 0.))
            g = g.at[i_idx, i_idx + 1].set(select(i + 2 < k, r, 0.))
        last_idx = offset + repeat_n - 2
        g = g.at[last_idx, b_idx].set(select(k == repeat_n, r, 0.))
        return g
    return fill_repeats

def _make_gmatrix_fn() -> Callable:
    n_states = 6
    repeat_filler = _make_repeat_filler(1, 2, n_states, 50)
    n_states = n_states + 49

    def gmatrix(rates: Dict[str, float], rtt_len: int) -> jnp.ndarray:
        g = jnp.zeros((n_states, n_states))
        for rate_name, idx_u, idx_v in EDGES:
            g = g.at[idx_u, idx_v].set(rates[rate_name])
        # k_syn is a repeated edge, since it is the *average* rate of adding nucleotides to the
        # growing flap, rather than the overall rate of making the flap.
        g = repeat_filler(g=g, r=rates['syn'], k=rtt_len)
        for idx in range(n_states):
            g = g.at[idx, idx].set(-g[idx, :].sum())
        return g

    return jit(gmatrix)


# noinspection PyAttributeOutsideInit
class RateModule(nn.Module):
    num_groups: int = 1

    def setup(self):
        self.generator = _make_gmatrix_fn()
        self.log_rates = {k: self.param(k, lambda _: jnp.zeros()) for k, _, _ in EDGES}  # Constants
        self.group_factors = {k: self.param(f'{k}_factor', lambda _: jnp.zeros(self.num_groups))
                              for k, _, _ in EDGES}
        self.init_idx = 0
        self.obs_idxs = jnp.array([0, 5], dtype=jnp.int32)

    def __call__(self, pred_rates: Dict[str, float], rtt_len: int, group_idx: int, time: float) \
            -> jnp.ndarray:
        rates = FrozenDict({k: v for k, v in self.log_rates.items()})
        rates = rates.copy(pred_rates)
        rates = FrozenDict({k: jnp.exp(v + self.group_factors[k][group_idx])
                            for k, v in rates.items()})
        g = self.generator(rates, rtt_len)
        self.sow('intermediates', 'g', g)
        p = expm(time * g, max_squarings=64)[self.init_idx, :]
        p = p / p.sum()
        obs_preds = jnp.take(p, self.obs_idxs)
        obs_preds = obs_preds / obs_preds.sum()
        return obs_preds


# noinspection PyAttributeOutsideInit
class OptiPrime(nn.Module):
    num_groups: int = 1

    def __call__(self, inputs: Dict[str, Any], rtt_len: int, group_idx: int = 0, time: float = 1.0):
        pe_on = PegRNAMLP(4, 32)(*inputs['pe_on'])
        syn = SynModel()(*inputs['syn'])
        mmr = MMRModel()(*inputs['mmr'])
        het_out = HetFormer(embed_dim=64, num_recycles=1, num_blocks=4,
                            attn_heads=8, attn_dim=8, mlp_dim=32,
                            pair_dim=16, dropout=0.25, out_dim=32)(*inputs['hetformer'])
        pred_rates = {
            'pe_on': pe_on,
            'syn': syn,
            'mmr': mmr,
            'rep_u': het_out[0],
            'rep_e': het_out[1],
            'muts_off': het_out[2]
        }
        obs_preds = RateModule(self.num_groups)(pred_rates, rtt_len)
        return obs_preds[1]
