from typing import Tuple

import jax.numpy as jnp
from jax.nn import sigmoid, softmax
from jax.random import PRNGKey, bernoulli, split
import flax.linen as nn
import flax.linen.initializers as init

from .utils import PositionalEncodings, make_attention_mask, make_sequence_mask, jnp_stable_log


_NINF = jnp.finfo(jnp.float32).min
_EPS = 0.001


# noinspection PyAttributeOutsideInit
class MultiHeadCrossAttentionWithBias(nn.Module):
    embed_dim: int
    num_heads: int
    head_dim: int  # NOT qkv_features; number of features PER head
    dropout_rate: float

    def setup(self):
        qkv_features = (self.num_heads, self.head_dim)
        self.w_query = nn.DenseGeneral(name='W_Q', axis=-1, features=qkv_features)
        self.w_key   = nn.DenseGeneral(name='W_K', axis=-1, features=qkv_features)
        self.w_value = nn.DenseGeneral(name='W_V', axis=-1, features=qkv_features)
        self.w_out   = nn.DenseGeneral(name='W_O', axis=(-2, -1), features=self.embed_dim)
        self.bpp_w = self.param('bpp_w', init.constant(1.0), ())
        self.bpp_b = self.param('bpp_b', init.constant(0.0), ())

    def __call__(self, u_enc: jnp.ndarray, e_enc: jnp.ndarray, logit_bpp: jnp.ndarray,
                 ue_mask: jnp.ndarray, eu_mask: jnp.ndarray,
                 dropout_key: PRNGKey = None, deterministic: bool = True) \
            -> Tuple[jnp.ndarray, jnp.ndarray]:
        assert u_enc.ndim == e_enc.ndim, 'u, e must have the same rank.'
        scale = jnp.reciprocal(jnp.sqrt(self.head_dim))
        # Use input_u and input_e as both queries and keys
        q_u, q_e = scale * self.w_query(u_enc), scale * self.w_query(e_enc)
        k_u, k_e = self.w_key(u_enc), self.w_key(e_enc)
        # Scale bpp and apply bias, then expand dims and compute its transpose
        logit_bpp = self.bpp_w * logit_bpp + self.bpp_b
        logit_bpp_t = logit_bpp.transpose()
        logit_bpp, logit_bpp_t = jnp.expand_dims(logit_bpp, -3), jnp.expand_dims(logit_bpp_t, -3)
        # Attention weights with logit bpp bias added
        ue_weights = jnp.einsum('...qhd,...khd->...hqk', q_u, k_e) + logit_bpp
        eu_weights = jnp.einsum('...qhd,...khd->...hqk', q_e, k_u) + logit_bpp_t
        # Apply masks to entries and compute softmax
        ue_ninf = ~ue_mask * _NINF
        eu_ninf = ~eu_mask * _NINF
        ue_weights = softmax(ue_weights + ue_ninf + _EPS)
        eu_weights = softmax(eu_weights + eu_ninf + _EPS)
        if not deterministic and self.dropout_rate > 0.0:
            ue_key, eu_key = split(dropout_key)
            keep_prob = 1.0 - self.dropout_rate
            ue_keep = bernoulli(ue_key, keep_prob, ue_weights.shape)
            eu_keep = bernoulli(eu_key, keep_prob, eu_weights.shape)
            ue_weights = ue_keep * ue_weights
            eu_weights = eu_keep * eu_weights
        # Value/output
        ue_weights = ue_mask * ue_weights
        eu_weights = eu_mask * eu_weights
        v_u, v_e = self.w_value(u_enc), self.w_value(e_enc)
        u_update = self.w_out(jnp.einsum('...hqk,...khd->...qhd', ue_weights, v_e))
        e_update = self.w_out(jnp.einsum('...hqk,...khd->...qhd', eu_weights, v_u))
        return u_update, e_update


# noinspection PyAttributeOutsideInit
class OuterProductUpdate(nn.Module):
    pair_dim: int

    def setup(self):
        self.w_pair = nn.Dense(name='pair', features=self.pair_dim)

    def __call__(self, u_enc: jnp.ndarray, e_enc: jnp.ndarray, ue_mask: jnp.ndarray):
        u_enc, e_enc = jnp.squeeze(self.w_pair(u_enc)), jnp.squeeze(self.w_pair(e_enc))
        update = jnp.einsum('...ud,...ed->...ue', u_enc, e_enc)
        return jnp.squeeze(ue_mask) * update


# noinspection PyAttributeOutsideInit
class HetFormerBlock(nn.Module):
    embed_dim: int
    attn_heads: int
    attn_dim: int
    mlp_dim: int
    pair_dim: int
    dropout: float

    def setup(self):
        # Compute total QKV dimension needed
        qkv_features = self.attn_heads * self.attn_dim
        # Sequence representation update blocks
        self.cross_attn = MultiHeadCrossAttentionWithBias(name='cross_attn',
                                                          embed_dim=self.embed_dim,
                                                          num_heads=self.attn_heads,
                                                          head_dim=self.attn_dim,
                                                          dropout_rate=self.dropout)
        self.self_attn = nn.MultiHeadDotProductAttention(name='self_attn',
                                                         num_heads=self.attn_heads,
                                                         qkv_features=qkv_features,
                                                         dropout_rate=self.dropout)
        self.mlp_in = nn.Dense(name='mlp_in', features=self.mlp_dim)
        self.mlp_out = nn.Dense(name='mlp_out', features=self.embed_dim)
        # LayerNorms
        self.cross_norm = nn.LayerNorm()
        self.self_norm = nn.LayerNorm()
        self.mlp_norm = nn.LayerNorm()
        # Pairing representation update block
        self.pair_w = self.param('pair_w', init.constant(0.001), ())
        self.pair_update = OuterProductUpdate(name='pair_update', pair_dim=self.pair_dim)

    def __call__(self, u_enc: jnp.ndarray, e_enc: jnp.ndarray, pair_enc: jnp.ndarray,
                 u_mask: jnp.ndarray, e_mask: jnp.ndarray,
                 ue_mask: jnp.ndarray, eu_mask: jnp.ndarray, deterministic: bool):
        # Cross-attention
        u_update, e_update = self.cross_attn(u_enc, e_enc, pair_enc, ue_mask, eu_mask)
        u_enc = self.cross_norm(u_enc + u_update)
        e_enc = self.cross_norm(e_enc + e_update)
        # Self-attention
        u_update = self.self_attn(u_enc, u_enc, mask=u_mask, deterministic=deterministic)
        e_update = self.self_attn(e_enc, e_enc, mask=e_mask, deterministic=deterministic)
        u_enc = self.self_norm(u_enc + u_update)
        e_enc = self.self_norm(e_enc + e_update)
        # MLP updates
        u_update = self.mlp_out(nn.relu(self.mlp_in(u_enc)))
        e_update = self.mlp_out(nn.relu(self.mlp_in(e_enc)))
        u_enc = self.mlp_norm(u_enc + u_update)
        e_enc = self.mlp_norm(e_enc + e_update)
        # Update pairwise representation
        pair_enc = pair_enc + self.pair_w * self.pair_update(u_enc, e_enc, ue_mask)
        return u_enc, e_enc, pair_enc


class SubOut(nn.Module):
    pair_dim: int

    @nn.compact
    def __call__(self, u_enc: jnp.ndarray, e_enc: jnp.ndarray, pair_enc: jnp.ndarray,
                 ue_mask: jnp.ndarray):
        bpp_w = self.param('bpp_w', init.constant(1.0), ())
        bpp_b = self.param('bpp_b', init.constant(0.0), ())
        op = OuterProductUpdate(name='pair_sub', pair_dim=self.pair_dim)(u_enc, e_enc, ue_mask)
        mask = jnp.squeeze(ue_mask) * sigmoid(bpp_w * pair_enc + bpp_b + op)
        mask = jnp.expand_dims(mask, -1)
        u_rep = jnp.expand_dims(u_enc, -2).repeat(e_enc.shape[-2], axis=-2)
        e_rep = jnp.expand_dims(e_enc, -3).repeat(u_enc.shape[-2], axis=-3)
        pair_rep = mask * jnp.concatenate([u_rep, e_rep], axis=-1)
        return pair_rep.sum(axis=(-2, -3))


# noinspection PyAttributeOutsideInit
class IndelOut(nn.Module):
    def setup(self):
        self.gate = nn.Dense(1)

    def __call__(self, u_enc: jnp.ndarray, e_enc: jnp.ndarray, pair_enc: jnp.ndarray,
                 ue_mask: jnp.ndarray):
        # Mask out positions outside sequence
        ue_mask = ue_mask.squeeze()
        u_mask = jnp.expand_dims(~(~ue_mask).all(axis=-1), -1)
        e_mask = jnp.expand_dims(~(~ue_mask).all(axis=-2), -1)
        # Learnable gate for which positions in u and e to take
        u_gate = u_mask * nn.sigmoid(self.gate(u_enc))
        e_gate = e_mask * nn.sigmoid(self.gate(e_enc))
        # Take positions in each sequence weighted by gate and sum
        u_enc = (u_gate * u_enc).sum(axis=-2)
        e_enc = (e_gate * e_enc).sum(axis=-2)
        return jnp.concatenate([u_enc, e_enc], axis=-1)


# noinspection PyAttributeOutsideInit
class HetFormer(nn.Module):
    # Sequence lengths
    max_u_len: int
    max_e_len: int
    # Model hyperparameters
    embed_dim: int
    # HetFormer blocks
    num_recycles: int
    num_blocks: int
    # HetFormer hyperparams
    attn_heads: int
    attn_dim: int
    mlp_dim: int
    pair_dim: int
    dropout: float
    # Output MLP features
    out_dim: int
    # Whether to output sequence-vectors or not
    pretraining: bool = False

    def setup(self):
        self.embedding = nn.Dense(self.embed_dim)
        self.pos_enc_u = PositionalEncodings(self.embed_dim, self.max_u_len)
        self.pos_enc_e = PositionalEncodings(self.embed_dim, self.max_e_len)
        self.hetformer = [HetFormerBlock(self.embed_dim, self.attn_heads, self.attn_dim,
                                         self.mlp_dim, self.pair_dim, self.dropout)
                          for _ in range(self.num_blocks)]
        # Obtaining output embeddings
        self.sub_out = SubOut(self.pair_dim)
        self.indel_out = IndelOut()
        # Output MLP
        self.out_norm = nn.LayerNorm(name='out_norm')
        self.out_dense = nn.Dense(name='out_dense', features=self.out_dim)
        self.output = nn.Dense(name='out', features=3)

    def __call__(self, u: jnp.ndarray, e: jnp.ndarray, bpp: jnp.ndarray, u_len: int, e_len: int,
                 hom_end_oh: jnp.ndarray, hom_end: jnp.ndarray, deterministic: bool):
        # Make various attention masks to be reused throughout model
        u_mask = make_attention_mask(u, u, u_len, u_len)
        e_mask = make_attention_mask(e, e, e_len, e_len)
        ue_mask = make_attention_mask(u, e, u_len, e_len)
        eu_mask = make_attention_mask(e, u, e_len, u_len)
        # Convert to embeddings and add positional embeddings
        u, e = self.embedding(u), self.embedding(e)
        u = self.pos_enc_u(u, 0, make_sequence_mask(u, u_len))
        e = self.pos_enc_e(e, 0, make_sequence_mask(e, e_len))
        # Turn base pair probs into logit space
        pair = jnp_stable_log(bpp, eps=1e-3) - jnp_stable_log(1 - bpp, eps=1e-3)
        if self.pretraining and not deterministic:
            # Randomly replace 10% of base pair probs to 0.5 while pretraining
            rng = self.make_rng('dropout')
            mask = bernoulli(rng, p=0.9, shape=pair.shape)
            pair = mask * pair
        # Run the HetFormer
        for _ in range(self.num_recycles):
            for block in self.hetformer:
                u, e, pair = block(u, e, pair, u_mask, e_mask, ue_mask, eu_mask,
                                   deterministic=deterministic)
        # Output layers
        sub = self.sub_out(u, e, pair, ue_mask)
        indel = self.indel_out(u, e, pair, ue_mask)
        out = jnp.hstack([sub, indel, hom_end_oh, hom_end])
        out = self.out_norm(out)
        out = self.out_dense(out)
        out = nn.relu(out)
        out = self.output(out)
        if self.pretraining:
            x = jnp.hstack([sub, indel])
            x = x.reshape(x.shape[:-2] + (-1,))
            return u, e, x
        else:
            return out
