__all__ = ['LayerNorm', 'LinearHead', 'QueryHead', 'init_transformer', 'sinusoids', 'MultiHeadAttention',
           'ResidualAttentionBlock', 'BaseDecoder', 'EmbeddingProjector', 'FlexEmbeddings']

import torch
import numpy as np
import math

from torch import Tensor, nn
import torch.nn.functional as F
from typing import Dict, Iterable, Optional

class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class LinearHead(nn.Linear):
    pass

class QueryHead(nn.Linear):
    pass

def init_transformer(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        torch.nn.init.trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)

def sinusoids(length, channels, max_timescale=10000):
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, qk_scale: float = 1, rope: bool = False, cross=False):
        super().__init__()
        self.n_state = n_state
        self.n_head = n_head
        self.sqrt_qk_scale = math.sqrt(qk_scale)
        self.query = QueryHead(n_state, n_state)
        self.key = nn.Linear(n_state, n_state, bias=False)
        self.value = nn.Linear(n_state, n_state)
        self.out = nn.Linear(n_state, n_state)
        self.cross = cross
        self.query_subsampling = 1
        self.key_subsampling = 1

        self.cached_kvx = None
        self.register_buffer('k_cache', None)
        self.register_buffer('v_cache', None)
        
        self.rotary = None
        if rope:
            self.rotary = Rotary(n_state // n_head)
        self.qkv = None
        self.kv = None

    def setup_kv_cache(self, max_batch_size, max_seq_len, dtype=torch.float32):
        cache_shape = (max_batch_size, self.n_head, max_seq_len, self.n_state//self.n_head)
        self.k_cache = torch.zeros(cache_shape, dtype=dtype, device=self.key.weight.device)
        self.v_cache = torch.zeros(cache_shape, dtype=dtype, device=self.value.weight.device)

    def merge_linears(self, layers, mults):
        bias = [x.bias for x in layers if x.bias is not None][0]
        din, dout = layers[0].weight.shape
        new = nn.Linear(din, len(layers) * dout).to(layers[0].weight.device)
        with torch.no_grad():
            new.weight[:] = torch.cat([x.weight * m for x,m in zip(layers, mults)])
            new.bias[:] = torch.cat([torch.zeros_like(bias) if x.bias is None else x.bias * m for x, m in zip(layers, mults)])
        return new

    def convert_for_eval(self):
        if self.qkv or self.kv: raise AttributeError("already converted")
        
        self.odim = self.key.weight.shape[1]
        if self.cross:
            self.q = self.merge_linears([self.query], [self.sqrt_qk_scale])
            self.kv = self.merge_linears([self.key, self.value],
                                         [self.sqrt_qk_scale, 1])
        else:
            self.qkv = self.merge_linears([self.query, self.key, self.value],
                                          [self.sqrt_qk_scale, self.sqrt_qk_scale, 1])
        
    def split_heads(self, x, x_positions, rope=False, subsampling=1):
        x = x.view(*x.shape[:2], self.n_head, -1)
        if rope:
            x = rope_rotate(x, x_positions * subsampling, *self.rotary(x))
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        qx,
        q_positions,
        kvx,
        kv_positions,
        causal = False,
        mask=None,
    ):
        if self.k_cache is not None:
            assert qx.shape[0] <= self.k_cache.shape[0], "please pass in a larger max_batch_size to setup_kv_cache"
        if self.qkv:
            q,k,v = self.qkv(qx).split(self.odim, dim=-1)
        elif self.kv:
            q = self.q(qx)
            k,v = self.kv(kvx).split(self.odim, dim=-1)
        else:
            q,k,v = None,None,None
        
        if q is None: q = self.query(qx) * self.sqrt_qk_scale
        q = self.split_heads(q, q_positions, rope = self.rotary, subsampling = self.query_subsampling)

        if kvx is not self.cached_kvx:
            if k is None: k = self.key(kvx) * self.sqrt_qk_scale
            k = self.split_heads(k, kv_positions, rope = self.rotary, subsampling = self.key_subsampling)
            if v is None: v = self.value(kvx)
            v = self.split_heads(v, kv_positions)
            if self.k_cache is not None:
                self.k_cache[:k.shape[0],:,kv_positions] = k
                self.v_cache[:v.shape[0],:,kv_positions] = v

        if self.k_cache is not None:
            k, v = self.k_cache[:k.shape[0]], self.v_cache[:v.shape[0]]

        if mask is not None:
            mask = mask[q_positions,:k.shape[-2]]
            
        wv = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0, is_causal=causal)
        
        return self.out(wv.permute(0, 2, 1, 3).flatten(start_dim=2))

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if not self.seq_len_cached or seq_len > self.seq_len_cached:
            self.seq_len_cached = 2500
            
            t = torch.arange(self.seq_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None, :, None, :]
            self.sin_cached = emb.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=len(x.shape)-1
    )

def rope_rotate(x, positions, cos, sin):
    return x * cos[:,positions] + rotate_half(x) * sin[:,positions]

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, rope: bool = False,
                 qk_scale: float = 1, ffn_mult: int = 4):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, qk_scale=qk_scale, rope=rope)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(n_state, n_head, qk_scale=qk_scale, rope=rope, cross=True) if cross_attention else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * ffn_mult
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)
    
    def setup_kv_cache(self, max_batch_size, max_seq_len, max_cross_seq_len=None):
        self.attn.setup_kv_cache(max_batch_size, max_seq_len)
        if self.cross_attn:
            self.cross_attn.setup_kv_cache(max_batch_size, max_cross_seq_len)
    
    def forward(
        self,
        x: Tensor,
        x_positions: Tensor = None,
        xa: Optional[Tensor] = None,
        xa_positions: Optional[Tensor] = None,
        causal = False,
        mask=None,
    ):
        lnx = self.attn_ln(x)
        x = x + self.attn(lnx, x_positions, lnx, x_positions, causal=causal, mask=mask)
        if self.cross_attn:
            lnx = self.cross_attn_ln(x)
            x = x + self.cross_attn(lnx, x_positions, xa, xa_positions)
        x = x + self.mlp(self.mlp_ln(x))
        return x

class BaseDecoder(nn.Module):
    def __init__(self, depth=6, n_head=6, width=384, qk_scale=1, ffn_mult=4, length=2250, rope=False):
        super().__init__()
        self.length = length
        self.width = width
        self.layers = nn.ModuleList([
            ResidualAttentionBlock(
                self.width, n_head, qk_scale=qk_scale, ffn_mult=ffn_mult, cross_attention=True, rope=rope
            ) for _ in range(math.floor(depth))
        ])

        self.ln_post = LayerNorm(width)
        
        mask = torch.empty(length, length).fill_(-torch.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x, x_positions, xenc, xenc_positions):
        for i,l in enumerate(self.layers):
            x = l(x, x_positions, xenc, xenc_positions, causal=self.training, mask=self.mask if not self.training else None)

        x = self.ln_post(x)

        return x

class EmbeddingProjector(nn.Linear):
    pass

class FlexEmbeddings(nn.Module):
    def __init__(self, codes, width, special_codes=None, frozen_width=None, special_embedding=None, unembed=True):
        super().__init__()
        self.codes = codes
        self.special_codes = special_codes
        if frozen_width is None: frozen_width = width
        
        self.main = nn.Embedding(codes, frozen_width or width)
        self.emb_to_hidden = EmbeddingProjector(frozen_width, width) if frozen_width != width else None
        self.hidden_to_emb = EmbeddingProjector(width, frozen_width) if unembed and frozen_width != width else None
        if special_codes:
            self.special = special_embedding or nn.Embedding(special_codes, width)
            
        self.register_buffer('merged_in', None)
        self.register_buffer('merged_out', None)
        self.register_buffer('bias_out', None)
    
    def set_frozen_embeddings(self, values):
        with torch.no_grad():
            self.main.weight[:] = values
            self.main.lr_scale = 0
    
    @torch.no_grad()
    def convert_for_eval(self):
        if not self.special_codes: return
        main_w = self.main.weight
        if self.emb_to_hidden is not None: main_w = self.emb_to_hidden(main_w)
        weight = torch.cat([main_w, self.special.weight], dim=0)
        self.merged_in = nn.Embedding(*weight.shape, _weight=weight)
        
        weight = self.main.weight
        if self.hidden_to_emb: weight = weight @ self.hidden_to_emb.weight
        self.merged_out = torch.cat([weight.T, self.special.weight.T], dim=1).T.contiguous()
        if self.hidden_to_emb:
            self.bias_out = torch.cat([
                self.hidden_to_emb.bias @ self.main.weight.T,
                torch.zeros(self.special.weight.shape[0], device=weight.device, dtype=weight.dtype)
            ], dim=0)
        else:
            self.bias_out = None

    def forward(self, toks):
        if not self.training and self.merged_in is not None:
            return self.merged_in(toks)
        
        if self.special_codes:
            special_mask = toks >= self.codes
            embs = self.main(torch.where(special_mask, 0, toks))
        else:
            embs = self.main(toks)
        
        if self.emb_to_hidden: embs = self.emb_to_hidden(embs)
        
        if self.special_codes:
            embs[special_mask] = self.special(toks[special_mask] - self.codes).to(embs.dtype)
        
        return embs
    
    def unembed(self, embs):
        if not self.training and self.merged_out is not None:
            return F.linear(embs, self.merged_out, self.bias_out)

        orig_embs = embs
        if self.hidden_to_emb: embs = self.hidden_to_emb(embs)
        
        main_logits = (embs @ self.main.weight.to(embs.dtype).T).float()
        
        if not self.special_codes:
            return main_logits
        
        special_logits = (orig_embs @ self.special.weight.to(orig_embs.dtype).T).float()
        return torch.cat([main_logits, special_logits], dim=-1)
