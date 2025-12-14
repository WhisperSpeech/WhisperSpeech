__all__ = ['DelSumEmbedding', 'DelSumHead', 'Tunables', 'CategoricalEmbedding', 'BinnedEmbedding',
           'SpeakerEmbedding', 'SADelARTransformer']

import math
import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.profiler import record_function
from fastcore.basics import store_attr
from huggingface_hub import hf_hub_download

from pathlib import Path
from fastprogress import progress_bar

from . import inference, languages
from .modules import *


def _collate(samples):
    values = [s[0] for s in samples]
    if isinstance(values[0], np.ndarray):
        return np.stack(values)
    elif isinstance(values[0], torch.Tensor):
        return torch.stack(values)
    else:
        return torch.tensor(values)


class DelSumEmbedding(nn.Module):
    def __init__(self, n_head=6, head_width=64, atoks_width=None, length=2250, codes=1024, quantizers=8, pos_embs=None):
        super().__init__()
        self.length = length
        width = n_head * head_width
        if atoks_width is None: atoks_width = width
        self.width = width
        self.quantizers = quantizers

        emb = None
        embs = []
        for _ in range(quantizers):
            emb = FlexEmbeddings(codes, width, special_codes=2, frozen_width=atoks_width,
                                 special_embedding=emb and emb.special)
            embs.append(emb)
        self.embeddings = nn.ModuleList(embs)
        if pos_embs is not None:
            self.register_buffer("positional_embedding", pos_embs)

    def forward(self, toks, xenc):
        with record_function("embeddings"):
            b,_,n = toks.shape
            newn = min(n, self.length)

            embs = torch.zeros((b,newn,self.width), dtype=xenc.dtype, device=xenc.device)
            for i in range(self.quantizers):
                embs[:, :] += self.embeddings[i](toks[:,i,:])
            
            x = embs.to(xenc.dtype)
        return x


class DelSumHead(nn.Module):
    def __init__(self, quantizers=8, n_head=6, head_width=64):
        super().__init__()
        self.width = n_head * head_width
        self.quantizers = quantizers
        self.splitter = nn.Sequential(
            nn.Linear(self.width, self.width * quantizers),
            nn.GELU(),
        )

    def forward(self, x, embeddings=None):
        b, newn, _ = x.shape
        with record_function("splitter"):
            split = self.splitter(x).view(b,newn,self.quantizers,self.width)
        with record_function("unembed"):
            logits = torch.stack([embeddings[q].unembed(split[:,:,q]) for q in range(self.quantizers)], dim=1)
        return logits


@dataclasses.dataclass
class Tunables:
    init_std :float = 9
    embeddings_std :float = 0.2
    embeddings_lr_scale: float = 10
    output_mult :float = 5.6
    query_mult :float = .3
    encoder_depth_ratio :float = 0.25
    linear_heads :bool = False
    rope :bool = True
    q0_loss_mult: float = 1
    causal_encoder :bool = False

    lr0 :float = 3e-3
    clip_gradient_norm :float = 2
    weight_decay :float = 1e-3
    warmup_steps :float = 2000

    force_hidden_to_emb: bool = False

    @staticmethod
    def upgrade(args):
        args = {k:v for k,v in args.items()}
        def old_default(name, value):
            if name not in args: args[name] = value
        old_default('rope', False)
        old_default('linear_heads', True)
        old_default('causal_encoder', False)
        old_default('force_hidden_to_emb', True)
        for k in ['random', 'random_finetune']:
            args.pop(k, None)
        return args


class CategoricalEmbedding(nn.Module):
    default = torch.nan

    def __init__(self, codes, width=512):
        super().__init__()
        store_attr('codes,width')
        self.embed = nn.Embedding(codes+1, width)

    def forward(self, x):
        x[torch.isnan(x)] = self.codes
        return self.embed(x.to(torch.long))


class BinnedEmbedding(nn.Module):
    default = torch.nan

    def __init__(self, vmin=0, vmax=1, bins=32, width=512):
        super().__init__()
        store_attr('vmin,vmax,bins')
        self.embed = nn.Embedding(bins+1, width)
        
    def forward(self, x):
        qx = ((x - self.vmin) / (self.vmax - self.vmin) * self.bins).to(torch.long)
        qx.clamp_(0,self.bins-1)
        qx[torch.isnan(x)] = self.bins
        return self.embed(qx.to(torch.long))

    
class SpeakerEmbedding(nn.Module):
    def __init__(self, spk_width, width):
        super().__init__()
        store_attr('spk_width,width')

        self.default = torch.full((spk_width,), 0, dtype=torch.float16)
        self.spk_to_hidden = nn.Linear(spk_width, width) if spk_width != width else None

    def forward(self, x):
        x = F.normalize(x, dim=-1)
        if self.spk_to_hidden: x = self.spk_to_hidden(x.to(self.spk_to_hidden.weight.dtype))
        return x

        
class SADelARTransformer(nn.Module):
    def __init__(self, depth=3, ctx_n=2250,
                 stoks_len=750, stoks_codes=4097, stoks_width=None,
                 spk_width=None,
                 atoks_width=None,
                 n_head=3, head_width=64, ffn_mult=4,
                 quantizers=8, speaker_map={"1":0}, tunables=Tunables()):
        super().__init__()
        self.quantizers = quantizers
        self.codes = 1024
        width = n_head * head_width
        store_attr("depth,ctx_n,stoks_len,stoks_codes,stoks_width,spk_width,atoks_width,n_head,head_width,ffn_mult,quantizers,speaker_map")
        self.width = width
        self.base_width = 3 * head_width
        self.tunables = tunables

        # CUDA GRAPH SUPPORT
        self.use_cuda_graph = False
        self._cuda_graph = None
        self._static_toks = None
        self._static_positions = None
        self._static_output = None
        self._graph_batch_size = None
        self._graph_langs = None
        self._graph_xenc = None
        self._graph_xenc_positions = None
        self._graph_T = None
        self._graph_top_k = None

        if stoks_width is None: stoks_width = width
        if spk_width is None: spk_width = width
        self.emb_factor = width != stoks_width

        if tunables.rope:
            self.positional_embeddings = None
        else:
            self.register_buffer('positional_embeddings', sinusoids(ctx_n, width))

        self.cond_embeddings = nn.ModuleDict({
            'lang': CategoricalEmbedding(len(languages.languages), width),
            'snr': BinnedEmbedding(vmin=-10, vmax=70, bins=10, width=width),
            'c50': BinnedEmbedding(vmin=0, vmax=60, bins=6, width=width),
            'speaker': SpeakerEmbedding(spk_width, width),
        })

        self.semantic_embedding = nn.Embedding(stoks_codes, stoks_width)
        if self.emb_factor:
            self.emb_to_hidden = nn.Linear(stoks_width, width)
            if self.tunables.causal_encoder or self.tunables.force_hidden_to_emb:
                self.hidden_to_emb = nn.Linear(width, stoks_width)

        qk_scale = self.tunables.query_mult * 8 / math.sqrt(head_width)

        encoder_depth = int(depth * 2 * tunables.encoder_depth_ratio)
        decoder_depth = depth * 2 - encoder_depth
        self.encoder = nn.Sequential(*[
            ResidualAttentionBlock(width, n_head, qk_scale=qk_scale, ffn_mult=ffn_mult, rope=tunables.rope) for _ in range(encoder_depth)
        ])
        self.ln_post = LayerNorm(width)

        self.embds = DelSumEmbedding(
            pos_embs=self.positional_embeddings, length=ctx_n,
            n_head=n_head, head_width=head_width, atoks_width=atoks_width,
            quantizers=quantizers,
        )
        self.decoder = BaseDecoder(qk_scale=qk_scale, length=ctx_n,
                                     n_head=n_head, width=n_head * head_width, 
                                     ffn_mult=ffn_mult, depth=decoder_depth,
                                     rope=tunables.rope)
        self.head = DelSumHead(n_head=n_head, head_width=head_width, quantizers=quantizers)
        for l in self.decoder.layers:
            l.cross_attn.key_subsampling = 3

        self.register_buffer('val_true', torch.zeros(self.quantizers))
        self.register_buffer('val_total', torch.zeros(self.quantizers))
        self.apply(self.init_transformer)

    def setup(self, device):
        pass
        
    def load_frozen_semantic_embeddings(self, vqmodel):
        with torch.no_grad():
            self.semantic_embedding.weight[:] = vqmodel.rq.layers[0]._codebook.embed[0]
            self.semantic_embedding.lr_scale = 0

    def load_frozen_acoustic_embeddings(self, amodel):
        for i in range(self.quantizers):
            self.decoder.embeddings[i].set_frozen_embeddings(amodel.quantizer.vq.layers[i].codebook)
            
    def init_transformer(self, m):
        if isinstance(m, LinearHead):
            m.no_weight_decay = True
            torch.nn.init.constant_(m.weight, 0)
        elif isinstance(m, QueryHead):
            m.lr_scale = 1/(m.weight.shape[1] / self.base_width)
            torch.nn.init.constant_(m.weight, 0)
        elif isinstance(m, nn.Embedding):
            m.no_weight_decay = True
            m.lr_scale = self.tunables.embeddings_lr_scale
            std = self.tunables.embeddings_std
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-3*std, b=3*std)
        elif isinstance(m, nn.Linear):
            m.lr_scale = 1/(m.weight.shape[1] / self.base_width)
            std = self.tunables.init_std / m.weight.shape[1]
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-3*std, b=3*std)
            if m.bias is not None:
                torch.nn.init.trunc_normal_(m.bias, std=std, a=-3*std, b=3*std)
        elif isinstance(m, nn.LayerNorm):
            m.no_weight_decay = True
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)

    def embed_stoks(self, Stoks):
        b,n = Stoks.shape
        if self.stoks_len == 1500:
            x = Stoks.reshape(b,n//2,2)
            x = x.repeat_interleave(2, -1)[:,:,:3]
            x[:,:,1] = 1024
            x = x.reshape(b,n//2*3)
        else:
            x = Stoks
        Sembs = self.semantic_embedding(x.to(torch.long))
        if self.emb_factor:
            Sembs = self.emb_to_hidden(Sembs)
        return Sembs

    def _encoder(self, semb, positions):
        x = semb
        for l in self.encoder: x = l(x, positions, causal=self.tunables.causal_encoder)
        return self.ln_post(x)
    
    def run_encoder(self, Stoks, conds):
        bs = Stoks.shape[0]

        semb = self.embed_stoks(Stoks)
        with record_function("encoder"):
            if self.positional_embeddings is not None: semb = semb + self.positional_embeddings
            positions = torch.arange(0, semb.shape[1], device=semb.device)
            xenc = self._encoder(semb, positions)
        if self.training and self.tunables.causal_encoder:
            enc_logits = (self.hidden_to_emb(xenc) @ self.semantic_embedding.weight.to(xenc.dtype).T).float()
            enc_logits = enc_logits * self.tunables.output_mult / (self.width / self.base_width)
        else:
            enc_logits = None

        cond_embs = torch.zeros((bs,semb.shape[-1]), dtype=semb.dtype, device=semb.device)
        for k in self.cond_embeddings.keys():
            samples = [(x.get(k, self.cond_embeddings[k].default),) for x in conds]
            c = _collate(samples)
            if isinstance(c, np.ndarray): c = torch.tensor(c, device=Stoks.device)
            if isinstance(c, torch.Tensor): c = c.to(device=Stoks.device)
            cond_embs += self.cond_embeddings[k](c)

        return xenc + cond_embs.unsqueeze(1), positions, enc_logits

    def forward(self, Stoks, Atoks, conds, out_stoks=None, out_atoks=None, noloss=False, xenc=None, xenc_positions=None, atoks_positions=None):
        if xenc is None:
            Stoks, Atoks = [x.to(dtype=torch.long) for x in (Stoks, Atoks)]
            xenc, xenc_positions, enc_logits = self.run_encoder(Stoks, conds)
        with record_function("decoder"):
            embs = self.embds(Atoks, xenc)
            if atoks_positions is None: atoks_positions = torch.arange(0, embs.shape[1], device=embs.device)
            x = self.decoder(embs, atoks_positions, xenc, xenc_positions)
            logits = self.head(x, embeddings=self.embds.embeddings)
            logits *= self.tunables.output_mult / (self.width / self.base_width)

        if noloss:
            return logits

        with record_function("loss"):
            loss = 0
            for i in range(self.quantizers):
                loss += F.cross_entropy(logits[:,i,:-1].reshape(-1,logits.shape[-1]), Atoks[:,i,1:].reshape(-1), ignore_index=1024)
                if self.training and i == 0:
                    loss *= self.tunables.q0_loss_mult
            loss_denom = self.quantizers
            if self.training: loss_denom += - 1 + self.tunables.q0_loss_mult
            loss /= loss_denom
            if self.training and self.tunables.causal_encoder:
                loss += 0.1 * F.cross_entropy(enc_logits[:,:-1].transpose(-1,-2), Stoks[:,1:])

        if not self.training:
            for i in range(self.quantizers):
                Atoks_i = Atoks[:,i,1:]
                valid_Atoks = Atoks_i != 1024
                self.val_true[i] += (logits[:,i,:-1].argmax(-1)[valid_Atoks] == Atoks_i[valid_Atoks]).float().sum()
                self.val_total[i] += valid_Atoks.float().sum()

        return logits, loss

    def get_metrics(self):
        metrics = {
            f'acc_{i}':x.item() for i,x in enumerate(self.val_true / self.val_total)
        }
        self.val_true[:] = 0
        self.val_total[:] = 0
        return metrics

    @classmethod
    def load_model(cls, ref="WhisperSpeech/WhisperSpeech:s2a-q4-small-en+pl.model", spec=None, device=None, cache_dir=None):
        spec = inference.load_model(ref=ref, spec=spec, device=device, cache_dir=cache_dir)
        if '_extra_state' not in spec['state_dict'] and 'speaker_map' in spec['config']: spec['state_dict']['_extra_state'] = { 'speaker_map': spec['config']['speaker_map'] }
        model = cls(**spec['config'], tunables=Tunables(**Tunables.upgrade(spec['tunables'])))
        model.load_state_dict(spec['state_dict'])
        model.eval().to(device)
        return model

    def get_extra_state(self):
        return { 'speaker_map': self.speaker_map }

    def set_extra_state(self, st):
        self.speaker_map = st['speaker_map']

    def save_model(self, fname):
        torch.save(dict(config = self.__stored_args__,
                        tunables = dataclasses.asdict(self.tunables),
                        state_dict = self.state_dict()), fname)

    def switch_dtypes(self, dtype=torch.float16):
        self.dtype = dtype
        for n,m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.to(dtype)
            for bn,b in m.named_buffers(recurse=False):
                setattr(m,bn,b.to(dtype))

    def optimize(self, max_batch_size=1, dtype=torch.float16, torch_compile=True, use_cuda_graph=False):
        for emb in self.embds.embeddings:
            emb.convert_for_eval()
        for l in self.encoder:
            l.attn.convert_for_eval()
        for l in self.decoder.layers:
            l.attn.convert_for_eval()
            l.cross_attn.convert_for_eval()
            l.setup_kv_cache(max_batch_size, self.ctx_n, self.stoks_len)
        self.switch_dtypes(dtype)

        self.use_cuda_graph = use_cuda_graph

        if torch_compile:
            self.generate_next = torch.compile(self.generate_next, mode="reduce-overhead", fullgraph=True)

    # CUDA GRAPH HELPER METHODS
    def _init_cuda_graph_buffers(self, bs, dev):
        self._static_toks = torch.zeros((bs, self.quantizers, 1), dtype=torch.long, device=dev)
        self._static_positions = torch.zeros((1,), dtype=torch.long, device=dev)
        self._static_output = torch.zeros((bs, self.quantizers, 1), dtype=torch.long, device=dev)
        self._graph_batch_size = bs

    def _capture_cuda_graph(self, langs, xenc, xenc_positions, T, top_k):
        self._cuda_graph = None

        torch.cuda.synchronize()

        self._cuda_graph = torch.cuda.CUDAGraph()

        self._graph_langs = langs
        self._graph_xenc = xenc
        self._graph_xenc_positions = xenc_positions
        self._graph_T = T
        self._graph_top_k = top_k
        
        with torch.cuda.graph(self._cuda_graph):
            self._static_output = self.generate_one(
                self._static_toks,
                self._static_positions,
                self._graph_langs,
                self._graph_xenc,
                self._graph_xenc_positions,
                self._graph_T,
                self._graph_top_k
            )
        torch.cuda.synchronize()

    def _run_cuda_graph(self, toks, positions):
        self._static_toks.copy_(toks)
        self._static_positions.copy_(positions)
        self._cuda_graph.replay()
        return self._static_output.clone()

    def reset_cuda_graph(self):
        self._cuda_graph = None
        self._static_toks = None
        self._static_positions = None
        self._static_output = None
        self._graph_batch_size = None
        self._graph_langs = None
        self._graph_xenc = None
        self._graph_xenc_positions = None
        self._graph_T = None
        self._graph_top_k = None

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_one(self, toks, positions, langs, xenc, xenc_positions, T, top_k):
        probs = self(None, toks, None, langs, noloss=True, xenc=xenc, xenc_positions=xenc_positions, atoks_positions=positions)
        probs = probs[:,:,-1]
        return inference.sample(probs, T, top_k)

    def generate_next(self, *args, **kwargs):
        return self.generate_one(*args, **kwargs)

    @torch.no_grad()
    def generate(self, stoks, speakers, langs=None, atoks_prompt=None, N=None, bs=1, T=0.7, top_k=None, show_progress_bar=True, step=None, subsample_enc=False):
        dev = self.device
        N = N or len(stoks) * 3
        stoks = F.pad(stoks.to(dev), (1, self.stoks_len - len(stoks) - 1), value=self.stoks_codes-1).unsqueeze(0)
        speakers = speakers.to(device=dev, dtype=self.dtype)
        toks = torch.full((bs, self.quantizers, self.ctx_n), self.codes+1, dtype=torch.long, device=dev)
        T = torch.tensor(T, device=dev)

        start = 0
        if atoks_prompt is not None:
            start = atoks_prompt.shape[-1]
            for i in range(self.quantizers):
                toks[:,i,1+i:start+i+1] = atoks_prompt[:,i]
        start += 1

        with record_function("encode"):
            stoks, speakers = [x.repeat(bs, 1) for x in (stoks, speakers)]
            xenc, xenc_positions, _ = self.run_encoder(stoks, [dict(speaker=s, snr=60, c50=60) for s in speakers])
            toks_positions = torch.arange(N, device=dev)
        with record_function("prefill"):
            initial = self.generate_one(toks[:,:,:start], toks_positions[:start], langs, xenc, xenc_positions, T, top_k)
            toks[:,:start,start:start+1] = initial[:,:start]
            start += 1

        with inference.inference_context():
            it = range(start, min(N, self.ctx_n-1))
            if show_progress_bar: it = progress_bar(it)

            if self.use_cuda_graph and dev.type == 'cuda':
                if self._static_toks is None or self._graph_batch_size != bs:
                    self._init_cuda_graph_buffers(bs, dev)

                capture_pos = start
                self._static_toks.copy_(toks[:,:,capture_pos-1:capture_pos])
                self._static_positions.copy_(toks_positions[capture_pos-1:capture_pos])
                self._capture_cuda_graph(langs, xenc, xenc_positions, T, top_k)
                toks[:,:capture_pos,capture_pos:capture_pos+1] = self._static_output[:,:capture_pos]
                graph_start = capture_pos + 1

                for i in it:
                    if i < graph_start:
                        continue

                    self._static_toks.copy_(toks[:,:,i-1:i])
                    self._static_positions.copy_(toks_positions[i-1:i])
                    self._cuda_graph.replay()
                    toks[:,:i,i:i+1] = self._static_output[:,:i]
                    
                    if step is not None: step()
            else:
                for i in it:
                    with record_function("generate_one"):
                        toks[:,:i,i:i+1] = self.generate_next(toks[:,:,i-1:i], toks_positions[i-1:i], langs, xenc, xenc_positions, T, top_k)[:,:i]
                    if step is not None: step()

        toks = toks[:,:,1:N]
        for j in range(self.quantizers):
            toks[:, j] = torch.roll(toks[:, j], -j)
        return toks[:,:,:N-4]
