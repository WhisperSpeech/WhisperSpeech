__all__ = ['Tunables', 'T2SEmbedding', 'Encoder', 'TSARTransformer']

import dataclasses
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from huggingface_hub import hf_hub_download
from fastcore.basics import store_attr
from fastprogress import progress_bar

from pathlib import Path

from whisperspeech2.modules import *
from whisperspeech2 import languages, inference


class CharTokenizer:
    eot = 0
    
    def encode(self, txt):
        return list(bytes(txt.strip(), 'utf-8'))

    def decode(self, tokens):
        return bytes(tokens).decode('utf-8')


@dataclasses.dataclass
class Tunables:
    init_std :float = 1
    embeddings_std :float = .01
    embeddings_lr_scale: float = 5
    embedding_projector_lr_scale: float = 2.5
    output_mult :float = .35
    query_mult :float = 1
    encoder_depth_ratio :float = 0.25
    causal_encoder: bool = True
    eot_dropout_p :float = .5
    cps_input: bool = True
    cps_bins: int = 32
    padding_token_offset: int = 0

    lr0 :float = 1.5e-3
    clip_gradient_norm :float = .2
    weight_decay :float = 1e-1
    warmup_steps :float = 4000

    @staticmethod
    def upgrade(args):
        args = {k:v for k,v in args.items()}
        def old_default(name, value):
            if name not in args: args[name] = value
        old_default('padding_token_offset', -1)
        for k in ['random', 'random_finetune']:
            args.pop(k, None)
        return args


class T2SEmbedding(nn.Module):
    def __init__(self, length=1500, codes=1024, width=384, pos_embs=None, stoks_width=384):
        super().__init__()
        self.embedding = FlexEmbeddings(codes, width, special_codes=1, frozen_width=stoks_width)
        if pos_embs is None: pos_embs = sinusoids(length, width)
        self.register_buffer("positional_embedding", pos_embs)

    def forward(self, Stoks, xenc, cps=None, offset=0):
        Sembs = self.embedding(Stoks)
        xin = (Sembs + self.positional_embedding[offset : offset + Sembs.shape[1]]).to(xenc.dtype)
        if cps is not None: xin = xin + cps
        return xin, offset


class Encoder(nn.Module):
    def __init__(self, depth=6, width=384, n_head=6, length=1500, codes=1024, emb_width=384, ffn_mult=4, pos_embs=None, tunables=Tunables()):
        super().__init__()
        self.emb_width = emb_width
        self.tunables = tunables

        self.embedding = FlexEmbeddings(codes, width, frozen_width=emb_width)

        if pos_embs is None: pos_embs = sinusoids(length, width)
        self.register_buffer("positional_embedding", pos_embs)

        self.layers = nn.ModuleList([
            ResidualAttentionBlock(width, n_head,
                                   qk_scale=tunables.query_mult*8/math.sqrt(width/n_head), ffn_mult=ffn_mult) for _ in range(depth)
        ])

        self.ln_post = LayerNorm(width)

        mask = torch.empty(length, length).fill_(-torch.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, Stoks, positions, lang_emb=None):
        xin = self.embedding(Stoks)

        if lang_emb is not None: xin = xin + lang_emb

        x = (xin +
             self.positional_embedding[positions]).to(xin.dtype)

        for l in self.layers: x = l(x, positions,
                                    causal=self.tunables.causal_encoder and self.training,
                                    mask=self.mask if self.tunables.causal_encoder and not self.training else None)

        return self.ln_post(x)


class TSARTransformer(nn.Module):
    def __init__(self, depth=6, n_head=6, head_width=64, ffn_mult=4,
                 ttoks_len=200, ttoks_codes=256, ttoks_width=None,
                 stoks_len=1500, stoks_codes=1024, stoks_width=None,
                 tunables=Tunables()):
        super().__init__()
        store_attr("depth,n_head,head_width,ffn_mult,stoks_width,ttoks_width,ttoks_len,stoks_len,ttoks_codes,stoks_codes")

        width = n_head * head_width
        self.width = width
        self.base_width = 3 * head_width
        self.tunables = tunables
        if self.stoks_width is None: self.stoks_width = self.width
        if self.ttoks_width is None: self.ttoks_width = self.width

        self.lang_embeddings = nn.Embedding(len(languages.languages), width)
        if tunables.cps_input:
            self.cps_embeddings = nn.Embedding(tunables.cps_bins, self.width)
        else:
            self.cps_embeddings = None        

        encoder_depth = int(depth * 2 * tunables.encoder_depth_ratio)
        decoder_depth = depth * 2 - encoder_depth
        tformer_args = dict(width=width, n_head=n_head, ffn_mult=ffn_mult, tunables=tunables)
        self.encoder = Encoder(length=ttoks_len, codes=ttoks_codes, emb_width=self.ttoks_width, depth=encoder_depth, **tformer_args)
        self.embeddings = T2SEmbedding(length=stoks_len, codes=stoks_codes, width=width, stoks_width=self.stoks_width)

        self.decoder = BaseDecoder(
            length=stoks_len, 
            depth=decoder_depth,
            qk_scale=tunables.query_mult*8/math.sqrt(width/n_head),
            width=width, n_head=n_head, ffn_mult=ffn_mult,
        )
        self.tokenizer = None

        # CUDA GRAPH SUPPORT
        self.use_cuda_graph = False
        self._cuda_graph = None
        self._static_toks = None
        self._static_positions = None
        self._static_output = None

        self.apply(self.init_transformer)

    def load_frozen_semantic_embeddings(self, vqmodel):
        self.embeddings.embedding.set_frozen_embeddings(vqmodel.rq.layers[0]._codebook.embed[0])

    def setup(self, device):
        pass

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
        elif isinstance(m, EmbeddingProjector):
            m.lr_scale = self.tunables.embedding_projector_lr_scale
            std = self.tunables.init_std
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

    def _embed_cps(self, cpss):
        if self.cps_embeddings is None: return None

        cps_bin = (cpss / 20 * self.tunables.cps_bins).to(torch.long)
        cps_bin[cps_bin >= self.tunables.cps_bins] = self.tunables.cps_bins-1
        return self.cps_embeddings(cps_bin).unsqueeze(1)

    def run_encoder(self, in_ttoks, languages, cpss):
        if len(languages.shape) != 3: lang_embs = self.lang_embeddings(languages)
        else: lang_embs = languages
        if len(lang_embs.shape) == 2: lang_embs = lang_embs.unsqueeze(1)

        cps_emb = self._embed_cps(cpss)

        with record_function("encoder"):
            positions = torch.arange(0, in_ttoks.shape[1], device=in_ttoks.device)
            xenc = self.encoder(in_ttoks.to(torch.long), positions, lang_emb=lang_embs)

        return xenc, positions, cps_emb

    def forward(self, in_ttoks, out_ttoks, languages, cpss, in_stoks, out_stoks=None, in_stoks_positions=None, loss=True, offset=None, xenc=None, xenc_positions=None, cps_emb=None):
        if xenc is None:
            xenc, xenc_positions, cps_emb = self.run_encoder(in_ttoks, languages, cpss)

        with record_function("decoder"):
            x = (self.embeddings.embedding(in_stoks) + 
                 self.embeddings.positional_embedding[in_stoks_positions] +
                 cps_emb).to(xenc[0].dtype)
            x = self.decoder(x, in_stoks_positions, xenc.clone(), xenc_positions)
            logits = self.embeddings.embedding.unembed(x)
            logits = logits * self.tunables.output_mult / (self.width / self.base_width)

        if loss is not None:
            with record_function("loss"):
                loss = F.cross_entropy(logits.transpose(-1,-2), out_stoks)
                if self.training and self.tunables.causal_encoder:
                    enc_logits = self.encoder.embedding.unembed(xenc)
                    enc_logits = enc_logits * self.tunables.output_mult / (self.width / self.base_width)
                    loss += 0.1 * F.cross_entropy(enc_logits.transpose(-1,-2), out_ttoks)

        return logits, loss

    @classmethod
    def load_model(cls, ref="WhisperSpeech/WhisperSpeech:t2s-small-en+pl.model",
                   repo_id=None, filename=None, local_filename=None, spec=None, device=None, cache_dir=None):
        if repo_id is None and filename is None and local_filename is None and spec is None:
            if ":" in ref:
                repo_id, filename = ref.split(":", 1)
            else:
                local_filename = ref
        if not local_filename and spec is None:
            local_filename = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        if spec is None:
            spec = torch.load(local_filename, map_location=device)
        model = cls(**spec['config'], tunables=Tunables(**Tunables.upgrade(spec['tunables'])))
        model.load_state_dict(spec['state_dict'])
        model.eval().to(device)
        return model

    def save_model(self, fname):
        torch.save(dict(config = self.__stored_args__,
                        tunables = dataclasses.asdict(self.tunables),
                        state_dict = self.state_dict()), fname)

    def ensure_tokenizer(self):
        assert not self.training
        if self.tokenizer is None: self.tokenizer = CharTokenizer()

    def switch_dtypes(self, dtype=torch.float16):
        self.dtype = dtype
        for n,m in self.named_modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.to(dtype)
            for bn,b in m.named_buffers(recurse=False):
                setattr(m,bn,b.to(dtype))

    # ADDED use_cuda_graph PARAMETER
    def optimize(self, max_batch_size=1, dtype=torch.float16, torch_compile=True, use_cuda_graph=False):
        for emb in [self.encoder.embedding, self.embeddings.embedding]:
            emb.convert_for_eval()
        for l in self.encoder.layers:
            l.attn.convert_for_eval()
        for l in self.decoder.layers:
            l.attn.convert_for_eval()
            l.cross_attn.convert_for_eval()
            l.setup_kv_cache(max_batch_size, self.stoks_len, self.ttoks_len)
        self.switch_dtypes(dtype)

        self.use_cuda_graph = use_cuda_graph

        if torch_compile:
            self.generate_next = torch.compile(self.generate_next, mode="reduce-overhead", fullgraph=True)

    @property
    def device(self):
        return next(self.parameters()).device

    def generate_one(self, toks, toks_positions, cps_emb, xenc, xenc_positions, T, top_k):
        probs, _ = self(None, None, None, None, toks, in_stoks_positions=toks_positions, loss=None, xenc=xenc, xenc_positions=xenc_positions, cps_emb=cps_emb)
        probs = probs[:,-1]
        probs[self.embeddings.embedding.codes:] = -torch.inf
        return inference.sample(probs, T, top_k)

    def generate_next(self, *args, **kwargs):
        return self.generate_one(*args, **kwargs)

    # CUDA GRAPH HELPER METHODS
    def _init_cuda_graph_buffers(self, bs, dev):
        self._static_toks = torch.zeros((bs, 1), dtype=torch.long, device=dev)
        self._static_positions = torch.zeros((1,), dtype=torch.long, device=dev)
        self._static_output = torch.zeros((bs, 1), dtype=torch.long, device=dev)
        self._graph_batch_size = bs
    
    def _capture_cuda_graph(self, cps_emb, xenc, xenc_positions, T, top_k):
        self._cuda_graph = None

        torch.cuda.synchronize()

        self._cuda_graph = torch.cuda.CUDAGraph()

        self._graph_cps_emb = cps_emb
        self._graph_xenc = xenc
        self._graph_xenc_positions = xenc_positions
        self._graph_T = T
        self._graph_top_k = top_k

        with torch.cuda.graph(self._cuda_graph):
            self._static_output = self.generate_one(
                self._static_toks,
                self._static_positions,
                self._graph_cps_emb,
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
        self._graph_cps_emb = None
        self._graph_xenc = None
        self._graph_xenc_positions = None
        self._graph_T = None
        self._graph_top_k = None

    @torch.no_grad()
    def generate(self, txt, cps=15, lang="en", stoks_prompt=None, N=None, bs=1, T=0.7, top_k=None, step=None, show_progress_bar=True):
        self.ensure_tokenizer()
        N = N or self.stoks_len
        dev = self.device
        ttoks = []
        langs = []
        if isinstance(lang, list):
            lang0 = lang[0]
            assert isinstance(txt, list), "lang and txt have to be both lists or strings"
            for txt, lang in zip(txt, lang):
                tt = self.tokenizer.encode(txt)
                ttoks += tt
                langs += [languages.to_id(lang)] * len(tt)
        elif isinstance(lang, torch.Tensor):
            langs = lang
            ttoks = self.tokenizer.encode(txt)
        else:
            lang0 = lang
            ttoks = self.tokenizer.encode(txt)
            langs = torch.tensor([languages.to_id(lang)], device=dev)
        ttoks = torch.tensor(ttoks, device=dev)
        ttoks = F.pad(ttoks, (1, self.ttoks_len - len(ttoks) - 1), value=self.tokenizer.eot)
        cpss = torch.tensor([cps], device=dev)
        T = torch.tensor(T, device=dev)
        if not isinstance(langs, torch.Tensor):
            langs = torch.tensor(langs, device=dev)
            langs = F.pad(langs, (1, self.ttoks_len - len(langs) - 1), value=languages.to_id(lang0))

        toks = torch.zeros((bs, N), dtype=torch.long, device=dev)
        toks[:, 0] = self.stoks_codes + self.tunables.padding_token_offset
        start = 0
        if stoks_prompt is not None:
            toks[:, 1:len(stoks_prompt)+1] = stoks_prompt
            start = len(stoks_prompt)

        toks_positions = torch.arange(N, device=dev)
        
        with record_function("encode"):
            ttoks = ttoks.repeat(bs, 1)
            langs, cpss = [x.repeat(bs) for x in (langs, cpss)]
            xenc, xenc_positions, cps_emb = self.run_encoder(ttoks, langs, cpss)

        with record_function("prefill"):
            toks[:, start+1] = self.generate_one(
                toks[:, :start+1].contiguous(), 
                toks_positions[:start+1], 
                cps_emb, xenc, xenc_positions, T, top_k
            )[:, 0]

        it = range(start + 1, N - 1)
        if show_progress_bar: 
            it = progress_bar(it)

        with inference.inference_context():
            if self.use_cuda_graph and dev.type == 'cuda':
                if self._static_toks is None or self._graph_batch_size != bs:
                    self._init_cuda_graph_buffers(bs, dev)

                capture_pos = start + 1
                self._static_toks.copy_(toks[:, capture_pos:capture_pos+1])
                self._static_positions.copy_(toks_positions[capture_pos:capture_pos+1])
                self._capture_cuda_graph(cps_emb, xenc, xenc_positions, T, top_k)
                toks[:, capture_pos+1] = self._static_output[:, 0]
                graph_start = capture_pos + 1

                for i in it:
                    if i < graph_start:
                        continue

                    self._static_toks.copy_(toks[:, i:i+1])
                    self._static_positions.copy_(toks_positions[i:i+1])
                    self._cuda_graph.replay()
                    toks[:, i+1] = self._static_output[:, 0]

                    if (toks[:, i+1] == self.stoks_codes + self.tunables.padding_token_offset).all():
                        return toks[:, 1:i+2]
                    if step is not None: 
                        step()
            else:
                for i in it:
                    toks[:, i+1] = self.generate_next(
                        toks[:, i:i+1], 
                        toks_positions[i:i+1], 
                        cps_emb, xenc, xenc_positions, T, top_k
                    )[:, 0]
                    if (toks[:, i+1] == self.stoks_codes + self.tunables.padding_token_offset).all():
                        return toks[:, 1:i+2]
                    if step is not None: 
                        step()

        return toks[:, 1:]
