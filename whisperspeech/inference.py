__all__ = ['get_compute_device']

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from contextlib import nullcontext

def get_default_compute_device():
    if torch.cuda.is_available() and (torch.version.cuda or torch.version.hip):
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'

preferred_device = None

def get_compute_device():
    global preferred_device
    if preferred_device is None: preferred_device = get_default_compute_device()
    return preferred_device

def load_model(ref=None, spec=None, device='cpu', cache_dir=None):
    if spec is not None: return spec
    if ":" in ref:
        repo_id, filename = ref.split(":", 1)
        local_filename = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    else:
        local_filename = ref
    return torch.load(local_filename, map_location=device)

def inference_context():
    if torch.cuda.is_available():
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    else:
        return nullcontext()

def multinomial_sample_one_no_sync(probs_sort):
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, T=1.0, top_k=None):
    # Convert T to tensor if needed (required for CUDA graph compatibility)
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, device=logits.device, dtype=logits.dtype)
    # Use torch.clamp instead of Python's max() for CUDA graph compatibility
    logits = logits / torch.clamp(T, min=1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, T=1.0, top_k=None):
    probs = logits_to_probs(logits, T, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next
