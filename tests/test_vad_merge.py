import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_chunk_merger():
    for name in (
        "webdataset",
        "fastprogress",
        "fastcore",
        "fastcore.script",
        "torchaudio",
        "huggingface_hub",
    ):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["fastprogress"].progress_bar = lambda items, **kwargs: items
    sys.modules["fastcore.script"].call_parse = lambda fn: fn
    shardlists = types.ModuleType("webdataset.shardlists")
    shardlists.single_node_only = lambda *args, **kwargs: None
    sys.modules["webdataset"].shardlists = shardlists

    pkg = types.ModuleType("whisperspeech")
    pkg.__path__ = [str(ROOT / "whisperspeech")]
    pkg.__package__ = "whisperspeech"
    sys.modules["whisperspeech"] = pkg
    sys.modules["whisperspeech.utils"] = types.ModuleType("whisperspeech.utils")

    spec = importlib.util.spec_from_file_location(
        "whisperspeech.vad_merge",
        ROOT / "whisperspeech" / "vad_merge.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["whisperspeech.vad_merge"] = mod
    spec.loader.exec_module(mod)
    return mod.chunk_merger


def sample(segments, speakers):
    return {
        "__key__": "clip",
        "vad.npy": np.array(segments, dtype=np.float32),
        "spk_emb.npy": np.stack(speakers).astype(np.float32),
    }


def test_a_new_speaker_is_not_divided_by_the_previous_count():
    chunk_merger = load_chunk_merger()
    A = np.array([1.0, 0.0], dtype=np.float32)
    B = np.array([0.0, 1.0], dtype=np.float32)
    merger = chunk_merger("mvad", should_cut=lambda secs: False)
    out = next(merger([sample([(0, 3), (3, 6), (6, 9), (9, 12)], [A, A, A, B])]))
    spk = out["mvad.spk_emb.npy"]
    assert len(spk) == 2
    torch.testing.assert_close(torch.tensor(spk[0]), torch.tensor(A))
    torch.testing.assert_close(torch.tensor(spk[1]), torch.tensor(B))


def test_one_speaker_stays_one_chunk():
    chunk_merger = load_chunk_merger()
    A = np.array([1.0, 0.0], dtype=np.float32)
    merger = chunk_merger("mvad", should_cut=lambda secs: False)
    out = next(merger([sample([(0, 3), (3, 6)], [A, A])]))
    spk = out["mvad.spk_emb.npy"]
    assert len(spk) == 1
    torch.testing.assert_close(torch.tensor(spk[0]), torch.tensor(A))
