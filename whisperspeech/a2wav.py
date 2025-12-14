__all__ = ['Vocoder']

from vocos import Vocos
from whisperspeech2 import inference
import torch

class Vocoder:
    def __init__(self, repo_id="charactr/vocos-encodec-24khz", device=None, cache_dir=None):
        if device is None: device = inference.get_compute_device()
        if device == 'mps': device = 'cpu'
        self.device = device
        self.vocos = Vocos.from_pretrained(repo_id).to(device)

    def is_notebook(self):
        try:
            return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
        except:
            return False

    @torch.no_grad()
    def decode(self, atoks):
        if len(atoks.shape) == 3:
            b,q,t = atoks.shape
            atoks = atoks.permute(1,0,2)
        else:
            q,t = atoks.shape
        atoks = atoks.to(self.device)
        features = self.vocos.codes_to_features(atoks)
        bandwidth_id = torch.tensor({2: 0, 4: 1, 8: 2}[q]).to(self.device)
        return self.vocos.decode(features, bandwidth_id=bandwidth_id)

    def _save_audio(self, fname, audio_tensor, sample_rate=24000):
        try:
            import torchaudio
            torchaudio.save(fname, audio_tensor, sample_rate, backend="soundfile")
            return
        except (ImportError, RuntimeError, TypeError):
            pass
        
        try:
            import torchaudio
            torchaudio.save(fname, audio_tensor, sample_rate)
            return
        except (ImportError, RuntimeError):
            pass
        
        try:
            import soundfile as sf
            audio_np = audio_tensor.numpy().T
            sf.write(fname, audio_np, sample_rate)
            return
        except ImportError:
            pass
        
        raise ImportError(
            "No audio backend available. Please install either torchaudio or soundfile:\n"
            "  pip install torchaudio\n"
            "or\n"
            "  pip install soundfile"
        )

    def decode_to_file(self, fname, atoks):
        audio = self.decode(atoks)
        self._save_audio(fname, audio.cpu(), 24000)
        if self.is_notebook():
            from IPython.display import display, HTML, Audio
            display(HTML(f'<a href="{fname}" target="_blank">Listen to {fname}</a>'))

    def decode_to_notebook(self, atoks):
        from IPython.display import display, HTML, Audio
        audio = self.decode(atoks)
        display(Audio(audio.cpu().numpy(), rate=24000))
