import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

import gradio as gr
import torch
import torchaudio
import whisper
from transformers import pipeline

from utils import load_model

# from trainer.utils import clean_whisper_text

device = "cuda" if torch.cuda.is_available() else "cpu"
Ichigo_name = "homebrewltd/ichigo-whisper:merge-medium-vi-2d-2560c-dim64.pth"
model_size = "merge-medium-vi-2d-2560c-dim64"
whisper_model_name = "medium"
language = "demo"

whisper_model = whisper.load_model(whisper_model_name)
whisper_model.to(device)

ichigo_model = load_model(ref=Ichigo_name, size=model_size)
ichigo_model.ensure_whisper(device, language)
ichigo_model.to(device)

phowhisper = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-large",
    device=device,
)


def transcribe_ichigo(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )
    wav, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    transcribe = ichigo_model.inference(wav.to(device))
    # return clean_whisper_text(transcribe[0].text)
    return transcribe[0].text


def transcribe_whisper(inputs):
    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please upload or record an audio file before submitting your request."
        )
    wav1, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav1 = torchaudio.functional.resample(wav1, sr, 16000)
    audio_sample = wav1.squeeze().float().numpy()
    transcribe = whisper_model.transcribe(
        audio_sample,
        task="transcribe",
        fp16=False,
    )
    # return clean_whisper_text(transcribe["text"])
    return transcribe["text"]


def transcribe_phowhisper(inputs):
    wav2, sr = torchaudio.load(inputs)
    if sr != 16000:
        wav2 = torchaudio.functional.resample(wav2, sr, 16000)
    audio_sample = wav2.squeeze().float().numpy()
    # return clean_whisper_text(phowhisper(audio_sample)["text"])
    return phowhisper(audio_sample)["text"]


with gr.Blocks(title="Ichigo Whisper Quantizer") as interface:
    gr.Markdown(
        "# Ichigo Whisper Quantizer: Enhanced Whisper Model for Low-Resource Languages Using Quantization"
    )
    gr.Markdown("Record your voice or upload audio and send it to the model.")
    gr.Markdown("Powered by [Homebrew Ltd](https://homebrew.ltd/)")
    with gr.Row():
        # Audio input section
        audio_input = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Audio Input"
        )

    with gr.Row():
        # Ichigo Model Column
        with gr.Column():
            gr.Markdown("### Ichigo Quantizer (Merged Codebook - Medium)")
            ichigo_output = gr.TextArea(
                label="Ichigo Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            ichigo_btn = gr.Button("Transcribe with Ichigo")
            ichigo_btn.click(
                fn=transcribe_ichigo, inputs=audio_input, outputs=ichigo_output
            )

        # Whisper Model Column
        with gr.Column():
            gr.Markdown(f"### Whisper {whisper_model_name.capitalize()}")
            whisper_output = gr.TextArea(
                label="Whisper Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            whisper_btn = gr.Button("Transcribe with Whisper")
            whisper_btn.click(
                fn=transcribe_whisper, inputs=audio_input, outputs=whisper_output
            )

        # PhoWhisper Model Column
        with gr.Column():
            gr.Markdown("### PhoWhisper Large")
            phowhisper_output = gr.TextArea(
                label="PhoWhisper Transcription",
                placeholder="Transcription will appear here...",
                lines=8,
            )
            phowhisper_btn = gr.Button("Transcribe with PhoWhisper")
            phowhisper_btn.click(
                fn=transcribe_phowhisper, inputs=audio_input, outputs=phowhisper_output
            )

    custom_css = """
        .gradio-container {
            font-family: 'Helvetica Neue', Arial, sans-serif;
        }
        .output-text {
            font-size: 16px;
            padding: 10px;
            border-radius: 4px;
            transition: all 0.3s ease;
        }
        .transcribe-button {
            margin-top: 10px;
        }
    """
    gr.HTML(f"<style>{custom_css}</style>")

if __name__ == "__main__":
    interface.queue()
    interface.launch("0.0.0.0", 7860, share=True)
