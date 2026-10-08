import os

import numpy as np
import torch

from whisper.audio import load_audio, log_mel_spectrogram, pad_or_trim
from whisper.model import ModelDimensions, Whisper


def load_model(path, device=None) -> Whisper:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location="cpu")
    dims = ModelDimensions(**checkpoint["dims"])
    print(device, dims)
    model = Whisper(dims)
    del model.decoder
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    if device != "cpu":
        model.half()
    model.to(device)
    return model


def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio = load_audio(wavPath, sr=16000)
    audln = audio.shape[0]
    ppgln = audln // 320
    audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).half().to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        # print(ppg.shape)
        ppg = ppg[:ppgln,]  # [length, dim=1024]
        # print(ppg.shape)
        # print(ppg.shape)
        os.makedirs(
            os.path.dirname(ppgPath), exist_ok=True
        )  # Create the directory if it doesn't exist
        np.save(ppgPath, ppg, allow_pickle=False)


def pred_ppg_infer(whisper: Whisper, wavPath):
    audio = load_audio(wavPath, sr=16000)
    audln = audio.shape[0]
    ppgln = audln // 320
    audio = pad_or_trim(audio)
    dtype = next(whisper.parameters()).dtype
    mel = log_mel_spectrogram(audio).to(whisper.device, dtype)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        ppg = ppg[:ppgln,]  # [length, dim=1024]
    return ppg
