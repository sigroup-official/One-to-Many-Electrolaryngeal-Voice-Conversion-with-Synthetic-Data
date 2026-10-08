import argparse
from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

from qvc.mel_processing import mel_spectrogram_torch
from qvc.models import SynthesizerTrn
from qvc.utils import get_hparams_from_file, load_checkpoint
from whisper.functionals import load_model, pred_ppg_infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav-dirpath",
        required=True,
        help="Directory to load the wav files",
        type=Path,
    )
    parser.add_argument(
        "--hpfile",
        required=True,
        help="Path to the hp file",
        type=Path,
    )
    parser.add_argument(
        "--ptfile",
        required=True,
        help="Path to the pth file",
        dest="ptfile",
        type=Path,
    )
    parser.add_argument(
        "--tgt-el-filepath",
        help="Path to the el file",
        default="data/carlos_jp_el100/carlos_jp_el100_wav_preprocessed/001.wav",
        type=Path,
    )
    parser.add_argument(
        "--suffix",
        help="Suffix of output files, e.g., X.wav -> X{suffix}.wav",
        default=".el",
    )
    args = parser.parse_args()
    hps = get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,  # type: ignore
        hps.train.segment_size // hps.data.hop_length,  # type: ignore
        **hps.model,  # type: ignore
    ).cuda()
    _ = net_g.eval()
    _ = load_checkpoint(args.ptfile, net_g, None)
    ssl_mode = hps.model.ssl_mode  # type: ignore
    if ssl_mode == "hubert":
        ssl_model = torch.hub.load("bshall/hubert:main", f"hubert_soft").cuda().eval()  # type: ignore
    elif ssl_mode == "whisper_large_v2":
        ssl_model = load_model("checkpoints/whisper/large-v2.pt")
    else:
        raise ValueError(f"Invalid ssl_mode: {ssl_mode}")

    # Only convert source wavs, skip derived ones (e.g., X.el.wav)
    filepaths = [p for p in args.wav_dirpath.glob("**/*.wav") if "." not in p.stem]
    for wav_filepath in tqdm(filepaths, desc="Processing WAV files"):
        out_filepath = wav_filepath.with_suffix(f"{args.suffix}.wav")
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        if out_filepath.exists():
            continue

        with torch.no_grad():
            # Src input
            if ssl_mode == "whisper_large_v2":
                weo = torch.from_numpy(pred_ppg_infer(ssl_model, wav_filepath))
            elif ssl_mode == "hubert":
                wav, sr = torchaudio.load(wav_filepath)
                if wav.shape[0] != 1:
                    # stereo to mono
                    wav = wav.mean(dim=0, keepdim=True)
                wav = resample(wav, sr, 16000)
                wav = wav.cuda().unsqueeze(0)
                weo = ssl_model.units(wav).squeeze(0)  # type: ignore
            weo = weo.transpose(1, 0)
            c = weo.cuda().unsqueeze(0)
            # Target input
            wav_tgt, _ = librosa.load(args.tgt_el_filepath, sr=hps.data.sampling_rate)  # type: ignore
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            mel_tgt = mel_spectrogram_torch(
                wav_tgt,
                hps.data.filter_length,  # type: ignore
                hps.data.n_mel_channels,  # type: ignore
                hps.data.sampling_rate,  # type: ignore
                hps.data.hop_length,  # type: ignore
                hps.data.win_length,  # type: ignore
                hps.data.mel_fmin,  # type: ignore
                hps.data.mel_fmax,  # type: ignore
            )
            # Synthesize
            audio = net_g.infer(c, mel=mel_tgt)
            audio = audio[0][0].data.cpu().float()
            # Save
            torchaudio.save(out_filepath, audio, hps.data.sampling_rate)  # type: ignore
