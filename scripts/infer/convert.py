import argparse
import logging
import os
import time

import librosa
import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write
from torchaudio.functional import resample
from tqdm import tqdm

from qvc.mel_processing import mel_spectrogram_torch
from qvc.models import SynthesizerTrn
from qvc.utils import get_hparams_from_file, load_checkpoint
from whisper.functionals import load_model, pred_ppg_infer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpfile",
        type=str,
        help="path to json config file",
        required=True,
    )
    parser.add_argument(
        "--ptfile",
        type=str,
        help="path to pth file",
        required=True,
    )
    parser.add_argument("--txtpath", type=str, help="path to txt file", required=True)
    parser.add_argument("--outdir", type=str, help="path to output dir", required=True)
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu (default: cuda if available)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)  # type: ignore
    hps = get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,  # type: ignore
        hps.train.segment_size // hps.data.hop_length,  # type: ignore
        **hps.model,  # type: ignore
    ).to(args.device)
    _ = net_g.eval()
    total = sum([param.nelement() for param in net_g.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    print("Loading checkpoint...")
    _ = load_checkpoint(args.ptfile, net_g, None)

    ssl_mode = hps.model.ssl_mode  # type: ignore
    if ssl_mode == "hubert":
        ssl_model = torch.hub.load("bshall/hubert:main", f"hubert_soft").to(args.device).eval()  # type: ignore
    elif ssl_mode == "whisper_large_v2":
        ssl_model = load_model("checkpoints/whisper/large-v2.pt", args.device)
    else:
        raise ValueError(f"Invalid ssl_mode: {ssl_mode}")

    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            rawline = rawline.strip()
            if rawline.startswith("#"):
                continue
            title, src, tgt = rawline.split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")

    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts), total=len(titles), leave=False):
            title, src, tgt = line

            st = time.perf_counter()

            # Src input
            if ssl_mode == "whisper_large_v2":
                weo = torch.from_numpy(pred_ppg_infer(ssl_model, src))
            elif ssl_mode == "hubert":
                wav, sr = torchaudio.load(src)
                if wav.shape[0] != 1:
                    # stereo to mono
                    wav = wav.mean(dim=0, keepdim=True)
                wav = resample(wav, sr, 16000)
                wav = wav.to(args.device).unsqueeze(0)
                weo = ssl_model.units(wav).squeeze(0)  # type: ignore
            weo = weo.transpose(1, 0)
            c = weo.to(args.device).unsqueeze(0)

            # Target input
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)  # type: ignore
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(args.device)
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
            audio = audio[0][0].data.cpu().float().numpy()

            print(
                f"Audio length: {len(audio) / 16000:.2f} | Elapsed time: {time.perf_counter() - st}"
            )

            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(
                    os.path.join(args.outdir, "{}.wav".format(timestamp + "_" + title)),
                    hps.data.sampling_rate,  # type: ignore
                    audio,
                )
            else:
                write(
                    os.path.join(args.outdir, f"{title}.wav"),
                    hps.data.sampling_rate,  # type: ignore
                    audio,
                )
