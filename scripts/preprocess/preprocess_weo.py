import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio.functional import resample
from tqdm import tqdm

from whisper.functionals import load_model, pred_ppg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav-dirpath",
        required=True,
        help="Directory to load the wav files",
        dest="wav_dirpath",
        type=Path,
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["whisper_large_v2", "hubert"],
        dest="mode",
    )
    parser.add_argument(
        "--glob",
        default="**/*.wav",
        help="Pattern of wav files to process under wav-dirpath, e.g., **/*.el_hubert.wav",
    )
    args = parser.parse_args()

    if args.mode == "whisper_large_v2":
        model = load_model("checkpoints/whisper/large-v2.pt")
    elif args.mode == "hubert":
        model = torch.hub.load("bshall/hubert:main", f"hubert_soft").cuda().eval()  # type: ignore

    filepaths = list(args.wav_dirpath.glob(args.glob))
    for wav_filepath in tqdm(filepaths, desc="Processing WAV files"):
        out_filepath = wav_filepath.with_suffix(f".ppg_{args.mode}.npy")
        if out_filepath.exists():
            continue
        if args.mode == "whisper_large_v2":
            pred_ppg(model, str(wav_filepath), str(out_filepath))
        elif args.mode == "hubert":
            wav, sr = torchaudio.load(wav_filepath)
            wav = resample(wav, sr, 16000)
            wav = wav.unsqueeze(0).cuda()
            with torch.inference_mode():
                units = model.units(wav).squeeze().cpu().numpy()  # type: ignore
            np.save(out_filepath, units)
