"""Create flat-F0 pseudo EL speech with the WORLD vocoder (the flat-F0 baseline).

Voiced F0 is set to a constant (100 Hz), aperiodicity is capped, and shaped noise
is added. Outputs are written next to the inputs as X{suffix}.wav (default
X.el_flat.wav), so they can be used for EL2NL training via `data.el_suffix`.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
from tqdm import tqdm


def flatten_f0(audio_filepath, out_filepath, sr: int = 16000, f0: float = 100.0, noise_scale=2.0,
               n_fft=1024, hop_length=256, win="hann"):

    try:
        audio, _ = librosa.load(audio_filepath, sr=sr)
        T = audio.shape[0]

        # WORLD analysis
        f0_track, t = pw.harvest(audio.astype(np.float64), sr)
        sp = pw.cheaptrick(audio.astype(np.float64), f0_track, t, sr)
        ap = pw.d4c(audio.astype(np.float64), f0_track, t, sr)

        # Flatten F0
        voiced = f0_track > 0
        f0_flat = f0_track.copy()
        f0_flat[voiced] = f0

        # Flatten aperiodicity
        ap_flat = np.minimum(ap, 0.2)

        # Synthesize
        y_flat = pw.synthesize(f0_flat, sp, ap_flat, sr).astype(np.float32)

        # ---------- STFT-domain shaped noise ----------
        win_fn = librosa.filters.get_window(win, n_fft, fftbins=True)
        S_signal = librosa.stft(y_flat, n_fft=n_fft, hop_length=hop_length, window=win_fn, center=True)
        n_bins, n_cols = S_signal.shape

        rng = np.random.default_rng(0)
        phase = rng.random((n_bins, n_cols)) * 2*np.pi

        # Frequency vector for weighting
        freqs = np.linspace(0, sr/2, n_bins)
        # Avoid divide by zero at DC
        weights = np.ones_like(freqs)
        weights[1:] = 1.0 / np.power(freqs[1:], 0.5)   # ~compensates for bin bandwidth
        weights /= np.max(weights)               # normalize to 1 at lowest nonzero freq

        # Broadcast across time frames
        amp = noise_scale * weights[:, None]
        S_noise = amp * np.exp(1j * phase)

        # Remove DC/Nyquist lines if you don’t want them
        S_noise[0, :] = 0.0
        if n_fft % 2 == 0:
            S_noise[-1, :] = 0.0

        # Invert
        n_noise = librosa.istft(S_noise, hop_length=hop_length, window=win_fn, length=len(y_flat))

        out = (y_flat + n_noise.astype(np.float32))[:T]

        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_filepath), out, 16000)

    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dirpath", type=Path, default="data/jvs/jvs_wav_preprocessed")
    parser.add_argument("--suffix", default=".el_flat", help="X.wav -> X{suffix}.wav")
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() - 2)
    args = parser.parse_args()

    audio_filepaths, out_filepaths = [], []
    # Only source wavs, skip derived ones (e.g., X.el.wav)
    for audio_filepath in sorted(args.wav_dirpath.glob('*/*.wav')):
        if "." in audio_filepath.stem:
            continue
        out_filepath = audio_filepath.with_suffix(f"{args.suffix}.wav")
        if out_filepath.exists():
            continue
        audio_filepaths.append(audio_filepath)
        out_filepaths.append(out_filepath)

    with ProcessPoolExecutor(args.n_workers) as pool:
        futures = [pool.submit(flatten_f0, audio_fp, out_fp) for audio_fp, out_fp in zip(audio_filepaths, out_filepaths)]
        results = list(tqdm(as_completed(futures), total=len(futures)))

    for future in results:
        future.result()


if __name__ == '__main__':
    main()
