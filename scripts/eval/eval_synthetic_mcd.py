"""MCD between real EL speech and synthetic EL speech of the same sentence.

For JVS speakers 1, 21, 41, 61, 81 and the 100 EL utterances XXX.wav, compute
MCD(real EL, synthetic EL jvs{sid}/VOICEACTRESS100_XXX{syn_suffix}.wav).
MCD(real EL, original NL) is also reported as a reference.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from mel_cepstral_distance import compare_audio_files
from tqdm import tqdm


def compute_mcds(el_filepath, nl_filepath, syn_filepath):
    try:
        nl_mcd, _ = compare_audio_files(el_filepath, nl_filepath)
        syn_mcd, _ = compare_audio_files(el_filepath, syn_filepath)
        return nl_mcd, syn_mcd
    except Exception as e:
        print(e)
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--el-dirpath", type=Path, default="data/carlos_jp_el100/carlos_jp_el100_wav_preprocessed"
    )
    parser.add_argument("--jvs-dirpath", type=Path, default="data/jvs/jvs_wav_preprocessed")
    parser.add_argument(
        "--syn-suffix", default=".el_whisper", help="Synthetic EL is X{syn_suffix}.wav"
    )
    parser.add_argument("--out-filepath", type=Path, required=True)
    args = parser.parse_args()

    el_filepaths, nl_filepaths, syn_filepaths = [], [], []
    for sid in range(1, 101, 20):
        for el_filepath in sorted(args.el_dirpath.glob("*.wav")):
            nl_filepath = args.jvs_dirpath / f"jvs{sid:03d}/VOICEACTRESS100_{el_filepath.stem}.wav"
            el_filepaths.append(el_filepath)
            nl_filepaths.append(nl_filepath)
            syn_filepaths.append(nl_filepath.with_suffix(f"{args.syn_suffix}.wav"))

    with ProcessPoolExecutor(os.cpu_count()) as pool:
        futures = [
            pool.submit(compute_mcds, el_fp, nl_fp, syn_fp)
            for el_fp, nl_fp, syn_fp in zip(el_filepaths, nl_filepaths, syn_filepaths)
        ]
        results = list(tqdm(as_completed(futures), total=len(futures)))

    nl_mcds, syn_mcds = [], []
    for future in results:
        result = future.result()
        if result is None:
            continue
        nl_mcd, syn_mcd = result
        nl_mcds.append(nl_mcd)
        syn_mcds.append(syn_mcd)

    result_str = f"# syn_suffix={args.syn_suffix}, n={len(syn_mcds)}/{len(futures)}\n"
    result_str += f"nl mcd, avg={np.mean(nl_mcds):.3f}, std={np.std(nl_mcds):.3f}\n"
    result_str += f"syn mcd, avg={np.mean(syn_mcds):.3f}, std={np.std(syn_mcds):.3f}\n"
    print(result_str)
    args.out_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_filepath, "w") as f:
        f.write(result_str)


if __name__ == "__main__":
    main()
