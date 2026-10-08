import argparse
from pathlib import Path

import torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dirpath", type=Path, default="data/carlos_jp_el100")
    parser.add_argument(
        "--out-dirpath",
        type=Path,
        default="data/carlos_jp_el100/carlos_jp_el100_wav_preprocessed",
    )
    parser.add_argument(
        "--dataset-dirpath",
        type=Path,
        default=None,
        help="If given, write a new random train/val split here. The splits used in "
        "the paper are provided in datasets/, so this is not needed to reproduce.",
    )
    args = parser.parse_args()

    filepaths = list(args.in_dirpath.glob("**/*.wav"))
    out_filepaths = []
    for wav_filepath in tqdm(filepaths, desc="Processing WAV files"):
        out_filepath = args.out_dirpath / wav_filepath.name
        if out_filepath.exists():
            continue
        wav, sr = torchaudio.load(wav_filepath)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        out_filepath.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_filepath, wav, 16000)
        out_filepaths.append(out_filepath)

    if args.dataset_dirpath is None:
        return
    train_filepaths, val_filepaths = train_test_split(
        out_filepaths, test_size=0.1, random_state=42
    )
    args.dataset_dirpath.mkdir(parents=True, exist_ok=True)
    with open(args.dataset_dirpath / "carlos_jp_el100_train.txt", "w") as f:
        for filepath in train_filepaths:
            f.write(f"{filepath}|\n")
    with open(args.dataset_dirpath / "carlos_jp_el100_val.txt", "w") as f:
        for filepath in val_filepaths:
            f.write(f"{filepath}|\n")


if __name__ == "__main__":
    main()
