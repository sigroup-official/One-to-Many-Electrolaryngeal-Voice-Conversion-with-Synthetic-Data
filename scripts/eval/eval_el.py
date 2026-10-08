"""Objective evaluation of EL2NL conversion on real EL speech (Table 2 in the paper).

Each of the 100 EL utterances XXX.wav (VOICEACTRESS100 sentences) is converted to
each target speaker jvs{sid}, using the NL recording of the same sentence
(jvs{sid}/VOICEACTRESS100_XXX.wav) as the target speech, and compared with it:
MCD, log F0 RMSE, F0 correlation, CER (whisper-large-v3 ASR) and voice similarity
(resemblyzer). By default the 5 target speakers of the paper are used.
"""

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
from mel_cepstral_distance import compare_audio_files
from scipy.io import wavfile
from tqdm import tqdm

from evaluation.metrics import (
    CERMetric,
    VoiceSimilarity,
    WhisperASR,
    evaluate_f0,
    generate,
    load_model_for_eval,
)

EL_DIRPATH = Path("data/carlos_jp_el100/carlos_jp_el100_wav_preprocessed")
JVS_DIRPATH = Path("data/jvs/jvs_wav_preprocessed")


def get_el_speech_to_transcription():
    with open("data/jsut_ver1.1/voiceactress100/transcript_utf8.txt", "r") as f:
        lines = f.readlines()
    el2text = {}
    for i, line in enumerate(lines):
        text = line.split(":", 1)[1]
        el2text[EL_DIRPATH / f"{i+1:03d}.wav"] = text.strip()
    return el2text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, required=True, help="path to json config file")
    parser.add_argument("--ptfile", type=str, required=True, help="path to pth file")
    parser.add_argument("--outdir", type=Path, required=True, help="path to output dir")
    parser.add_argument(
        "--sids",
        type=int,
        nargs="+",
        default=[1, 21, 43, 61, 81],
        help="JVS target speakers (3 male, 2 female, as in the paper)",
    )
    args = parser.parse_args()

    el2text = get_el_speech_to_transcription()
    asr_model = WhisperASR()
    cer_metric = CERMetric()
    voice_sim = VoiceSimilarity()
    net_g, whisper_model, hps = load_model_for_eval(args.hpfile, args.ptfile)

    output_dirpath: Path = args.outdir
    output_dirpath.mkdir(parents=True, exist_ok=True)
    tmp_filepath = output_dirpath / "tgt_audio_tmp.wav"
    mcds, log_f0_rmses, f0_corrs, sims = [], [], [], []
    for sid in tqdm(args.sids, leave=False):
        for el_filepath, text in tqdm(el2text.items()):
            tgt_filepath = JVS_DIRPATH / f"jvs{sid:03d}/VOICEACTRESS100_{el_filepath.stem}.wav"
            out_filepath = output_dirpath / f"audio/{tgt_filepath.parent.stem}/{el_filepath.stem}.wav"
            out_filepath.parent.mkdir(parents=True, exist_ok=True)
            if not out_filepath.exists():
                out_audio = generate(
                    net_g,
                    whisper_model,
                    hps,
                    str(el_filepath),
                    str(tgt_filepath),
                    str(out_filepath),
                )
            else:
                out_audio, _ = librosa.load(str(out_filepath), sr=16000)

            tgt_audio, tgt_sr = librosa.load(tgt_filepath, sr=16000)
            wavfile.write(tmp_filepath, tgt_sr, tgt_audio)

            out_text = asr_model(out_audio)
            cer_metric.push(out_text, text)

            sim = voice_sim.score(str(tmp_filepath), str(out_filepath))
            sims.append(sim)

            mcd, _ = compare_audio_files(tmp_filepath, out_filepath)
            mcds.append(mcd)

            log_f0_rmse, f0_corr = evaluate_f0(tgt_audio, tgt_sr, out_audio, hps.data.sampling_rate)  # type: ignore
            log_f0_rmses.append(log_f0_rmse)
            f0_corrs.append(f0_corr)

    os.remove(tmp_filepath)

    result_str = f"# {args.ptfile}, sids={args.sids}, n={len(mcds)}\n"
    result_str += f"MCD, avg={np.mean(mcds):.3f}, std={np.std(mcds):.3f}\n"
    result_str += f"f0_rmse, avg={np.mean(log_f0_rmses):.3f}, std={np.std(log_f0_rmses):.3f}\n"
    result_str += f"f0_corr, avg={np.mean(f0_corrs):.3f}, std={np.std(f0_corrs):.3f}\n"
    result_str += f"cer={cer_metric.get_score():.3f}\n"
    result_str += f"sim, avg={np.mean(sims):.3f}, std={np.std(sims):.3f}\n"
    print(result_str)
    with open(output_dirpath / "results.txt", "w") as f:
        f.write(result_str)


if __name__ == "__main__":
    main()
