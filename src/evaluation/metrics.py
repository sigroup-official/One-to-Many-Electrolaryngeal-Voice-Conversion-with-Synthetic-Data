"""Objective metrics and conversion utilities for EL2NL evaluation.

MCD (mel_cepstral_distance), log F0 RMSE / F0 correlation (WORLD + DTW),
CER (whisper-large-v3 ASR) and voice similarity (Resemblyzer).
"""

import logging

import evaluate
import librosa
import numpy as np
import pysptk
import pyworld as pw
import torch
from fastdtw.fastdtw import fastdtw
from scipy import spatial
from scipy.io import wavfile
from scipy.stats import pearsonr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from qvc.mel_processing import mel_spectrogram_torch
from qvc.models import SynthesizerTrn
from qvc.utils import get_hparams_from_file, load_checkpoint
from whisper.functionals import load_model, pred_ppg_infer

logging.getLogger("numba").setLevel(logging.WARNING)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


class VoiceSimilarity:
    def __init__(self):
        from resemblyzer import VoiceEncoder, preprocess_wav

        self.resemb_encoder = VoiceEncoder()
        self.preprocess_wav = preprocess_wav

    def score(self, wav1_filepath: str, wav2_filepath: str) -> float:
        wav1 = self.preprocess_wav(wav1_filepath)
        wav2 = self.preprocess_wav(wav2_filepath)
        wav1_embed = self.resemb_encoder.embed_utterance(wav1)
        wav2_embed = self.resemb_encoder.embed_utterance(wav2)
        sim = np.inner(wav1_embed, wav2_embed)
        return sim


def _get_best_mcep_params(fs: int):
    if fs == 16000:
        return 23, 0.42
    elif fs == 22050:
        return 34, 0.45
    elif fs == 24000:
        return 34, 0.46
    elif fs == 44100:
        return 39, 0.53
    elif fs == 48000:
        return 39, 0.55
    else:
        raise ValueError(f"Not found the setting for {fs}.")


class WhisperASR:
    def __init__(self):
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.cuda()
        processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device="cuda",
        )

    def __call__(self, audio: np.ndarray):
        return self.pipe(audio, generate_kwargs={"language": "ja"})["text"]


class CERMetric:
    def __init__(self):
        self.metric = evaluate.load("cer")
        self.normalizer = BasicTextNormalizer()
        self.normalize = (
            lambda x: self.normalizer(x)
            .replace(" ", "")
            .replace("。.", "")
            .replace(",、", "")
        )
        self.reset()

    def reset(self):
        self.preds = []
        self.refs = []

    def push(self, pred, ref):
        self.preds.append(pred)
        self.refs.append(ref)

    def get_score(self):
        preds = [self.normalize(i) for i in self.preds]
        refs = [self.normalize(i) for i in self.refs]
        flag = [len(i) != 0 for i in refs]
        preds = list(list(zip(*filter(lambda x: x[1], zip(preds, flag))))[0])
        refs = list(list(zip(*filter(lambda x: x[1], zip(refs, flag))))[0])
        return self.metric.compute(predictions=preds, references=refs)


def load_model_for_eval(hpfile, ptfile):
    hps = get_hparams_from_file(hpfile)
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,  # type: ignore
        hps.train.segment_size // hps.data.hop_length,  # type: ignore
        **hps.model,  # type: ignore
    ).cuda()
    net_g.eval()
    # load_checkpoint silently keeps random init for missing keys, so check first
    # (legacy weight-norm ckpts need scripts/tools/convert_legacy_ckpt.py)
    saved = torch.load(ptfile, map_location="cpu", weights_only=False)["model"]
    missing = [k for k in net_g.state_dict() if k not in saved]
    assert not missing, f"{len(missing)} keys missing in {ptfile}, e.g. {missing[:3]}"
    load_checkpoint(ptfile, net_g, None)
    assert hps.model.ssl_mode == "whisper_large_v2", "Only whisper is supported"  # type: ignore
    whisper_model = load_model("checkpoints/whisper/large-v2.pt")
    return net_g, whisper_model, hps


@torch.no_grad()
def generate(net_g, whisper_model, hps, src_filepath, tgt_filepath, rec_filepath):
    # Source input
    weo = torch.from_numpy(pred_ppg_infer(whisper_model, src_filepath))
    weo = weo.transpose(1, 0)
    c = weo.cuda().unsqueeze(0)

    # Target input
    wav_tgt, _ = librosa.load(tgt_filepath, sr=hps.data.sampling_rate)  # type: ignore
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
    audio = audio[0][0].data.cpu().float().numpy()

    wavfile.write(
        rec_filepath,
        hps.data.sampling_rate,  # type: ignore
        audio,
    )

    return audio


def extract_mcep_f0(audio, sr=16000):
    # F0
    f0, time_axis = pw.harvest(
        audio.astype((np.float64)),
        sr,
        f0_floor=40,
        f0_ceil=800,
        frame_period=256 / sr * 1000,
    )
    tgt_sp = pw.cheaptrick(
        audio.astype((np.float64)),
        f0,
        time_axis,
        sr,
        fft_size=1024,
    )
    tgt_mcep = pysptk.sp2mc(tgt_sp, *_get_best_mcep_params(sr))
    return f0, tgt_mcep, time_axis


def evaluate_f0(audio1, sr1, audio2, sr2):
    f0_1, mcep_1, _ = extract_mcep_f0(audio1, sr1)
    f0_2, mcep_2, _ = extract_mcep_f0(audio2, sr2)
    _, path = fastdtw(mcep_1, mcep_2, dist=spatial.distance.euclidean)
    twf = np.array(path).T
    f0_1_dtw = f0_1[twf[0]]
    f0_2_dtw = f0_2[twf[1]]

    # Get voiced part
    nonzero_idxs = np.where((f0_1_dtw != 0) & (f0_2_dtw != 0))[0]
    f0_1_dtw_voiced = np.log(f0_1_dtw[nonzero_idxs])
    f0_2_dtw_voiced = np.log(f0_2_dtw[nonzero_idxs])

    # log F0 RMSE
    log_f0_rmse = np.sqrt(np.mean((f0_1_dtw_voiced - f0_2_dtw_voiced) ** 2))

    # F0 CORR
    f0_corr, _ = pearsonr(f0_1_dtw_voiced, f0_2_dtw_voiced)

    return log_f0_rmse, f0_corr
