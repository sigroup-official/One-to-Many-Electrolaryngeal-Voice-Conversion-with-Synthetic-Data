# One-to-Many Electrolaryngeal Voice Conversion with Synthetic Data

Bowen Wu, Haruto Ueno, Carlos Toshinori Ishi, Chaoran Liu

[[Paper]](https://www.isca-archive.org/interspeech_2026/wu26l_interspeech.pdf)
[[Demo page]](https://sigroup-official.github.io/One-to-Many-Electrolaryngeal-Voice-Conversion-with-Synthetic-Data/)

Restoring natural (NL) speech from electrolarynx (EL) speech is crucial for EL users who have undergone laryngectomy.
Previous methods for time-aligned EL2NL conversion suffer from data scarcity for training effective deep learning models.
We propose to utilize any-to-many voice conversion (VC) with a small amount of EL speech to synthesize EL speech from NL speech, using which we finetune an existing VC for EL2NL conversion.
Our model outperforms previous methods on intonation and intelligibility, and inherits the ability of converting to many target voices.

```txt
 EL speech (what to say)  ──────► content encoder ───┐
                                                     ├──► EL2NL model ──► NL speech
 target NL speech (voice) ──────► speaker encoder ───┘                    (content of the EL speech,
                                                                           voice of the target speech)
```

The model is [QuickVC (QVC)](https://github.com/quickvc/QuickVC-VoiceConversion) with the Whisper large-v2 encoder as the content encoder,
finetuned on synthetic EL speech. The target voice, one of the speakers in the training data (JVS), is specified by an NL speech recording of that speaker.

## Quick start

Convert EL speech to NL speech of many target voices (the speakers in the training data, i.e., JVS) with the released EL2NL model.
The environment is managed by [uv](https://docs.astral.sh/uv/). Inference runs on GPU or CPU; training requires a CUDA GPU.

```bash
git clone https://github.com/sigroup-official/One-to-Many-Electrolaryngeal-Voice-Conversion-with-Synthetic-Data.git
cd One-to-Many-Electrolaryngeal-Voice-Conversion-with-Synthetic-Data
uv sync && source .venv/bin/activate

# Whisper large-v2 (content encoder)
mkdir -p checkpoints/whisper
wget -O checkpoints/whisper/large-v2.pt https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt

# EL2NL model (extracted to checkpoints/el2nl/)
wget -O el2nl.tar.gz "https://www.dropbox.com/scl/fi/medxno6ev7o4l128gimwy/el2nl.tar.gz?rlkey=f05weu1ie5yru1z5hrhf4ss22&st=geoem401&dl=1"
tar -xzf el2nl.tar.gz

# Convert the examples
python scripts/infer/convert.py \
    --hpfile checkpoints/el2nl/config.json \
    --ptfile checkpoints/el2nl/G_204765.pth \
    --txtpath assets/example/convert.txt \
    --outdir outputs/example        # add --device cpu to run on CPU (default: GPU if available)
```

This converts two EL utterances (`assets/example/el/`) to a male (jvs001) and a female (jvs002) voice (`assets/example/reference/`);
the expected outputs are in `assets/example/converted/`.
To convert your own speech, write a list of `title|EL speech wav|target speech wav` lines like `assets/example/convert.txt`
(any sampling rate; both are loaded as mono and resampled to 16 kHz).

## Reproduction

All commands are run from the repository root in the environment above.

### Overview

```txt
JVS NL speech
      |
(1) pretrain QVC (any-to-many VC)
      |
(2) finetune on 90 EL utterances -> NL2EL model
      |
(3) convert JVS NL speech -> synthetic EL speech
      |
(4) finetune the pretrained QVC on (synthetic EL -> NL) pairs -> EL2NL model
```

The flat-F0 baseline (5) replaces the synthetic EL speech of (3) with NL speech whose F0 is flattened.

### Data

**JVS corpus (NL speech).**
Download the [JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) and unzip it to `data/jvs/jvs_ver1`.
Extract the normal utterances (`nonpara30`, `parallel100`) and resample them to 16 kHz:

```bash
python scripts/preprocess/preprocess_wav.py \
    --jvs-dirpath data/jvs/jvs_ver1 \
    --out-dirpath data/jvs/jvs_wav_preprocessed
```

**EL speech.**
100 utterances (`001.wav`-`100.wav`) of the VOICEACTRESS100 sentences, i.e., `XXX.wav` reads the same sentence as `jvsNNN/VOICEACTRESS100_XXX.wav`.
The EL speech data is available upon request; please contact us (bowen.wu@atr.jp).
Extract it at the repository root, which places the 16 kHz wavs at `data/carlos_jp_el100/carlos_jp_el100_wav_preprocessed/XXX.wav`.

**Data splits.**
The splits used in the paper are provided in `datasets/`:

| Directory | Used for | Train | Val |
|---|---|---|---|
| `datasets/jvs` | pretraining QVC on NL speech | 11697 | 1300 |
| `datasets/carlos_jp_el100` | NL2EL finetuning on EL speech | 90 | 10 |
| `datasets/jvs_el2nl_10k` | EL2NL finetuning ("10k" pairs) | 9999 | 500 |

The EL2NL lists contain NL wavs `X.wav`; the paired EL input is `X{el_suffix}.wav`, where `el_suffix` is set in the config
(`.el_whisper` for ours, `.el_flat` for flat-F0).

### Checkpoints

You can use our EL2NL checkpoints below, or train the models yourself following [Training](#training).
Extract the archives at the repository root (e.g., `tar -xzf el2nl.tar.gz`):

| Archive | Extracted to | Description |
|---|---|---|
| [el2nl.tar.gz](https://www.dropbox.com/scl/fi/medxno6ev7o4l128gimwy/el2nl.tar.gz?rlkey=f05weu1ie5yru1z5hrhf4ss22&st=geoem401&dl=0) (143 MB) | `checkpoints/el2nl/G_204765.pth` | EL2NL model, ours-10k (generator only). |
| [el2nl_flat.tar.gz](https://www.dropbox.com/scl/fi/qjametivgu3jvnlkbez6v/el2nl_flat.tar.gz?rlkey=n5rrte9ewdr7nke6901omb5v8&st=z4k88xl3&dl=0) (143 MB) | `checkpoints/el2nl_flat/G_204765.pth` | EL2NL model, flat-F0 baseline (generator only). |

Each directory also contains the `config.json` used for training.
The NL2EL model (`checkpoints/nl2el/G_35000.pth`) is available upon request (bowen.wu@atr.jp).

### Evaluation

CER is computed against the transcriptions of JSUT. Download [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)
and put it (at least the `*/transcript_utf8.txt` files) at `data/jsut_ver1.1`.

Both evaluations require the EL speech and the preprocessed JVS speech (see [Data](#data)).

**EL2NL on real EL speech** (Table 2): each of the 100 EL utterances is converted to 5 target speakers of JVS
(jvs001, jvs021, jvs081: male; jvs043, jvs061: female), using the NL recording of the same sentence as the target speech,
and compared with it (MCD, log F0 RMSE, F0 correlation, CER with whisper-large-v3, voice similarity with Resemblyzer):

```bash
python scripts/eval/eval_el.py --hpfile checkpoints/el2nl/config.json --ptfile checkpoints/el2nl/G_204765.pth \
    --outdir outputs/eval_el/el2nl
```

**Synthetic EL quality**: MCD between the real EL speech and the synthetic EL speech of the same sentence (jvs001, 021, 041, 061, 081).
This requires the synthetic EL speech generated in [step 3](#3-synthesize-el-speech) (or [step 5](#5-baseline-flat-f0) with `--syn-suffix .el_flat`).
To generate it with our NL2EL model, please contact us (bowen.wu@atr.jp);
otherwise, use the NL2EL model you trained in [Training](#training) (step 2):

```bash
python scripts/eval/eval_synthetic_mcd.py --syn-suffix .el_whisper --out-filepath outputs/synthetic_mcd/el_whisper.txt
```

### Results

#### EL2NL on real EL speech

Rows marked *(this repo)* are reproduced with this repository and the released checkpoints.
Conversion samples a latent variable, so the metrics vary slightly between runs (about ±0.02 in MCD).

|  | MCD | F0 R. | F0 C. | CER | Sim. |
|---|---|---|---|---|---|
| QVC | 7.150 | 0.332 | 0.431 | 0.563 | 0.893 |
| QVC-mix | 9.017 | 0.716 | 0.088 | 0.682 | 0.591 |
| flat-F0 | 6.885 | 0.337 | 0.548 | <u>0.373</u> | 0.916 |
| flat-F0 *(this repo)* | 6.844 | 0.321 | 0.568 | 0.390 | 0.911 |
| ours-100 | <u>6.600</u> | 0.315 | 0.555 | 0.375 | 0.907 |
| ours-1k | **6.587** | <u>0.311</u> | <u>0.571</u> | 0.381 | <u>0.917</u> |
| ours-10k | 6.607 | **0.309** | **0.585** | **0.366** | **0.923** |
| ours-10k *(this repo)* | 6.530 | 0.308 | 0.594 | 0.358 | 0.922 |

F0 is in log-scale. R.: RMSE, C.: correlation, Sim.: voice similarity.
Bold and underline mark the best and second best among the paper's results.

#### Synthetic EL speech

MCD between real EL speech and synthetic EL speech of the same sentence (lower is closer to real EL).

|  | MCD |
|---|---|
| NL speech (no conversion) | 12.825 |
| flat-F0 *(paper)* | 11.916 |
| flat-F0 *(this repo)* | 11.867 |
| NL2EL *(paper)* | 5.942 |
| NL2EL *(this repo)* | 5.793 |

### Training

Training scripts save logs and checkpoints to `logs/{model}` (`-m {model}`), resume from the latest `G_*.pth`/`D_*.pth` in it,
and use all visible GPUs. All models in the paper were trained on a single GPU (`CUDA_VISIBLE_DEVICES=0`).
Finetuning is done by copying the pretrained checkpoint into the new log directory before training;
`epochs` in the finetuning configs therefore counts from the epoch of the pretrained checkpoint (938).

> Note: if a checkpoint in `logs/{model}` has no optimizer state, training silently starts from scratch.
> Always initialize finetuning with `checkpoints/qvc_pretrained/{G,D}_180000.pth`.

#### 1. Pretrain QVC on NL speech

Extract the Whisper features of the JVS speech (saved as `X.ppg_whisper_large_v2.npy` next to `X.wav`); they are needed in the following steps either way:

```bash
python scripts/preprocess/preprocess_weo.py --wav-dirpath data/jvs/jvs_wav_preprocessed --mode whisper_large_v2
```

Then either use our pretrained QVC or train it yourself; both result in `checkpoints/qvc_pretrained/{G,D}_180000.pth`.

**Option A: use our checkpoint.**
Download [qvc_pretrained.tar.gz](https://www.dropbox.com/scl/fi/69jsqmsl5sa2ut649vv53/qvc_pretrained.tar.gz?rlkey=e01ej1repqdgbvzjnzv6gnp1a&st=n5pk5hum&dl=0) (922 MB) and extract it at the repository root.
It contains the generator and discriminator at step 180k (epoch 938), including optimizer states.

**Option B: train it yourself.**
Train on JVS as in the paper: `G_180000.pth`/`D_180000.pth` are saved at step 180k (epoch 938);
stop the training after that (the config runs up to epoch 2000) and copy them:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train/train.py -c configs/qvc_whisper.json -m qvc_whisper
mkdir -p checkpoints/qvc_pretrained
cp logs/qvc_whisper/G_180000.pth logs/qvc_whisper/D_180000.pth checkpoints/qvc_pretrained/
```

#### 2. Finetune QVC for NL2EL

Finetune the pretrained QVC on the EL speech with the content encoder frozen (until epoch 2500, i.e., step 35k):

```bash
python scripts/preprocess/preprocess_weo.py --wav-dirpath data/carlos_jp_el100/carlos_jp_el100_wav_preprocessed --mode whisper_large_v2

mkdir -p logs/qvc_whisper_ft_nl2el
cp checkpoints/qvc_pretrained/G_180000.pth checkpoints/qvc_pretrained/D_180000.pth logs/qvc_whisper_ft_nl2el/
CUDA_VISIBLE_DEVICES=0 python scripts/train/ft_nl2el.py -c configs/qvc_whisper_ft_nl2el.json -m qvc_whisper_ft_nl2el
```

#### 3. Synthesize EL speech

Convert all JVS NL speech to EL speech with the NL2EL model (target voice: `001.wav` of the EL speech),
saved as `X.el_whisper.wav`, and extract their Whisper features.
The commands use the NL2EL model of the paper (available upon request, bowen.wu@atr.jp);
the model trained in step 2 can be used instead (`--hpfile configs/qvc_whisper_ft_nl2el.json --ptfile logs/qvc_whisper_ft_nl2el/G_35000.pth`).

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/preprocess/convert_to_el.py \
    --wav-dirpath data/jvs/jvs_wav_preprocessed \
    --hpfile checkpoints/nl2el/config.json \
    --ptfile checkpoints/nl2el/G_35000.pth \
    --suffix .el_whisper
python scripts/preprocess/preprocess_weo.py --wav-dirpath data/jvs/jvs_wav_preprocessed \
    --mode whisper_large_v2 --glob "**/*.el_whisper.wav"
```

#### 4. Finetune QVC for EL2NL (ours-10k)

Finetune the pretrained QVC on 10k (synthetic EL, NL) pairs for 304 epochs (50k steps):

```bash
mkdir -p logs/qvc_whisper_ft_el2nl
cp checkpoints/qvc_pretrained/G_180000.pth checkpoints/qvc_pretrained/D_180000.pth logs/qvc_whisper_ft_el2nl/
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_el2nl.py -c configs/qvc_whisper_ft_el2nl.json -m qvc_whisper_ft_el2nl
```

The final checkpoint is `logs/qvc_whisper_ft_el2nl/G_204765.pth` (this is `checkpoints/el2nl`).

#### 5. Baseline: flat-F0

The flat-F0 baseline replaces the synthetic EL speech of step 3 with NL speech whose F0 is flattened by the WORLD vocoder
(voiced F0 set to 100 Hz, aperiodicity capped, noise added), saved as `X.el_flat.wav`.
Finetuning is the same as step 4:

```bash
python scripts/preprocess/create_const_f0_data.py --wav-dirpath data/jvs/jvs_wav_preprocessed --suffix .el_flat
python scripts/preprocess/preprocess_weo.py --wav-dirpath data/jvs/jvs_wav_preprocessed \
    --mode whisper_large_v2 --glob "**/*.el_flat.wav"

mkdir -p logs/qvc_whisper_ft_el2nl_flat
cp checkpoints/qvc_pretrained/G_180000.pth checkpoints/qvc_pretrained/D_180000.pth logs/qvc_whisper_ft_el2nl_flat/
CUDA_VISIBLE_DEVICES=0 python scripts/train/train_el2nl.py -c configs/qvc_whisper_ft_el2nl_flat.json -m qvc_whisper_ft_el2nl_flat
```

## Acknowledgement

This implementation is based on [QuickVC](https://github.com/quickvc/QuickVC-VoiceConversion)
and uses [Whisper](https://github.com/openai/whisper).

## License

This repository, including the released checkpoints, is licensed under [CC BY-NC 4.0](LICENSE): commercial use is not permitted.
`src/qvc` is based on [QuickVC](https://github.com/quickvc/QuickVC-VoiceConversion) (MIT License, `src/qvc/LICENSE`)
and `src/whisper` is from [Whisper](https://github.com/openai/whisper) (MIT License, `src/whisper/LICENSE`).

## Citation

```bibtex
@inproceedings{wu26l_interspeech,
  title     = {{One-to-Many Electrolaryngeal Voice Conversion with Synthetic Data}},
  author    = {Bowen Wu and Haruto Ueno and Carlos Toshinori Ishi and Chaoran Liu},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {6851--6855},
  doi       = {10.21437/Interspeech.2026-2150},
  issn      = {2958-1796},
}
```
