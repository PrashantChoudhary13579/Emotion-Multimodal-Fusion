# Emotion-Multimodal-Fusion

Audio-visual emotion recognition using **Whisper** (audio) and **V-JEPA2** (video) embeddings, combined with three different multimodal fusion strategies: **early**, **late**, and **mid (gated)** fusion.

Target classes across all experiments: `anger`, `disgust`, `fear`, `happy`, `sad`. A sixth class, `neutral`, exists in some raw data but is excluded to keep the label space consistent across datasets.

## Overview

Each sample (an audio clip + its paired video clip) is first converted into two fixed-size embedding vectors:

- **Audio embedding** — extracted with a pretrained speech model (Whisper or Wav2Vec2, see `src/audio_embeddings.py`) → Whisper-small produces `[B, 1500, 768]` pooled to `[B, 768]`
- **Video embedding** — extracted with V-JEPA2 (see `src/video_embeddings.py`) → V-JEPA2 ViT-L produces `[B, 8192, 1024]` pooled to `[B, 1024]`

These embeddings are then combined with one of three fusion heads and trained to classify the emotion label:

| Fusion strategy | Idea | Training script |
|---|---|---|
| **Early fusion** | Concatenates audio + video embeddings into a single vector, fed through one MLP (`FusionMLP`). Simplest and fastest, but gives the model no explicit way to weigh one modality against the other — the higher-dimensional/more-discriminative modality (typically video) tends to dominate. | `src/train_baseline.py` |
| **Mid / gated fusion** | Projects audio and video into a shared space; a learned, per-sample gate weighs how much each modality contributes before fusion (`GatedFusionClassifier`). Addresses early fusion's imbalance directly, at the cost of a harder optimization problem (trained with a lower LR and higher dropout). | `src/train_gated_fusion.py` |
| **Late fusion** | Trains separate classifier heads per modality, then combines their output logits with a learned/searched blend weight `alpha` (`LateFusionHeads`). Most conservative — easiest to debug, degrades gracefully if one modality is missing — but has the least capacity to learn cross-modal interactions, since audio and video are never seen together until the final logit blend. | `src/train_late_fusion.py` |

Cross-validation variants of each (`*_cv_fold.py`) are also included, along with early-stopping and additional test-set variants of the baseline.

**Pipeline:** raw videos → audio extracted from video (`audio_extraction.py`) → audio/video pairs matched by label + filename stem → embeddings extracted and cached → fed into one of the three fusion trainers → checkpoint + `predictions.csv` / `metrics.json` / confusion matrices via the evaluation step. A trained checkpoint can be run on a single new audio+video pair via `inference.py`.

> **Important:** the `--audio_model` / `--video_model` used at evaluation time must always match what was used during training, and audio source (separately recorded vs. extracted from video) must match too — mismatches here silently degrade results.

## Training methodology

All three fusion strategies were trained on a purpose-built **Mixed Dataset** combining 200 samples per emotion from each of four source datasets (SAMVEDNA, PUMAVE-D, RAVDESS, eNTERFACE'05) — 4,000 clips total, split 2,400 train / 400 val / 1,200 test — rather than on any single dataset alone. This was a deliberate fix for a generalization gap found in an earlier phase: models trained on SAMVEDNA alone reached ~98% in-domain but only 29–50% accuracy on the other three datasets, because they never saw more than one recording condition during training.

Each fusion strategy was trained for **150 epochs**, batch size 32, AdamW, under **5-fold cross-validation** (`*_cv_fold.py` scripts), on a 24GB NVIDIA RTX A5500 (~3 hours per run):

| Fusion | LR | Dropout | Hidden dim |
|---|---|---|---|
| Early | 1e-4 | 0.3 | — |
| Mid (gated) | 2e-5 | 0.4 | 256 |
| Late | 1e-4 | 0.2 | 256 |

Because SAMVEDNA is part of the Mixed training set, its test-set numbers below reflect **in-domain** performance; PUMAVE-D, RAVDESS, and eNTERFACE'05 are **cross-dataset** (held out entirely from training) — which is the harder, more meaningful generalization test. Across all three architectures, `fear` and `disgust` are the most-confused emotion pair.

## Results

Test-set accuracy (from 5-fold CV training on the Mixed Dataset), by fusion strategy and evaluation dataset:

| Fusion | MIXED (in-domain, held-out) | SAMVEDNA (in-domain) | PUMAVE-D (cross-dataset) | eNTERFACE'05 (cross-dataset) | RAVDESS (cross-dataset) |
|---|---|---|---|---|---|
| Early Fusion | 98.88% | 93.49% | 96.89% | 93.00% | 89.11% |
| Mid (Gated) Fusion | 97.81% | 92.81% | 96.80% | 95.42% | 88.12% |
| Late Fusion | 98.75% | **95.08%** | **97.48%** | **96.08%** | **93.12%** |

- **Early fusion** wins on the MIXED test set, but generalizes worst of the three to unseen datasets (RAVDESS in particular).
- **Late fusion** is the most consistent — highest accuracy on every cross-dataset evaluation (eNTERFACE'05, RAVDESS, SAMVEDNA) and a close second on MIXED — suggesting combining modality-specific predictions at a later stage improves robustness to unseen recording conditions.
- **Mid (gated) fusion** is competitive but doesn't consistently beat the other two.

The MIXED test-set metrics above aren't yet in [`results/`](results/) (only the four per-dataset evaluation folders are); the late-fusion MIXED-checkpoint metrics.json specifically wasn't retained from the original experiment run and would need to be regenerated. Full per-class metrics, confusion matrices, and classification reports for the available runs are in [`results/`](results/) — see [`results/README.md`](results/README.md) for details.

## Repository structure

```
.
├── src/                     # All training / inference / preprocessing code
│   ├── audio_embeddings.py       # Audio embedding extraction (Whisper / Wav2Vec2)
│   ├── video_embeddings.py       # Video embedding extraction (V-JEPA2)
│   ├── audio_extraction.py       # Extract audio track from video files
│   ├── audio_conversion.py       # Audio format conversion utilities
│   ├── data_loaders.py           # Dataset loading utilities
│   ├── dataset_reorganized_*.py  # Dataset-specific folder reorganization scripts
│   ├── mixed_dataset.py          # Utilities for the combined/mixed dataset
│   ├── multimodal_embed.py       # Joint audio+video embedding helper
│   ├── merge_whisper_vjepa.py    # Merge separately-cached audio/video embeddings
│   ├── train_baseline.py         # Early fusion training
│   ├── train_baseline_cv_fold*.py# Early fusion, cross-validation / early-stop variants
│   ├── train_late_fusion*.py     # Late fusion training (+ CV variant)
│   ├── train_gated_fusion*.py    # Gated fusion training (+ CV / smoothed variants)
│   ├── train_script.py           # Convenience driver script
│   ├── plot_metrics.py           # Plot confusion matrices / metric curves
│   └── inference.py              # Run inference with a trained checkpoint
├── results/                 # Evaluation outputs (metrics, confusion matrices, predictions)
│   ├── early_fusion/
│   ├── late_fusion/
│   └── gated_fusion/
├── docs/
│   ├── INSTALL.md                        # Detailed environment setup / troubleshooting guide
│   ├── training_code.md                  # Example training commands used for these runs
│   └── chapter3_fusion_methodology.md    # Full write-up: fusion architectures, mixed-dataset construction, results discussion
├── environment-cpu.yml      # Conda environment (CPU-only)
├── environment-gpu.yml      # Conda environment (GPU / CUDA)
├── requirements.txt         # Pip requirements
├── pyproject.toml
└── uv.lock
```

## Setup

See [`docs/INSTALL.md`](docs/INSTALL.md) for a detailed guide (including why a plain `pip install torch` can take over an hour, and how to avoid it).

**Quick start with conda/mamba:**

```bash
conda install -n base -c conda-forge mamba -y

# GPU:
mamba env create -f environment-gpu.yml
# CPU-only:
mamba env create -f environment-cpu.yml

conda activate mer
```

**Or with pip:**

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### 1. Extract embeddings

Precompute audio and video embeddings for your dataset (organized as `<root>/<label>/<file>`):

```bash
python src/audio_embeddings.py --audio_root path/to/audio --out_dir cache/audio_emb --model openai/whisper-small
python src/video_embeddings.py --video_root path/to/video --out_dir cache/video_emb --model facebook/vjepa2-vitl-fpc64-256
```

### 2. Train a fusion model

```bash
# Early fusion
python src/train_baseline.py --audio_root path/to/audio --video_root path/to/video --out_dir models/early

# Late fusion
python src/train_late_fusion.py cache/audio_emb cache/video_emb --out_dir models/late

# Gated fusion
python src/train_gated_fusion.py cache/audio_emb cache/video_emb --out_dir models/gated
```

See [`docs/training_code.md`](docs/training_code.md) for the exact commands (with hyperparameters) used to produce the results in this repo. Run any script with `--help` for the full list of options.

### 3. Run inference

```bash
python src/inference.py --model_path models/early/best_model.pt --audio path/to/clip.wav --video path/to/clip.mp4
```

## Datasets

This project was evaluated on:

- **[RAVDESS](https://zenodo.org/record/1188976)** — Ryerson Audio-Visual Database of Emotional Speech and Song
- **[eNTERFACE'05](https://enterface.net/results/)** — audio-visual emotion database
- **SAMVEDNA** — Our own Multilingual Dataset consisting of English, Hindi and Punjabi
- **PUMAVE-D** — Punjab University Multilingual Audio and Video facial Expression Dataset

Datasets themselves are not included in this repository — point the scripts at your local copies via `--audio_root` / `--video_root`.

## License

Add a license of your choice (e.g. MIT) before publishing — none is currently specified.
