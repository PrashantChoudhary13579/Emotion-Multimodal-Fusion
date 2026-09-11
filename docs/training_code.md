# Example training commands

These are the commands (with hyperparameters) used to produce the cross-validation
results in [`results/`](../results/). Replace the paths with your own dataset/cache
locations.

## Early fusion (baseline, 5-fold CV)

```bash
python src/train_baseline_cv_fold.py \
  --audio_root "/path/to/Mixed_dataset_audio" \
  --video_root "/path/to/Mixed_dataset_video" \
  --audio_cache "/path/to/cache/audio_emb_whisper_small" \
  --video_cache "/path/to/cache/video_emb_vjepa2" \
  --audio_model "openai/whisper-small" \
  --video_model "facebook/vjepa2-vitl-fpc64-256" \
  --hidden 256 \
  --dropout 0.4 \
  --epochs 150 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --cv_folds 5 \
  --out_dir "results/early_fusion"
```

## Late fusion (5-fold CV)

```bash
python src/train_late_fusion_cv_fold.py \
  cache/audio_emb_whisper_small cache/video_emb_vjepa2 \
  --hidden 512 \
  --dropout 0.2 \
  --epochs 150 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --alpha 0.55 \
  --cv_folds 5 \
  --out_dir "results/late_fusion"
```

## Gated fusion (5-fold CV)

```bash
python src/train_gated_fusion_cv_fold.py \
  cache/audio_emb_whisper_small cache/video_emb_vjepa2 \
  --proj_dim 512 \
  --hidden 512 \
  --dropout 0.3 \
  --epochs 150 \
  --lr 2e-5 \
  --weight_decay 1e-4 \
  --cv_folds 5 \
  --out_dir "results/gated_fusion"
```

Run any script with `--help` to see the full list of available options.
