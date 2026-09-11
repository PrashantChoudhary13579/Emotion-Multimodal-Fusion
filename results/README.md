# Results

Full evaluation outputs for all three fusion strategies across all four datasets. Each `<fusion>/<dataset>/` folder contains:

- `metrics.json` — accuracy, precision/recall/F1 (macro, weighted, micro)
- `classification_report.txt` — per-class scikit-learn classification report
- `per_class_metrics.csv` — per-class precision/recall/F1
- `predictions.csv` — raw model predictions vs. ground truth
- `confusion_matrix.png` / `confusion_matrix_normalized.png`
- `per_class_f1.png`

Folders named `*_cv_dp*_ep150` (e.g. `early_fusion/cv_dp0.3_1e-4_ep150/`) contain the 5-fold cross-validation training runs themselves: per-fold checkpoints, loss/accuracy curves, and fold-level confusion matrices. All three fusion strategies were trained on a combined **Mixed Dataset** (200 samples/emotion from each of SAMVEDNA, PUMAVE-D, RAVDESS, eNTERFACE'05) — see the main [README](../README.md#training-methodology) for the full methodology.

**SAMVEDNA is in-domain** (part of the Mixed training set); **PUMAVE-D, RAVDESS, and eNTERFACE'05 are cross-dataset** — held out entirely from training, so they're the more meaningful generalization test. A MIXED held-out test set was also evaluated (Early 98.88%, Mid/Gated 97.81%, Late 98.75%) but those `metrics.json` files aren't included here — only the late-fusion one is missing from the original experiment archive; the early/mid ones could be added.

## Summary

| Fusion | Dataset | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | Samples |
|---|---|---|---|---|---|---|
| Early Fusion | enterface | 93.00% | 93.00% | 93.02% | 92.99% | 1072 |
| Early Fusion | pumave | 96.90% | 96.94% | 96.89% | 96.89% | 1032 |
| Early Fusion | ravdess | 89.11% | 89.38% | 89.11% | 88.98% | 1920 |
| Early Fusion | samvedna | 93.50% | 93.54% | 93.50% | 93.46% | 6750 |
| Late Fusion | enterface | 96.08% | 96.19% | 96.09% | 96.10% | 1072 |
| Late Fusion | pumave | 97.48% | 97.50% | 97.48% | 97.48% | 1032 |
| Late Fusion | ravdess | 93.12% | 93.21% | 93.12% | 93.10% | 1920 |
| Late Fusion | samvedna | 95.08% | 95.10% | 95.08% | 95.06% | 6748 |
| Gated Fusion | enterface | 95.43% | 95.48% | 95.43% | 95.45% | 1072 |
| Gated Fusion | pumave | 96.80% | 96.81% | 96.82% | 96.81% | 1032 |
| Gated Fusion | ravdess | 88.12% | 88.42% | 88.12% | 87.89% | 1920 |
| Gated Fusion | samvedna | 92.81% | 92.83% | 92.81% | 92.79% | 6750 |

**Late fusion is the strongest strategy on every dataset**, followed by gated fusion, with early (concatenation-based) fusion consistently weakest — most notably on RAVDESS.
