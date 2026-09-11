import os
import json
import argparse
import pathlib
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.model_selection import KFold
import statistics

import matplotlib
matplotlib.use("Agg")  # no display needed, just saves to file
import matplotlib.pyplot as plt
import numpy as np


def _list_pt(root: str) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(".pt"):
                files.append(os.path.join(dp, fn))
    return files


def _by_stem(files: List[str]) -> Dict[str, str]:
    return {pathlib.Path(p).stem: p for p in files}


def find_emb_pairs(audio_emb_root: str, video_emb_root: str) -> List[Tuple[str, str, str]]:
    """
    Pair saved embeddings by label directory and basename stem.
    Expects structure: <root>/<label>/**/<name>.pt for both modalities.
    Returns list of (label, audio_emb_path, video_emb_path).
    """
    pairs: List[Tuple[str, str, str]] = []
    a_labels = {d for d in os.listdir(audio_emb_root) if os.path.isdir(os.path.join(audio_emb_root, d))}
    v_labels = {d for d in os.listdir(video_emb_root) if os.path.isdir(os.path.join(video_emb_root, d))}
    labels = sorted(a_labels & v_labels)
    for label in labels:
        a_files = _list_pt(os.path.join(audio_emb_root, label))
        v_files = _list_pt(os.path.join(video_emb_root, label))
        v_map = _by_stem(v_files)
        for ap in a_files:
            stem = pathlib.Path(ap).stem
            vp = v_map.get(stem)
            if vp:
                pairs.append((label, ap, vp))
    return pairs


def split_train_val_test(
    items: List[Tuple[str, str, str]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    rng = torch.Generator()
    rng.manual_seed(seed)
    # Torch permutation for reproducibility
    idx = torch.randperm(len(items), generator=rng).tolist()
    items = [items[i] for i in idx]
    n = len(items)
    if n == 0:
        return [], [], []
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_val = max(0, min(n, n_val))
    n_test = max(0, min(n - n_val, n_test))
    n_train = n - n_val - n_test
    if n_train <= 0:
        # Reduce val/test to keep at least one train example
        while n_train <= 0 and (n_val > 0 or n_test > 0):
            if n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            n_train = n - n_val - n_test
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return train, val, test


class EmbPairEmbeddingsDataset(Dataset):
    def __init__(self, items: List[Tuple[str, str, str]], label_to_id: Dict[str, int]):
        self.items = items
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        label, a_path, v_path = self.items[idx]
        a = torch.load(a_path).view(-1).float()
        v = torch.load(v_path).view(-1).float()
        y = torch.tensor(self.label_to_id[label], dtype=torch.long)
        return a, v, y


class GatedFusionClassifier(nn.Module):
    def __init__(self, d_a: int, d_v: int, num_classes: int, proj_dim: int = 512, hidden: int = 512, dropout: float = 0.2, normalize: bool = True):
        super().__init__()
        self.normalize = normalize
        self.proj_a = nn.Linear(d_a, proj_dim)
        self.proj_v = nn.Linear(d_v, proj_dim)
        self.gate = nn.Linear(2 * proj_dim, proj_dim)
        self.cls = nn.Sequential(
            nn.Linear(proj_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # a, v: [B, d_a], [B, d_v]
        if self.normalize:
            a = F.normalize(a, p=2.0, dim=1)
            v = F.normalize(v, p=2.0, dim=1)
        a_p = self.proj_a(a)
        v_p = self.proj_v(v)
        if self.normalize:
            a_p = F.normalize(a_p, p=2.0, dim=1)
            v_p = F.normalize(v_p, p=2.0, dim=1)
        g = torch.sigmoid(self.gate(torch.cat([a_p, v_p], dim=1)))  # [B, proj_dim]
        h = g * a_p + (1.0 - g) * v_p
        if self.normalize:
            h = F.normalize(h, p=2.0, dim=1)
        logits = self.cls(h)
        return logits


def confusion_and_f1(num_classes: int, y_true: List[int], y_pred: List[int]):
    conf = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    per_class_f1 = []
    macro_sum, count = 0.0, 0
    for i in range(num_classes):
        tp = conf[i][i]
        col_sum = sum(conf[r][i] for r in range(num_classes))
        row_sum = sum(conf[i][c] for c in range(num_classes))
        fp = col_sum - tp
        fn = row_sum - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)
        if row_sum > 0:
            macro_sum += f1
            count += 1
    macro_f1 = macro_sum / count if count > 0 else 0.0
    return conf, per_class_f1, macro_f1


def print_confusion_matrix(conf, id_to_label):
    labels = [id_to_label[i] for i in range(len(conf))]
    col_width = max(len(l) for l in labels) + 2
    col_width = max(col_width, 6)

    header = " " * col_width + "".join(f"{l[:6]:>7}" for l in labels)
    print(header)
    for i, row in enumerate(conf):
        row_label = f"{labels[i]:<{col_width}}"
        row_str = "".join(f"{v:>7}" for v in row)
        print(row_label + row_str)
    print()

    print("Per-class recall (diagonal / row sum):")
    for i, row in enumerate(conf):
        row_sum = sum(row)
        recall = row[i] / row_sum if row_sum > 0 else 0.0
        print(f"  {labels[i]:<15} recall={recall:.3f}  (n={row_sum})")


def save_normalized_confusion_matrix(conf, id_to_label, out_path, title="Confusion Matrix (Normalized)"):
    conf_arr = np.array(conf, dtype=float)
    row_sums = conf_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_conf = conf_arr / row_sums

    labels = [id_to_label[i] for i in range(len(conf))]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), max(5, len(labels) * 0.8)))
    im = ax.imshow(norm_conf, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            val = norm_conf[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="Fraction of true class")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved normalized confusion matrix to {out_path}")


def plot_fold_convergence(fold_history, out_path, fold_idx):
    epochs = [h["epoch"] for h in fold_history]
    tr_loss = [h["train_loss"] for h in fold_history]
    va_loss = [h["val_loss"] for h in fold_history]
    tr_acc = [h["train_acc"] for h in fold_history]
    va_acc = [h["val_acc"] for h in fold_history]
    has_test = fold_history[0].get("test_loss") is not None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, tr_loss, label="Train loss")
    ax1.plot(epochs, va_loss, label="Val loss")
    if has_test:
        te_loss = [h["test_loss"] for h in fold_history]
        ax1.plot(epochs, te_loss, label="Test loss", linestyle="--", alpha=0.7)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Fold {fold_idx}: Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, tr_acc, label="Train acc")
    ax2.plot(epochs, va_acc, label="Val acc")
    if has_test:
        te_acc = [h["test_acc"] for h in fold_history]
        ax2.plot(epochs, te_acc, label="Test acc", linestyle="--", alpha=0.7)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.set_title(f"Fold {fold_idx}: Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved convergence plot to {out_path}")


# =========================================================================
# Single (non-CV) training path — unchanged from your original script
# =========================================================================
def train(args):
    pairs = find_emb_pairs(args.audio_emb_root, args.video_emb_root)
    if not pairs:
        raise SystemExit("No matched audio/video embeddings found.")
    labels = sorted({lbl for lbl, _, _ in pairs})
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    train_items, val_items, test_items = split_train_val_test(
        pairs, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dims from a probe sample
    a0 = torch.load(train_items[0][1]).view(-1)
    v0 = torch.load(train_items[0][2]).view(-1)
    d_a, d_v = int(a0.numel()), int(v0.numel())

    train_ds = EmbPairEmbeddingsDataset(train_items, label_to_id)
    val_ds = EmbPairEmbeddingsDataset(val_items, label_to_id)
    test_ds = EmbPairEmbeddingsDataset(test_items, label_to_id)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = GatedFusionClassifier(
        d_a=d_a, d_v=d_v, num_classes=len(labels), proj_dim=args.proj_dim,
        hidden=args.hidden, dropout=args.dropout, normalize=not args.no_normalize,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    def run_epoch(loader: DataLoader, train: bool, desc: str):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        for a, v, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            a = a.to(device)
            v = v.to(device)
            y = y.to(device)
            if train:
                opt.zero_grad()
            logits = model(a, v)
            loss = crit(logits, y)
            if train:
                loss.backward()
                opt.step()
            loss_sum += float(loss.item()) * a.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += a.size(0)
        return loss_sum / max(1, total), correct / max(1, total)

    def evaluate(loader: DataLoader, desc: str):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for a, v, y in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
                a = a.to(device)
                v = v.to(device)
                y = y.to(device)
                logits = model(a, v)
                loss = crit(logits, y)
                loss_sum += float(loss.item()) * a.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += a.size(0)
                all_true.extend(y.tolist())
                all_pred.extend(preds.tolist())
        conf, per_class_f1, macro_f1 = confusion_and_f1(len(labels), all_true, all_pred)
        return {
            "loss": loss_sum / max(1, total),
            "acc": correct / max(1, total),
            "macro_f1": macro_f1,
            "per_class_f1": per_class_f1,
            "confusion": conf,
            "n": total,
        }

    os.makedirs(args.out_dir, exist_ok=True)
    best_metric = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True, desc=f"Train {epoch}/{args.epochs}")
        va_metrics = evaluate(val_loader, desc=f"Val {epoch}/{args.epochs}")
        log_msg = (
            f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_metrics['loss']:.4f} acc {va_metrics['acc']:.4f} macroF1 {va_metrics['macro_f1']:.4f}"
        )
        te_metrics_epoch = None
        if getattr(args, "eval_test_each_epoch", False):
            te_metrics_epoch = evaluate(test_loader, desc=f"Test {epoch}/{args.epochs}")
            log_msg += (
                f" | test loss {te_metrics_epoch['loss']:.4f} acc {te_metrics_epoch['acc']:.4f} macroF1 {te_metrics_epoch['macro_f1']:.4f}"
            )
        print(log_msg)
        score = (va_metrics["macro_f1"] + va_metrics["acc"]) / 2.0
        if score > best_metric:
            best_metric = score
            torch.save({
                "state_dict": model.state_dict(),
                "labels": labels,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "d_a": d_a,
                "d_v": d_v,
                "proj_dim": args.proj_dim,
                "hidden": args.hidden,
                "normalize": not args.no_normalize,
            }, os.path.join(args.out_dir, "gated_fusion.pt"))
        epoch_entry = {
            "epoch": epoch,
            "train": {"loss": tr_loss, "acc": tr_acc},
            "val": va_metrics,
        }
        if te_metrics_epoch is not None:
            epoch_entry["test"] = te_metrics_epoch
        history.append(epoch_entry)

    # Final test
    te_metrics = evaluate(test_loader, desc="Test")
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)
    with open(os.path.join(args.out_dir, "metrics_gated.json"), "w") as f:
        json.dump({"best_val_score": best_metric, "test": te_metrics, "labels": labels}, f, indent=2)
    if args.log_history:
        with open(os.path.join(args.out_dir, args.log_history), "w") as f:
            json.dump(history, f, indent=2)
    print(f"Test: loss {te_metrics['loss']:.4f} acc {te_metrics['acc']:.4f} macroF1 {te_metrics['macro_f1']:.4f}")


# =========================================================================
# NEW: K-fold cross-validation path, mirroring the baseline CV script
# =========================================================================
def train_one_fold_gated(train_items, val_items, label_to_id, id_to_label, args, device,
                          d_a, d_v, fold_idx=None, test_items=None):
    train_ds = EmbPairEmbeddingsDataset(train_items, label_to_id)
    val_ds = EmbPairEmbeddingsDataset(val_items, label_to_id)

    test_ds = None
    test_loader = None
    if test_items:
        test_ds = EmbPairEmbeddingsDataset(test_items, label_to_id)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    num_classes = len(label_to_id)

    model = GatedFusionClassifier(
        d_a=d_a, d_v=d_v, num_classes=num_classes, proj_dim=args.proj_dim,
        hidden=args.hidden, dropout=args.dropout, normalize=not args.no_normalize,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_val_loss = float('inf')
    best_val_metrics = None
    best_confusion = None
    best_train_metrics = None
    best_state_dict = None
    fold_history = []

    for epoch in range(args.epochs):
        model.train()
        tr_total, tr_correct, tr_loss_sum = 0, 0, 0.0
        for a, v, y in tqdm(train_loader, desc=f"Fold {fold_idx} Train {epoch}/{args.epochs}", leave=False, dynamic_ncols=True):
            a, v, y = a.to(device), v.to(device), y.to(device)
            opt.zero_grad()
            logits = model(a, v)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            tr_loss_sum += float(loss.item()) * a.size(0)
            tr_correct += int((logits.argmax(dim=1) == y).sum().item())
            tr_total += a.size(0)
        train_acc = tr_correct / max(1, tr_total)
        train_loss = tr_loss_sum / max(1, tr_total)

        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for a, v, y in tqdm(val_loader, desc=f"Fold {fold_idx} Val {epoch}/{args.epochs}", leave=False, dynamic_ncols=True):
                a, v, y = a.to(device), v.to(device), y.to(device)
                logits = model(a, v)
                loss = crit(logits, y)
                loss_sum += float(loss.item()) * a.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += a.size(0)
                all_true.extend(y.tolist())
                all_pred.extend(preds.tolist())
        val_loss = loss_sum / max(1, total)
        val_acc = correct / max(1, total)
        conf, per_class_f1, macro_f1 = confusion_and_f1(num_classes, all_true, all_pred)

        test_loss, test_acc = None, None
        if test_loader is not None:
            t_total, t_correct, t_loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for a, v, y in tqdm(test_loader, desc=f"Fold {fold_idx} Test {epoch}/{args.epochs}", leave=False, dynamic_ncols=True):
                    a, v, y = a.to(device), v.to(device), y.to(device)
                    logits = model(a, v)
                    loss = crit(logits, y)
                    t_loss_sum += float(loss.item()) * a.size(0)
                    t_correct += int((logits.argmax(dim=1) == y).sum().item())
                    t_total += a.size(0)
            test_loss = t_loss_sum / max(1, t_total)
            test_acc = t_correct / max(1, t_total)

        fold_history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": macro_f1,
            "test_loss": test_loss, "test_acc": test_acc,
            "per_class_f1": per_class_f1,
            "confusion": conf,
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_metrics = {
                "loss": val_loss,
                "acc": val_acc,
                "macro_f1": macro_f1,
                "per_class_f1": per_class_f1,
                "confusion": conf,
                "n": total,
            }
            best_confusion = conf
            best_train_metrics = {"loss": train_loss, "acc": train_acc}
            best_state_dict = {
                "state_dict": model.state_dict(),
                "d_a": d_a, "d_v": d_v,
            }

    return best_val_metrics, best_confusion, best_train_metrics, best_state_dict, fold_history


def cross_validate_gated(args):
    pairs = find_emb_pairs(args.audio_emb_root, args.video_emb_root)
    if not pairs:
        raise SystemExit("No matched audio/video embeddings found.")

    labels = sorted({lbl for lbl, _, _ in pairs})
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    # Hold out a real test set ONCE, before any folding happens
    trainval_items, _, test_items = split_train_val_test(
        pairs, val_ratio=0.0, test_ratio=args.test_ratio, seed=args.seed
    )
    if not trainval_items:
        raise SystemExit("No train/val items left after carving out the test set.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Probe embedding dims once, from the first train/val item
    a0 = torch.load(trainval_items[0][1]).view(-1)
    v0 = torch.load(trainval_items[0][2]).view(-1)
    d_a, d_v = int(a0.numel()), int(v0.numel())

    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    fold_losses, fold_accs, fold_f1s = [], [], []
    fold_train_accs, fold_gaps = [], []
    best_overall_score = -1.0
    all_fold_histories = []
    best_overall_state = None

    os.makedirs(args.out_dir, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(trainval_items), start=1):
        train_items = [trainval_items[i] for i in train_idx]
        val_items = [trainval_items[i] for i in val_idx]

        metrics, conf, train_metrics, state_dict, fold_history = train_one_fold_gated(
            train_items, val_items, label_to_id, id_to_label, args, device,
            d_a, d_v, fold_idx, test_items=test_items,
        )
        gap = train_metrics["acc"] - metrics["acc"]
        print(f"Fold {fold_idx} metrics: Loss {metrics['loss']:.4f}, Acc {metrics['acc']:.4f}, "
              f"Macro F1 {metrics['macro_f1']:.4f} | Train Acc {train_metrics['acc']:.4f} "
              f"(gap={gap:.4f})")

        print_confusion_matrix(conf, id_to_label)
        cm_path = os.path.join(args.out_dir, f"confusion_matrix_fold{fold_idx}.png")
        save_normalized_confusion_matrix(conf, id_to_label, cm_path,
                                          title=f"Fold {fold_idx} Confusion Matrix (Normalized)")

        conv_path = os.path.join(args.out_dir, f"convergence_fold{fold_idx}.png")
        plot_fold_convergence(fold_history, conv_path, fold_idx)
        all_fold_histories.append(fold_history)

        fold_losses.append(metrics['loss'])
        fold_accs.append(metrics['acc'])
        fold_f1s.append(metrics['macro_f1'])
        fold_train_accs.append(train_metrics['acc'])
        fold_gaps.append(gap)

        score = (metrics["macro_f1"] + metrics["acc"]) / 2.0
        if score > best_overall_score:
            best_overall_score = score
            best_overall_state = state_dict

    print("\n=== Cross-Validation Summary (Gated Fusion) ===")
    print(f"Mean val loss: {statistics.mean(fold_losses):.4f} ± {statistics.stdev(fold_losses):.4f}")
    print(f"Mean val acc:  {statistics.mean(fold_accs):.4f} ± {statistics.stdev(fold_accs):.4f}")
    print(f"Mean macroF1:  {statistics.mean(fold_f1s):.4f} ± {statistics.stdev(fold_f1s):.4f}")
    print(f"Mean train acc: {statistics.mean(fold_train_accs):.4f}")
    print(f"Mean train-val gap: {statistics.mean(fold_gaps):.4f} ± {statistics.stdev(fold_gaps):.4f}")

    # Persist the best fold's model to disk
    if best_overall_state is not None:
        ckpt_path = os.path.join(args.out_dir, "gated_fusion_best_fold.pt")
        torch.save({
            "state_dict": best_overall_state["state_dict"],
            "labels": labels,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "d_a": best_overall_state["d_a"],
            "d_v": best_overall_state["d_v"],
            "proj_dim": args.proj_dim,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "normalize": not args.no_normalize,
            "best_overall_score": best_overall_score,
        }, ckpt_path)
        print(f"Saved best-fold model checkpoint to {ckpt_path}")

    # Final, single, honest test-set evaluation using the best fold's model
    if test_items:
        model = GatedFusionClassifier(
            d_a=best_overall_state["d_a"], d_v=best_overall_state["d_v"],
            num_classes=len(labels), proj_dim=args.proj_dim,
            hidden=args.hidden, dropout=args.dropout, normalize=not args.no_normalize,
        ).to(device)
        model.load_state_dict(best_overall_state["state_dict"])
        model.eval()

        test_ds = EmbPairEmbeddingsDataset(test_items, label_to_id)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        crit = nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0
        all_true, all_pred = [], []
        with torch.no_grad():
            for a, v, y in test_loader:
                a, v, y = a.to(device), v.to(device), y.to(device)
                logits = model(a, v)
                loss_sum += float(crit(logits, y).item()) * a.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += a.size(0)
                all_true.extend(y.tolist())
                all_pred.extend(preds.tolist())
        test_conf, _, test_macro_f1 = confusion_and_f1(len(labels), all_true, all_pred)
        print(f"\n=== Held-out TEST (best fold model) ===")
        print(f"Test loss {loss_sum/max(1,total):.4f} acc {correct/max(1,total):.4f} macroF1 {test_macro_f1:.4f}")
        print_confusion_matrix(test_conf, id_to_label)

    with open(os.path.join(args.out_dir, "fold_epoch_history.json"), "w") as f:
        json.dump(all_fold_histories, f, indent=2)

    with open(os.path.join(args.out_dir, "cv_results.json"), "w") as f:
        json.dump({
            "fold_losses": fold_losses, "fold_accs": fold_accs, "fold_f1s": fold_f1s,
            "fold_train_accs": fold_train_accs, "fold_gaps": fold_gaps,
            "mean_loss": statistics.mean(fold_losses), "std_loss": statistics.stdev(fold_losses),
            "mean_acc": statistics.mean(fold_accs), "std_acc": statistics.stdev(fold_accs),
            "mean_f1": statistics.mean(fold_f1s), "std_f1": statistics.stdev(fold_f1s),
        }, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Train classifier with normalized projections + gated fusion on saved embeddings")
    p.add_argument("audio_emb_root", help="Root dir of saved audio embeddings (.pt) grouped by label")
    p.add_argument("video_emb_root", help="Root dir of saved video embeddings (.pt) grouped by label")
    p.add_argument("--out_dir", default="models_gated")
    p.add_argument("--proj_dim", type=int, default=512)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no_normalize", action="store_true", help="Disable L2 normalization of inputs and fused vector")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_history", default="metrics_history.json")
    p.add_argument("--eval_test_each_epoch", action="store_true", help="Evaluate test set every epoch (slower)")
    # NEW: K-fold cross-validation option, same flag name as the baseline script
    p.add_argument("--cv_folds", type=int, default=0,
                    help="Number of folds for K-fold cross-validation. If > 1, runs cross_validate_gated() instead of a single train() run.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cv_folds and args.cv_folds > 1:
        cross_validate_gated(args)
    else:
        train(args)