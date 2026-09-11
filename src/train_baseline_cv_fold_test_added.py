import os
import json
import argparse
import pathlib
from typing import List, Tuple, Dict, Optional
import subprocess

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from audio_embeddings import extract_audio_embedding
from video_embeddings import extract_video_embedding
from sklearn.model_selection import KFold
import statistics

import matplotlib
matplotlib.use("Agg")  # no display needed, just saves to file
import matplotlib.pyplot as plt
import numpy as np



def _list_files(root: str, exts: set) -> List[str]:
    paths = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if pathlib.Path(fn).suffix.lower() in exts:
                paths.append(os.path.join(dp, fn))
    return paths


def _by_stem(files: List[str]) -> Dict[str, str]:   
    return {pathlib.Path(p).stem: p for p in files}


def find_pairs(audio_root: str, video_root: str) -> List[Tuple[str, str, str]]:
    """ 
    Return list of (label, audio_path, video_path) for basenames present in both trees.
    Assumes structure: <root>/<label>/**/<file>.
    """
    pairs = []
    audio_exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}

    # Iterate labels from intersection of top-level subdirs
    audio_labels = {d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d))}
    video_labels = {d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))}
    labels = sorted(audio_labels & video_labels)

    for label in labels:
        a_files = _list_files(os.path.join(audio_root, label), audio_exts)
        v_files = _list_files(os.path.join(video_root, label), video_exts)
        v_map = _by_stem(v_files)
        for ap in a_files:
            stem = pathlib.Path(ap).stem
            vp = v_map.get(stem)
            if vp:
                pairs.append((label, ap, vp))
    return pairs


class EmbeddingPairDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[str, str, str]],
        label_to_id: Dict[str, int],
        audio_cache: Optional[str] = None,
        video_cache: Optional[str] = None,
        audio_model: str = "facebook/wav2vec2-base",
        video_model: str = "facebook/vjepa2-vitl-fpc64-256",
        device: Optional[torch.device] = None,
        num_frames: int = 32,
        convert_mp3_to_wav: bool = False,
        converted_wav_cache: Optional[str] = None,
        ffmpeg_path: str = "ffmpeg",
    ):
        self.items = items
        self.label_to_id = label_to_id
        self.audio_cache = audio_cache
        self.video_cache = video_cache
        self.audio_model = audio_model
        self.video_model = video_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.convert_mp3_to_wav = convert_mp3_to_wav
        self.converted_wav_cache = converted_wav_cache
        self.ffmpeg_path = ffmpeg_path

    def __len__(self):
        return len(self.items)

    def _cache_path(self, cache_root: str, label: str, src_path: str) -> str:
        rel_name = pathlib.Path(src_path).stem + ".pt"
        target = pathlib.Path(cache_root) / label
        target.mkdir(parents=True, exist_ok=True)
        return str(target / rel_name)

    def __getitem__(self, idx):
        label, a_path, v_path = self.items[idx]

        # Optionally convert MP3 -> WAV (16k mono) once, reuse from cache
        a_input_path = a_path
        if self.convert_mp3_to_wav and pathlib.Path(a_path).suffix.lower() == ".mp3":
            a_input_path = self._ensure_wav16k_cached(label, a_path)

        # Audio embedding (cached if possible)
        if self.audio_cache:
            a_out = self._cache_path(self.audio_cache, label, a_path)
            if os.path.exists(a_out):
                a_emb = torch.load(a_out)
            else:
                a_emb = extract_audio_embedding(a_input_path, model_name=self.audio_model, device=self.device)
                torch.save(a_emb, a_out)
        else:
            a_emb = extract_audio_embedding(a_input_path, model_name=self.audio_model, device=self.device)

        # Video embedding (cached if possible)
        if self.video_cache:
            v_out = self._cache_path(self.video_cache, label, v_path)
            if os.path.exists(v_out):
                v_emb = torch.load(v_out)
            else:
                v_emb = extract_video_embedding(v_path, model_name=self.video_model, device=self.device, num_frames=self.num_frames)
                torch.save(v_emb, v_out)
        else:
            v_emb = extract_video_embedding(v_path, model_name=self.video_model, device=self.device, num_frames=self.num_frames)

        x = torch.cat([a_emb, v_emb], dim=0).float()
        y = torch.tensor(self.label_to_id[label], dtype=torch.long)
        return x, y

    def _ensure_wav16k_cached(self, label: str, mp3_path: str) -> str:
        """
        Convert an MP3 to 16kHz mono WAV into a cache folder once and return the WAV path.
        Falls back to the original MP3 on failure.
        """
        cache_root = self.converted_wav_cache or os.path.join(
            os.path.dirname(mp3_path), ".converted_wav16k"
        )
        stem = pathlib.Path(mp3_path).stem
        target_dir = pathlib.Path(cache_root) / label
        target_dir.mkdir(parents=True, exist_ok=True)
        wav_path = str(target_dir / f"{stem}.wav")

        if os.path.exists(wav_path):
            return wav_path

        cmd = [
            self.ffmpeg_path, "-y", "-i", mp3_path,
            "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return wav_path
        except Exception:
            # If ffmpeg isn't available or conversion fails, just use the original MP3
            return mp3_path


class FusionMLP(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 512, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def split_train_val_test(
    items: List[Tuple[str, str, str]],
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List, List, List]:
    rng = torch.Generator()
    rng.manual_seed(seed)
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

def train_one_fold(train_items, val_items, label_to_id, id_to_label, args, device, fold_idx=None, test_items=None):
    train_ds = EmbeddingPairDataset(
        train_items, label_to_id,
        audio_cache=args.audio_cache, video_cache=args.video_cache,
        audio_model=args.audio_model, video_model=args.video_model,
        device=device, num_frames=args.num_frames,
        convert_mp3_to_wav=args.convert_mp3_to_wav,
        converted_wav_cache=args.converted_wav_cache,
        ffmpeg_path=args.ffmpeg_path,
    )
    val_ds = EmbeddingPairDataset(
        val_items, label_to_id,
        audio_cache=args.audio_cache, video_cache=args.video_cache,
        audio_model=args.audio_model, video_model=args.video_model,
        device=device, num_frames=args.num_frames,
        convert_mp3_to_wav=args.convert_mp3_to_wav,
        converted_wav_cache=args.converted_wav_cache,
        ffmpeg_path=args.ffmpeg_path,
    )

    test_ds = None
    test_loader = None
    if test_items:
        test_ds = EmbeddingPairDataset(
            test_items, label_to_id,
            audio_cache=args.audio_cache, video_cache=args.video_cache,
            audio_model=args.audio_model, video_model=args.video_model,
            device=device, num_frames=args.num_frames,
            convert_mp3_to_wav=args.convert_mp3_to_wav,
            converted_wav_cache=args.converted_wav_cache,
            ffmpeg_path=args.ffmpeg_path,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)


    x0, _ = train_ds[0]
    in_dim = x0.numel()
    num_classes = len(label_to_id)

    model = FusionMLP(in_dim=in_dim, num_classes=num_classes, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    # Training loop would go here
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
        for xb, yb in tqdm(train_loader, desc=f"Fold {fold_idx} Train {epoch}/{args.epochs}", leave=False, dynamic_ncols=True):
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += float(loss.item()) * xb.size(0)
            tr_correct += int((logits.argmax(dim=1) == yb).sum().item())
            tr_total += xb.size(0)
        train_acc = tr_correct / max(1, tr_total)
        train_loss = tr_loss_sum / max(1, tr_total)


        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Fold {fold_idx} Val {epoch}/{args.epochs}", leave=False, dynamic_ncols=True):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                loss_sum += float(loss.item()) * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += xb.size(0)
                all_true.extend(yb.tolist())
                all_pred.extend(preds.tolist())
        val_loss = loss_sum / max(1, total)
        val_acc = correct / max(1, total)
        conf, per_class_f1, macro_f1 = confusion_and_f1(num_classes, all_true, all_pred)

        test_loss, test_acc = None, None
        if test_loader is not None:
            t_total, t_correct, t_loss_sum = 0, 0, 0.0
            with torch.no_grad():
                for xb, yb in tqdm(test_loader, desc=f"Fold {fold_idx} Test {epoch}/{args.epochs}", leave=False, dynamic_ncols=True):
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = crit(logits, yb)
                    t_loss_sum += float(loss.item()) * xb.size(0)
                    t_correct += int((logits.argmax(dim=1) == yb).sum().item())
                    t_total += xb.size(0)
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
            best_state_dict = {"state_dict": model.state_dict(), "in_dim": in_dim}
    return best_val_metrics, best_confusion, best_train_metrics, best_state_dict, fold_history

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

def cross_validate(args):
    pairs = find_pairs(args.audio_root, args.video_root)
    if not pairs:
        raise SystemExit("No matched audio/video pairs found.")

    labels = sorted({lbl for lbl, _, _ in pairs})
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    # Hold out a real test set ONCE, before any folding happens
    trainval_items, _, test_items = split_train_val_test(
        pairs, val_ratio=0.0, test_ratio=args.test_ratio, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)

    fold_losses, fold_accs, fold_f1s = [], [], []
    fold_train_accs, fold_gaps = [], []
    best_overall_score = -1.0
    all_fold_histories = []   # NEW
    best_overall_state = None

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(trainval_items), start=1):
        train_items = [trainval_items[i] for i in train_idx]
        val_items = [trainval_items[i] for i in val_idx]

        metrics, conf, train_metrics, state_dict , fold_history= train_one_fold(
            train_items, val_items, label_to_id, id_to_label, args, device, fold_idx, test_items=test_items
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

    print("\n=== Cross-Validation Summary ===")
    print(f"Mean val loss: {statistics.mean(fold_losses):.4f} ± {statistics.stdev(fold_losses):.4f}")
    print(f"Mean val acc:  {statistics.mean(fold_accs):.4f} ± {statistics.stdev(fold_accs):.4f}")
    print(f"Mean macroF1:  {statistics.mean(fold_f1s):.4f} ± {statistics.stdev(fold_f1s):.4f}")
    print(f"Mean train acc: {statistics.mean(fold_train_accs):.4f}")
    print(f"Mean train-val gap: {statistics.mean(fold_gaps):.4f} ± {statistics.stdev(fold_gaps):.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    if best_overall_state is not None:
        ckpt_path = os.path.join(args.out_dir, "multimodal_classifier_best_fold.pt")
        torch.save({
            "state_dict": best_overall_state["state_dict"],
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
            "in_dim": best_overall_state["in_dim"],
            "hidden": args.hidden,
            "dropout": args.dropout,
            "best_overall_score": best_overall_score,
        }, ckpt_path)
        print(f"Saved best-fold model checkpoint to {ckpt_path}")

    # Final, single, honest test-set evaluation using the best fold's model
    if test_items:
        model = FusionMLP(
            in_dim=best_overall_state["in_dim"], num_classes=len(labels),
            hidden=args.hidden, dropout=args.dropout,
        ).to(device)
        model.load_state_dict(best_overall_state["state_dict"])
        model.eval()

        test_ds = EmbeddingPairDataset(
            test_items, label_to_id,
            audio_cache=args.audio_cache, video_cache=args.video_cache,
            audio_model=args.audio_model, video_model=args.video_model,
            device=device, num_frames=args.num_frames,
            convert_mp3_to_wav=args.convert_mp3_to_wav,
            converted_wav_cache=args.converted_wav_cache,
            ffmpeg_path=args.ffmpeg_path,
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        crit = nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0
        all_true, all_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_sum += float(crit(logits, yb).item()) * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += xb.size(0)
                all_true.extend(yb.tolist())
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

    # Also print per-row accuracy (how well each true class is recalled)
    print("Per-class recall (diagonal / row sum):")
    for i, row in enumerate(conf):
        row_sum = sum(row)
        recall = row[i] / row_sum if row_sum > 0 else 0.0
        print(f"  {labels[i]:<15} recall={recall:.3f}  (n={row_sum})")



def save_normalized_confusion_matrix(conf, id_to_label, out_path, title="Confusion Matrix (Normalized)"):
    """
    conf: list of lists (raw counts), rows = true label, cols = predicted label
    Normalizes each row by its sum (i.e., shows recall per true class).
    """
    conf_arr = np.array(conf, dtype=float)
    row_sums = conf_arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid div by zero
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

    # Annotate each cell with its normalized value
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


def train(args):
    pairs = find_pairs(args.audio_root, args.video_root)
    if not pairs:
        raise SystemExit("No matched audio/video pairs found.")

    labels = sorted({lbl for lbl, _, _ in pairs})
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    if args.val_ratio + args.test_ratio >= 0.9:
        raise SystemExit("val_ratio + test_ratio too large; leave room for train set.")
    train_items, val_items, test_items = split_train_val_test(
        pairs, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EmbeddingPairDataset(
        train_items, label_to_id,
        audio_cache=args.audio_cache, video_cache=args.video_cache,
        audio_model=args.audio_model, video_model=args.video_model,
        device=device, num_frames=args.num_frames,
        convert_mp3_to_wav=args.convert_mp3_to_wav,
        converted_wav_cache=args.converted_wav_cache,
        ffmpeg_path=args.ffmpeg_path,
    )
    print(len(train_ds))
    val_ds = EmbeddingPairDataset(
        val_items, label_to_id,
        audio_cache=args.audio_cache, video_cache=args.video_cache,
        audio_model=args.audio_model, video_model=args.video_model,
        device=device, num_frames=args.num_frames,
        convert_mp3_to_wav=args.convert_mp3_to_wav,
        converted_wav_cache=args.converted_wav_cache,
        ffmpeg_path=args.ffmpeg_path,
    )
    print(len(val_ds))
    test_ds = EmbeddingPairDataset(
        test_items, label_to_id,
        audio_cache=args.audio_cache, video_cache=args.video_cache,
        audio_model=args.audio_model, video_model=args.video_model,
        device=device, num_frames=args.num_frames,
        convert_mp3_to_wav=args.convert_mp3_to_wav,
        converted_wav_cache=args.converted_wav_cache,
        ffmpeg_path=args.ffmpeg_path,
    )
    print(len(test_ds))

    # Optional: precompute and cache embeddings with progress bars
    if hasattr(args, "precompute_embeddings") and args.precompute_embeddings:
        def _precompute(ds: Dataset, name: str):
            for i in tqdm(range(len(ds)), desc=f"Precompute {name}", leave=False, dynamic_ncols=True):
                _ = ds[i]

        _precompute(train_ds, "train")
        if len(val_ds) > 0:
            _precompute(val_ds, "val")
        if len(test_ds) > 0:
            _precompute(test_ds, "test")

    # Peek dims (uses cached embeddings if precomputed)
    x0, _ = train_ds[0]
    in_dim = x0.numel()

    model = FusionMLP(in_dim=in_dim, num_classes=len(labels), hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    def run_epoch(loader: DataLoader, train: bool, desc: str):
        model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
            xb = xb.to(device)
            yb = yb.to(device)
            if train:
                opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            if train:
                loss.backward()
                opt.step()
            loss_sum += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += xb.size(0)
        return loss_sum / max(1, total), correct / max(1, total)

    def evaluate(loader: DataLoader, desc: str):
        model.eval()
        total, correct, loss_sum = 0, 0, 0.0
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for xb, yb in tqdm(loader, desc=desc, leave=False, dynamic_ncols=True):
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = crit(logits, yb)
                loss_sum += float(loss.item()) * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += xb.size(0)
                all_true.extend(yb.tolist())
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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    best_metric = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
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
                f" | test loss {te_metrics_epoch['loss']:.4f} acc {te_metrics_epoch['acc']:.4f}"
            )
        print(log_msg)
        # Track best by macro-F1, fallback to acc
        score = (va_metrics["macro_f1"] + va_metrics["acc"]) / 2.0
        if score > best_metric:
            best_metric = score
            torch.save({
                "state_dict": model.state_dict(),
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "in_dim": in_dim,
                "hidden": args.hidden,
            }, os.path.join(args.out_dir, "multimodal_classifier.pt"))

        # Append to history
        history.append({
            "epoch": epoch,
            "train": {"loss": tr_loss, "acc": tr_acc},
            "val": va_metrics,
            **({"test": te_metrics_epoch} if te_metrics_epoch is not None else {}),
        })

    # Save label map for reference
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump({"label_to_id": label_to_id, "id_to_label": id_to_label}, f, indent=2)
    # Final evaluation on test split
    te_metrics = evaluate(test_loader, desc="Test")
    metrics_out = {
        "val_best_score": best_metric,
        "test": te_metrics,
        "labels": labels,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(
        f"Test: loss {te_metrics['loss']:.4f} acc {te_metrics['acc']:.4f} macroF1 {te_metrics['macro_f1']:.4f}"
    )
    # Save history if requested or if test was evaluated each epoch
    if history and getattr(args, "log_history", None):
        hist_path = os.path.join(args.out_dir, args.log_history)
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Train multimodal (audio+video) emotion classifier from embeddings")
    p.add_argument("audio_root", help="Root of audio files grouped by label (e.g., wav_data)")
    p.add_argument("video_root", help="Root of video files grouped by label (e.g., video_data)")
    p.add_argument("--audio_cache", default="audio_emb", help="Directory to cache/load audio embeddings (.pt)")
    p.add_argument("--video_cache", default="video_emb", help="Directory to cache/load video embeddings (.pt)")
    p.add_argument(
        "--audio_model",
        default="facebook/wav2vec2-base",
        help=(
            "Audio embedding model preset or Hugging Face repo id "
            "(e.g., wav2vec2-base, facebook/wav2vec2-base, whisper-small, openai/whisper-base)."
        ),
    )
    p.add_argument("--video_model", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--num_frames", type=int, default=32)
    p.add_argument("--out_dir", default="models")
    p.add_argument("--cv_folds", type=int, default=0, help="Number of folds for cross-validation")
    # Optional: convert MP3s once to 16k mono WAV before embedding extraction
    p.add_argument("--convert_mp3_to_wav", action="store_true",
                   help="Convert .mp3 to 16k mono .wav into a cache and use that for audio embeddings")
    p.add_argument("--converted_wav_cache", default=".cache/converted_wav16k",
                   help="Directory where converted WAVs will be stored if --convert_mp3_to_wav is set")
    p.add_argument("--ffmpeg_path", default="ffmpeg",
                   help="Path to ffmpeg binary used for conversion when --convert_mp3_to_wav is set")
    p.add_argument("--precompute_embeddings", action="store_true",
                   help="Precompute and cache embeddings for all splits with tqdm before training")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_test_each_epoch", action="store_true",
                   help="Also evaluate test set every epoch and log its metrics (slower)")
    p.add_argument("--log_history", default="metrics_history.json",
                   help="Filename saved under out_dir with per-epoch metrics history")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.cv_folds and args.cv_folds > 1:
        cross_validate(args)
    else:
        train(args)
