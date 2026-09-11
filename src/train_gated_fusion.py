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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
