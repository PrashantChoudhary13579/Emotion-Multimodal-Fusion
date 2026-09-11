import os
import json
import argparse
from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion(metrics_data: Dict[str, Any], out_path: Optional[str] = None, title: str = "Confusion Matrix"):
    labels: List[str] = metrics_data.get("labels", [])
    conf = metrics_data["test"]["confusion"]

    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf, annot=True, fmt="d", cmap="Blues",
                     xticklabels=labels, yticklabels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(metrics_path) or ".", "confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved confusion matrix to {out_path}")


def plot_loss_history(history_path: str, which_sets: List[str], out_path: Optional[str] = None, title: str = "Loss vs Epoch", metrics_data: Optional[Dict[str, Any]] = None):
    with open(history_path, "r") as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    for s in which_sets:
        if s not in ("train", "val", "test"):
            continue
        losses = [h[s]["loss"] for h in history if s in h and isinstance(h[s], dict) and "loss" in h[s]]
        if not losses:
            if s == "test" and metrics_data:
                test_loss = metrics_data.get("test", {}).get("loss")
                if test_loss is not None and epochs:
                    start = epochs[0]
                    end = epochs[-1]
                else:
                    start = end = None
                if start is not None:
                    plt.plot([start, end], [test_loss, test_loss], linestyle="--", label="test loss (final)")
                    continue
            print(f"No '{s}' loss found in history; skipping.")
            continue
        plt.plot(epochs[:len(losses)], losses, label=f"{s} loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(history_path) or ".", "loss_convergence.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved loss convergence plot to {out_path}")


def plot_acc_history(history_path: str, which_sets: List[str], out_path: Optional[str] = None, title: str = "Accuracy vs Epoch", metrics_data: Optional[Dict[str, Any]] = None):
    with open(history_path, "r") as f:
        history = json.load(f)
    epochs = [h["epoch"] for h in history]

    plt.figure(figsize=(8, 5))
    for s in which_sets:
        if s not in ("train", "val", "test"):
            continue
        accs = [h[s]["acc"] for h in history if s in h and isinstance(h[s], dict) and "acc" in h[s]]
        if not accs:
            if s == "test" and metrics_data:
                test_acc = metrics_data.get("test", {}).get("acc")
                if test_acc is not None and epochs:
                    start = epochs[0]
                    end = epochs[-1]
                else:
                    start = end = None
                if start is not None:
                    plt.plot([start, end], [test_acc, test_acc], linestyle="--", label="test acc (final)")
                    continue
            print(f"No '{s}' accuracy found in history; skipping.")
            continue
        plt.plot(epochs[:len(accs)], accs, label=f"{s} acc")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if out_path is None:
        out_path = os.path.join(os.path.dirname(history_path) or ".", "accuracy_convergence.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved accuracy convergence plot to {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot confusion matrix and loss curves from saved metrics")
    ap.add_argument("--metrics", default="models/metrics.json", help="Path to metrics.json containing test confusion")
    ap.add_argument("--history", default=None, help="Path to metrics history JSON (from trainer with logging)")
    ap.add_argument("--out_dir", default=None, help="Output directory for plots; defaults next to input files")
    ap.add_argument("--sets", nargs="+", default=["train", "val", "test"], help="Which sets to plot curves for")
    ap.add_argument("--conf_title", default="Confusion Matrix")
    ap.add_argument("--loss_title", default="Loss vs Epoch")
    ap.add_argument("--acc_title", default="Accuracy vs Epoch")
    args = ap.parse_args()

    # Confusion matrix from metrics.json
    metrics_data = None
    if args.metrics and os.path.exists(args.metrics):
        with open(args.metrics, "r") as f:
            metrics_data = json.load(f)
        cm_out = None
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            cm_out = os.path.join(args.out_dir, "confusion_matrix.png")
        plot_confusion(metrics_data, out_path=cm_out, title=args.conf_title)
    else:
        print("No metrics file found or path invalid; skipping confusion matrix.")

    # Loss curves from history JSON
    if args.history:
        if os.path.exists(args.history):
            loss_out = None
            if args.out_dir:
                os.makedirs(args.out_dir, exist_ok=True)
                loss_out = os.path.join(args.out_dir, "loss_convergence.png")
            plot_loss_history(args.history, which_sets=args.sets, out_path=loss_out, title=args.loss_title, metrics_data=metrics_data)
            # Also plot accuracy curves
            acc_out = None
            if args.out_dir:
                acc_out = os.path.join(args.out_dir, "accuracy_convergence.png")
            plot_acc_history(args.history, which_sets=args.sets, out_path=acc_out, title=args.acc_title, metrics_data=metrics_data)
        else:
            print("History file not found; skipping loss plot.")


if __name__ == "__main__":
    main()
