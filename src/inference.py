import argparse
import json
import os
from typing import Dict, Tuple

import torch

from audio_embeddings import extract_audio_embedding
from video_embeddings import extract_video_embedding
from train_baseline import FusionMLP


def _load_checkpoint(model_path: str, device: torch.device) -> Tuple[FusionMLP, Dict[int, str]]:
    ckpt = torch.load(model_path, map_location=device)
    id_to_label = ckpt.get("id_to_label") or {}
    # Keys might be strings -> convert to ints
    label_map: Dict[int, str] = {}
    for k, v in id_to_label.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            idx = int(v)
            v = k
        label_map[idx] = v

    in_dim = ckpt["in_dim"]
    hidden = ckpt.get("hidden", 512)

    model = FusionMLP(in_dim=in_dim, num_classes=len(label_map), hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, label_map


def predict(
    audio_path: str,
    video_path: str,
    model_path: str,
    audio_model: str = "facebook/wav2vec2-base",
    video_model: str = "facebook/vjepa2-vitl-fpc64-256",
    num_frames: int = 32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[str, Dict[str, float]]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    model, id_to_label = _load_checkpoint(model_path, device)

    audio_emb = extract_audio_embedding(audio_path, model_name=audio_model, device=device)
    video_emb = extract_video_embedding(video_path, model_name=video_model, device=device, num_frames=num_frames)
    fused = torch.cat([audio_emb, video_emb], dim=0).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(fused)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    top_idx = int(probs.argmax().item())
    label = id_to_label.get(top_idx, str(top_idx))
    prob_dict = {id_to_label.get(i, str(i)): float(p) for i, p in enumerate(probs.tolist())}
    return label, prob_dict


def main():
    parser = argparse.ArgumentParser(description="Run emotion inference from a multimodal checkpoint.")
    parser.add_argument("model_path", help="Path to trained FusionMLP checkpoint (.pt).")
    parser.add_argument("audio_path", help="Path to input audio file.")
    parser.add_argument("video_path", help="Path to input video file.")
    parser.add_argument("--audio_model", default="facebook/wav2vec2-base", help="Audio embedding model/preset.")
    parser.add_argument("--video_model", default="facebook/vjepa2-vitl-fpc64-256", help="Video embedding model.")
    parser.add_argument("--num_frames", type=int, default=32, help="Frames sampled from the video.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label, probs = predict(
        audio_path=args.audio_path,
        video_path=args.video_path,
        model_path=args.model_path,
        audio_model=args.audio_model,
        video_model=args.video_model,
        num_frames=args.num_frames,
        device=device,
    )
    print(f"Predicted emotion: {label}")
    print("Class probabilities:")
    for name, score in probs.items():
        print(f"  {name}: {score:.4f}")


if __name__ == "__main__":
    main()
