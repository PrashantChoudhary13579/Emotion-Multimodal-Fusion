import os
import argparse
import pathlib
from typing import Optional, List, Tuple

import torch
from tqdm import tqdm

from data_loaders import discover_modalities, find_pairs_from_roots
from audio_embeddings import extract_audio_embedding
from video_embeddings import extract_video_embedding


def _cache_path(cache_root: str, label: str, src_path: str) -> str:
    p = pathlib.Path(src_path)
    out_dir = pathlib.Path(cache_root) / label
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / (p.stem + ".pt"))


def _precompute_audio(pairs: List[Tuple[str, str, str]], cache_root: str, model_name: str, device: torch.device, overwrite: bool):
    done, skipped, failed = 0, 0, 0
    for label, a_path, _ in tqdm(pairs, desc="Audio precompute", dynamic_ncols=True):
        out = _cache_path(cache_root, label, a_path)
        if (not overwrite) and os.path.exists(out):
            skipped += 1
            continue
        try:
            emb = extract_audio_embedding(a_path, model_name=model_name, device=device)
            torch.save(emb, out)
            done += 1
        except Exception:
            failed += 1
    return done, skipped, failed


def _precompute_video(pairs: List[Tuple[str, str, str]], cache_root: str, model_name: str, device: torch.device, num_frames: int, overwrite: bool):
    done, skipped, failed = 0, 0, 0
    for label, _, v_path in tqdm(pairs, desc="Video precompute", dynamic_ncols=True):
        out = _cache_path(cache_root, label, v_path)
        if (not overwrite) and os.path.exists(out):
            skipped += 1
            continue
        try:
            emb = extract_video_embedding(v_path, model_name=model_name, device=device, num_frames=num_frames)
            torch.save(emb, out)
            done += 1
        except Exception:
            failed += 1
    return done, skipped, failed


def parse_args():
    ap = argparse.ArgumentParser(description="Precompute and cache audio+video embeddings for multimodal training")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset_root", default=None, help="Root containing 'Audio*/Video*' subfolders with labels")
    group.add_argument("--roots", nargs=2, metavar=("AUDIO_ROOT", "VIDEO_ROOT"), default=None,
                       help="Explicit audio and video roots grouped by label")
    ap.add_argument("--audio_cache", default=None, help="Directory to save audio embeddings (.pt). Defaults to <dataset_root>/.cache/audio_emb")
    ap.add_argument("--video_cache", default=None, help="Directory to save video embeddings (.pt). Defaults to <dataset_root>/.cache/video_emb")
    ap.add_argument(
        "--audio_model",
        default="facebook/wav2vec2-base",
        help=(
            "Audio embedding model preset or Hugging Face repo id "
            "(e.g., wav2vec2-base, facebook/wav2vec2-base, whisper-small, openai/whisper-base)."
        ),
    )
    ap.add_argument("--video_model", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--num_frames", type=int, default=32)
    ap.add_argument("--skip_audio", action="store_true")
    ap.add_argument("--skip_video", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing cached embeddings")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.roots is not None:
        audio_root, video_root = args.roots
        dataset_root = os.path.commonpath([audio_root, video_root])
    else:
        dataset_root = args.dataset_root
        audio_root, video_root = discover_modalities(dataset_root)

    pairs = find_pairs_from_roots(audio_root, video_root)
    if not pairs:
        raise SystemExit("No matched audio/video pairs found for precompute.")

    # Default caches under dataset root
    audio_cache = args.audio_cache or os.path.join(dataset_root, ".cache", "audio_emb")
    video_cache = args.video_cache or os.path.join(dataset_root, ".cache", "video_emb")
    os.makedirs(audio_cache, exist_ok=True)
    os.makedirs(video_cache, exist_ok=True)

    if not args.skip_audio:
        a_done, a_sk, a_fail = _precompute_audio(
            pairs, cache_root=audio_cache, model_name=args.audio_model, device=device, overwrite=args.overwrite
        )
        print(f"Audio embeddings -> done {a_done}, skipped {a_sk}, failed {a_fail}")

    if not args.skip_video:
        v_done, v_sk, v_fail = _precompute_video(
            pairs, cache_root=video_cache, model_name=args.video_model, device=device, num_frames=args.num_frames, overwrite=args.overwrite
        )
        print(f"Video embeddings -> done {v_done}, skipped {v_sk}, failed {v_fail}")


if __name__ == "__main__":
    main()
