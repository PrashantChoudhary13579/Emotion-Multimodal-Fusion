import argparse
import json
import os
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from audio_embeddings import extract_audio_embedding
from data_loaders import discover_modalities, find_pairs_from_roots
from video_embeddings import extract_video_embedding


def _cache_path(cache_root: str, label: str, src_path: str) -> str:
    base = pathlib.Path(src_path)
    target_dir = pathlib.Path(cache_root) / label
    target_dir.mkdir(parents=True, exist_ok=True)
    return str(target_dir / (base.stem + ".pt"))


def _save_tensor(path: Optional[str], tensor: torch.Tensor):
    if path is None:
        return
    torch.save(tensor.cpu(), path)


def _load_cached(path: Optional[str]) -> Optional[torch.Tensor]:
    if path is None or (not os.path.exists(path)):
        return None
    return torch.load(path, map_location="cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge Whisper audio embeddings with V-JEPA video embeddings for training caches"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--dataset_root",
        default=None,
        help="Root folder containing both audio and video subdirectories grouped by label",
    )
    group.add_argument(
        "--roots",
        nargs=2,
        metavar=("AUDIO_ROOT", "VIDEO_ROOT"),
        default=None,
        help="Explicit audio and video roots grouped by label",
    )
    parser.add_argument(
        "--audio_out",
        default=None,
        help="Directory to save Whisper audio embeddings. Defaults to <dataset_root>/.cache/whisper_audio",
    )
    parser.add_argument(
        "--video_out",
        default=None,
        help="Directory to save V-JEPA video embeddings. Defaults to <dataset_root>/.cache/vjepa_video",
    )
    parser.add_argument(
        "--merged_out",
        default=None,
        help="Optional directory to save concatenated audio+video embeddings",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON file summarizing cached file paths per sample",
    )
    parser.add_argument(
        "--whisper_model",
        default="whisper-medium",
        help="Whisper model preset or repo id (e.g., whisper-small, openai/whisper-base)",
    )
    parser.add_argument(
        "--vjepa_model",
        default="facebook/vjepa2-vitl-fpc64-256",
        help="V-JEPA model repo id",
    )
    parser.add_argument("--num_frames", type=int, default=32, help="Frames sampled per video clip")
    parser.add_argument("--skip_audio", action="store_true", help="Skip audio embedding extraction")
    parser.add_argument("--skip_video", action="store_true", help="Skip video embedding extraction")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached embeddings")
    parser.add_argument(
        "--no_merge_cast",
        action="store_true",
        help="Do not force merged embeddings to float32 (defaults to casting for training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.roots is not None:
        audio_root, video_root = args.roots
        dataset_root = os.path.commonpath([audio_root, video_root])
    else:
        dataset_root = args.dataset_root
        audio_root, video_root = discover_modalities(dataset_root)

    pairs = find_pairs_from_roots(audio_root, video_root)
    if not pairs:
        raise SystemExit("No matched audio/video pairs found.")

    audio_out = args.audio_out or os.path.join(dataset_root, ".cache", "whisper_audio")
    video_out = args.video_out or os.path.join(dataset_root, ".cache", "vjepa_video")
    merged_out = args.merged_out
    if merged_out:
        os.makedirs(merged_out, exist_ok=True)

    os.makedirs(audio_out, exist_ok=True)
    os.makedirs(video_out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest: List[Dict[str, str]] = [] if args.manifest else None

    audio_counts = {"done": 0, "skipped": 0, "failed": 0}
    video_counts = {"done": 0, "skipped": 0, "failed": 0}
    merged_counts = {"done": 0, "skipped": 0, "failed": 0}

    for label, a_path, v_path in tqdm(pairs, desc="Merging embeddings", dynamic_ncols=True):
        audio_cache = _cache_path(audio_out, label, a_path)
        video_cache = _cache_path(video_out, label, v_path)
        merged_cache = _cache_path(merged_out, label, a_path) if merged_out else None

        audio_emb: Optional[torch.Tensor] = None
        video_emb: Optional[torch.Tensor] = None

        # Audio extraction or load
        if audio_cache and (not args.skip_audio):
            if os.path.exists(audio_cache) and (not args.overwrite):
                audio_counts["skipped"] += 1
            else:
                try:
                    audio_emb = extract_audio_embedding(
                        a_path, model_name=args.whisper_model, device=device
                    )
                    _save_tensor(audio_cache, audio_emb)
                    audio_counts["done"] += 1
                except Exception:
                    audio_counts["failed"] += 1

        # Video extraction or load
        if video_cache and (not args.skip_video):
            if os.path.exists(video_cache) and (not args.overwrite):
                video_counts["skipped"] += 1
            else:
                try:
                    video_emb = extract_video_embedding(
                        v_path,
                        model_name=args.vjepa_model,
                        device=device,
                        num_frames=args.num_frames,
                    )
                    _save_tensor(video_cache, video_emb)
                    video_counts["done"] += 1
                except Exception:
                    video_counts["failed"] += 1

        if merged_cache:
            if os.path.exists(merged_cache) and (not args.overwrite):
                merged_counts["skipped"] += 1
            else:
                try:
                    if audio_emb is None:
                        audio_emb = _load_cached(audio_cache)
                        if audio_emb is None:
                            raise RuntimeError("Missing audio embedding for merge")
                    if video_emb is None:
                        video_emb = _load_cached(video_cache)
                        if video_emb is None:
                            raise RuntimeError("Missing video embedding for merge")

                    a_feat = audio_emb.float() if not args.no_merge_cast else audio_emb
                    v_feat = video_emb.float() if not args.no_merge_cast else video_emb
                    merged = torch.cat([a_feat, v_feat], dim=0)
                    _save_tensor(merged_cache, merged)
                    merged_counts["done"] += 1
                except Exception:
                    merged_counts["failed"] += 1

        if manifest is not None:
            entry: Dict[str, str] = {
                "label": label,
                "audio_path": a_path,
                "video_path": v_path,
                "audio_cache": audio_cache or "",
                "video_cache": video_cache or "",
                "merged_cache": merged_cache or "",
            }
            manifest.append(entry)

    print(
        f"Audio cache -> done {audio_counts['done']} skipped {audio_counts['skipped']} failed {audio_counts['failed']}"
    )
    print(
        f"Video cache -> done {video_counts['done']} skipped {video_counts['skipped']} failed {video_counts['failed']}"
    )
    if merged_out:
        print(
            f"Merged cache -> done {merged_counts['done']} skipped {merged_counts['skipped']} failed {merged_counts['failed']}"
        )

    if manifest is not None and args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote manifest to {args.manifest}")


if __name__ == "__main__":
    main()
