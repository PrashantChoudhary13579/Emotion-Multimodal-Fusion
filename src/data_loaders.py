import os
import pathlib
import random
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from audio_embeddings import extract_audio_embedding
from video_embeddings import extract_video_embedding


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _list_files(root: str, exts: set) -> List[str]:
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if pathlib.Path(fn).suffix.lower() in exts:
                files.append(os.path.join(dp, fn))
    return files


def _guess_modality_subdir(dataset_root: str, modality: str) -> Optional[str]:
    """
    Try to find a subdirectory for the given modality within dataset_root.
    Returns the absolute path or None.
    """
    candidates = {
        "audio": [
            "Audio", "audio", "audios",
            "wav_data", "audio_data", "AudioData", "audioData",
            "Audio Data", "audio data", "Audio_Data", "audio_data",
        ],
        "video": [
            "Video", "video", "videos",
            "video_data", "VideoData", "videoData",
            "Video Data", "video data", "Video_Data", "video_data",
        ],
    }[modality]
    for name in candidates:
        p = os.path.join(dataset_root, name)
        if os.path.isdir(p):
            return p
    # Fallback: choose the subdir with most files of that modality
    best_dir, best_count = None, 0
    for d in os.listdir(dataset_root):
        p = os.path.join(dataset_root, d)
        if os.path.isdir(p):
            if modality == "audio":
                cnt = len(_list_files(p, AUDIO_EXTS))
            else:
                cnt = len(_list_files(p, VIDEO_EXTS))
            if cnt > best_count:
                best_dir, best_count = p, cnt
    return best_dir


def discover_modalities(dataset_root: str) -> Tuple[str, str]:
    """
    Return (audio_root, video_root) under dataset_root by guessing common folder names
    or by counting files of each modality.
    """
    audio_root = _guess_modality_subdir(dataset_root, "audio")
    video_root = _guess_modality_subdir(dataset_root, "video")
    if not audio_root or not video_root:
        raise FileNotFoundError(
            f"Could not locate audio/video subfolders under {dataset_root}.\n"
            f"Expected something like 'Audio/' and 'Video/' with label subfolders."
        )
    return audio_root, video_root


def _by_stem(files: List[str]) -> Dict[str, str]:
    return {pathlib.Path(p).stem: p for p in files}


def find_pairs_from_roots(audio_root: str, video_root: str) -> List[Tuple[str, str, str]]:
    """
    Pair audio and video files by matching basenames within the same label folder.
    Structure expected: <root>/<label>/**/<file> for both modalities.
    Returns list of (label, audio_path, video_path).
    """
    pairs: List[Tuple[str, str, str]] = []
    audio_labels = {d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d))}
    video_labels = {d for d in os.listdir(video_root) if os.path.isdir(os.path.join(video_root, d))}
    labels = sorted(audio_labels & video_labels)
    for label in labels:
        a_files = _list_files(os.path.join(audio_root, label), AUDIO_EXTS)
        v_files = _list_files(os.path.join(video_root, label), VIDEO_EXTS)
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
) -> Tuple[List, List, List]:
    """
    Robust randomized split that ensures non-empty train set where possible,
    and avoids over-allocating to val/test on tiny datasets.
    """
    rng = random.Random(seed)
    items = items[:]
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return [], [], []

    # Initial counts via rounding
    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    # Clamp to bounds and ensure at least 1 train sample if possible
    n_val = max(0, min(n, n_val))
    n_test = max(0, min(n - n_val, n_test))
    n_train = n - n_val - n_test
    if n_train <= 0:
        # Reduce val/test to free at least one for train
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
    ):
        self.items = items
        self.label_to_id = label_to_id
        self.audio_cache = audio_cache
        self.video_cache = video_cache
        self.audio_model = audio_model
        self.video_model = video_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames

    def __len__(self):
        return len(self.items)

    def _cache_path(self, cache_root: str, label: str, src_path: str) -> str:
        from pathlib import Path
        rel = Path(src_path).stem + ".pt"
        tgt = Path(cache_root) / label
        tgt.mkdir(parents=True, exist_ok=True)
        return str(tgt / rel)

    def __getitem__(self, idx):
        label, a_path, v_path = self.items[idx]

        if self.audio_cache:
            a_cache = self._cache_path(self.audio_cache, label, a_path)
            if os.path.exists(a_cache):
                a_emb = torch.load(a_cache)
            else:
                a_emb = extract_audio_embedding(a_path, model_name=self.audio_model, device=self.device)
                torch.save(a_emb, a_cache)
        else:
            a_emb = extract_audio_embedding(a_path, model_name=self.audio_model, device=self.device)

        if self.video_cache:
            v_cache = self._cache_path(self.video_cache, label, v_path)
            if os.path.exists(v_cache):
                v_emb = torch.load(v_cache)
            else:
                v_emb = extract_video_embedding(v_path, model_name=self.video_model, device=self.device, num_frames=self.num_frames)
                torch.save(v_emb, v_cache)
        else:
            v_emb = extract_video_embedding(v_path, model_name=self.video_model, device=self.device, num_frames=self.num_frames)

        x = torch.cat([a_emb, v_emb], dim=0).float()
        y = torch.tensor(self.label_to_id[label], dtype=torch.long)
        return x, y


def create_dataloaders(
    dataset_root: str,
    batch_size: int = 32,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    audio_model: str = "facebook/wav2vec2-base",
    video_model: str = "facebook/vjepa2-vitl-fpc64-256",
    num_frames: int = 32,
    audio_cache: Optional[str] = None,
    video_cache: Optional[str] = None,
):
    """
    Discover modality subfolders under dataset_root, pair files, split into
    train/val/test, and return DataLoaders plus metadata.

    Returns: (train_loader, val_loader, test_loader, meta)
    where meta = {label_to_id, id_to_label, in_dim}
    """
    audio_root, video_root = discover_modalities(dataset_root)
    pairs = find_pairs_from_roots(audio_root, video_root)
    if not pairs:
        raise RuntimeError("No matched audio/video pairs found under dataset root.")

    labels = sorted({lbl for lbl, _, _ in pairs})
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}

    # Default caches inside dataset_root/.cache
    if audio_cache is None:
        audio_cache = os.path.join(dataset_root, ".cache", "audio_emb")
    if video_cache is None:
        video_cache = os.path.join(dataset_root, ".cache", "video_emb")

    os.makedirs(audio_cache, exist_ok=True)
    os.makedirs(video_cache, exist_ok=True)

    train_items, val_items, test_items = split_train_val_test(
        pairs, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = EmbeddingPairDataset(
        train_items, label_to_id,
        audio_cache=audio_cache, video_cache=video_cache,
        audio_model=audio_model, video_model=video_model,
        device=device, num_frames=num_frames,
    )
    val_ds = EmbeddingPairDataset(
        val_items, label_to_id,
        audio_cache=audio_cache, video_cache=video_cache,
        audio_model=audio_model, video_model=video_model,
        device=device, num_frames=num_frames,
    )
    test_ds = EmbeddingPairDataset(
        test_items, label_to_id,
        audio_cache=audio_cache, video_cache=video_cache,
        audio_model=audio_model, video_model=video_model,
        device=device, num_frames=num_frames,
    )

    # Probe one sample to get input dim
    x0, _ = train_ds[0]
    in_dim = x0.numel()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    meta = {
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "in_dim": in_dim,
        "audio_root": audio_root,
        "video_root": video_root,
        "num_labels": len(labels),
    }
    return train_loader, val_loader, test_loader, meta
