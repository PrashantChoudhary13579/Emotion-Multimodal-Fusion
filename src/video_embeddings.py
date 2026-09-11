import os
import argparse
import pathlib
import torch
import numpy as np
from typing import Optional, List
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor
from torchvision.io import read_video
from torchvision.transforms.functional import to_pil_image


_VIDEO_MODEL = None
_VIDEO_PROCESSOR = None


def _ensure_video_model(model_name: str = "facebook/vjepa2-vitl-fpc64-256", device: Optional[torch.device] = None):
    global _VIDEO_MODEL, _VIDEO_PROCESSOR
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _VIDEO_MODEL is None or _VIDEO_PROCESSOR is None:
        _VIDEO_PROCESSOR = AutoVideoProcessor.from_pretrained(model_name)
        # Use float16 on CUDA where possible; fallback to float32 on CPU
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        _VIDEO_MODEL = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        _VIDEO_MODEL.eval()
    return _VIDEO_MODEL, _VIDEO_PROCESSOR


def _frame_indices(total_frames: int, num_frames: int = 32) -> np.ndarray:
    if total_frames <= 0:
        return np.arange(num_frames, dtype=int)
    return np.linspace(0, max(0, total_frames - 1), num=num_frames, dtype=int)


def _load_video_frames(video_path: str, num_frames: int) -> List[np.ndarray]:
    video, _, _ = read_video(video_path, pts_unit="sec")
    if video.numel() == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    total = video.shape[0]
    idx = _frame_indices(total, num_frames=num_frames).astype(int)
    frames = []
    for i in idx:
        clamped = int(max(0, min(total - 1, i)))
        frame = video[clamped]
        if frame.ndim == 3 and frame.shape[-1] in (1, 3, 4):
            frame = frame.permute(2, 0, 1)
        elif frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            pass  # already CHW
        else:
            frame = frame.permute(2, 0, 1)
        frames.append(np.array(to_pil_image(frame)))
    return frames


def extract_video_embedding(video_path: str, model_name: str = "facebook/vjepa2-vitl-fpc64-256", device: Optional[torch.device] = None, num_frames: int = 32) -> torch.Tensor:
    """
    Decode frames, run V-JEPA 2, mean-pool tokens to a fixed embedding [D].
    """
    model, processor = _ensure_video_model(model_name=model_name, device=device)

    frames = _load_video_frames(video_path, num_frames=num_frames)
    inputs = processor(frames, return_tensors="pt").to(model.device)
    with torch.no_grad():
        tokens = model(**inputs).last_hidden_state  # [B, N_tokens, D]
    emb = tokens.mean(dim=1).squeeze(0).detach().cpu()  # [D]
    return emb


def _iter_video_files(root: str) -> List[str]:
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if pathlib.Path(fn).suffix.lower() in exts:
                files.append(os.path.join(dirpath, fn))
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract mean-pooled V-JEPA2 video embeddings")
    parser.add_argument(
        "--input_path",
        required=True,
        help="Single video file or directory root containing video files grouped by label.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Directory where extracted video embeddings (.pt) will be written.",
    )
    parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256", help="V-JEPA2 model name")
    parser.add_argument("--num_frames", type=int, default=32, help="Frames sampled per clip")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ensure_video_model(args.model, device=device)

    in_path = pathlib.Path(args.input_path)
    out_root = pathlib.Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        emb = extract_video_embedding(str(in_path), model_name=args.model, device=device, num_frames=args.num_frames)
        out_path = out_root / (in_path.stem + ".pt")
        torch.save(emb, out_path)
        print(f"Saved video embedding to {out_path} with shape {tuple(emb.shape)}")
    else:
        files = _iter_video_files(str(in_path))
        for fp in tqdm(files, desc="Video embeddings", dynamic_ncols=True):
            rel = pathlib.Path(fp).relative_to(in_path)
            target_dir = out_root / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / (pathlib.Path(fp).stem + ".pt")
            try:
                emb = extract_video_embedding(fp, model_name=args.model, device=device, num_frames=args.num_frames)
                torch.save(emb, out_path)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed {fp}: {e}")


if __name__ == "__main__":
    main()
