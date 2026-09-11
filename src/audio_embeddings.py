import os
import argparse
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
import librosa
from tqdm import tqdm
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    WhisperModel,
    WhisperProcessor,
)


_AUDIO_MODELS: Dict[Tuple[str, str, str], Tuple[torch.nn.Module, object]] = {}

_PRESET_MODELS = {
    "wav2vec2-base": ("wav2vec2", "facebook/wav2vec2-base"),
    "wav2vec2-large": ("wav2vec2", "facebook/wav2vec2-large-960h"),
    "whisper-tiny": ("whisper", "openai/whisper-tiny"),
    "whisper-base": ("whisper", "openai/whisper-base"),
    "whisper-small": ("whisper", "openai/whisper-small"),
    "whisper-medium": ("whisper", "openai/whisper-medium"),
    "whisper-large-v2": ("whisper", "openai/whisper-large-v2"),
}


def _resolve_model_spec(model_name: str) -> Tuple[str, str]:
    """Return (backend, huggingface_repo_id)."""
    name = model_name.strip()
    if not name:
        raise ValueError("Model name must be a non-empty string")

    preset = _PRESET_MODELS.get(name.lower())
    if preset:
        return preset

    repo_id = name
    lower = repo_id.lower()
    if "whisper" in lower:
        backend = "whisper"
    elif "wav2vec" in lower:
        backend = "wav2vec2"
    else:
        raise ValueError(
            f"Could not infer audio backend for model '{model_name}'. "
            "Specify a known preset (e.g., 'whisper-base') or a Hugging Face repo "
            "id containing 'wav2vec' or 'whisper'."
        )
    return backend, repo_id


def _ensure_audio_model(model_name: str = "wav2vec2-base", device: Optional[torch.device] = None):
    backend, repo_id = _resolve_model_spec(model_name)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_key = (backend, repo_id, device.type)
    cached = _AUDIO_MODELS.get(cache_key)
    if cached is not None:
        return backend, cached

    if backend == "wav2vec2":
        processor = Wav2Vec2Processor.from_pretrained(repo_id)
        model = Wav2Vec2Model.from_pretrained(repo_id)
        model = model.to(device)
    elif backend == "whisper":
        processor = WhisperProcessor.from_pretrained(repo_id)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        try:
            model = WhisperModel.from_pretrained(repo_id, dtype=dtype)
        except TypeError:
            model = WhisperModel.from_pretrained(repo_id, torch_dtype=dtype)
        model = model.to(device)
    else:
        raise RuntimeError(f"Unsupported audio backend '{backend}'")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    _AUDIO_MODELS[cache_key] = (model, processor)
    return backend, (model, processor)


def extract_audio_embedding(
    audio_path: str,
    model_name: str = "wav2vec2-base",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load audio, run the requested model, and return a pooled embedding [D].
    """
    backend, (model, processor) = _ensure_audio_model(model_name=model_name, device=device)

    speech, _ = librosa.load(audio_path, sr=16000)

    if backend == "wav2vec2":
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            hidden = model(**inputs).last_hidden_state  # [B, T, D]

        mask = inputs.get("attention_mask")
        if mask is None:
            emb = hidden.mean(dim=1)
        else:
            mask = mask.unsqueeze(-1)  # [B, T, 1]
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    elif backend == "whisper":
        proc_out = processor(speech, sampling_rate=16000, return_tensors="pt")
        target_dtype = next(model.encoder.parameters()).dtype
        input_features = proc_out.input_features.to(device=model.device, dtype=target_dtype)

        with torch.no_grad():
            hidden = model.encoder(input_features).last_hidden_state  # [B, T, D]

        emb = hidden.mean(dim=1)

    else:
        raise RuntimeError(f"Unsupported audio backend '{backend}'")

    return emb.squeeze(0).detach().cpu()


def _iter_audio_files(root: str) -> List[str]:
    exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if pathlib.Path(fn).suffix.lower() in exts:
                files.append(os.path.join(dirpath, fn))
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract mean-pooled audio embeddings with selectable models")
    parser.add_argument("--input_path", required=True, help="Single audio file or directory root containing audio files grouped by label.")
    parser.add_argument("--output_path", required=True, help="Directory where extracted embeddings (.pt) will be written.")
    parser.add_argument(
        "--model",
        default="wav2vec2-base",
        help=(
            "Audio model preset or Hugging Face repo id. "
            "Examples: wav2vec2-base, facebook/wav2vec2-base, whisper-small, openai/whisper-base."
        ),
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend, _ = _ensure_audio_model(args.model, device=device)
    print(f"Using {backend} model '{args.model}' on {device}.")

    in_path = pathlib.Path(args.input_path)
    out_root = pathlib.Path(args.output_path)
    out_root.mkdir(parents=True, exist_ok=True)

    if in_path.is_file():
        emb = extract_audio_embedding(str(in_path), model_name=args.model, device=device)
        out_path = out_root / (in_path.stem + ".pt")
        torch.save(emb, out_path)
        print(f"Saved audio embedding to {out_path} with shape {tuple(emb.shape)}")
    else:
        files = _iter_audio_files(str(in_path))
        for fp in tqdm(files, desc="Audio embeddings", dynamic_ncols=True):
            rel = pathlib.Path(fp).relative_to(in_path)
            target_dir = out_root / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / (pathlib.Path(fp).stem + ".pt")
            try:
                emb = extract_audio_embedding(fp, model_name=args.model, device=device)
                torch.save(emb, out_path)
                print(f"Saved: {out_path}")
            except Exception as e:
                print(f"Failed {fp}: {e}")


if __name__ == "__main__":
    main()
