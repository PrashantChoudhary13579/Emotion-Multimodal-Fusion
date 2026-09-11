import argparse
import logging
import os
import subprocess
from typing import Tuple


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")


def has_audio(input_path: str) -> bool:
    """Check if a video has an audio stream using ffprobe."""
    command = [
        "ffprobe",
        "-i",
        input_path,
        "-show_streams",
        "-select_streams",
        "a",
        "-loglevel",
        "error",
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return bool(result.stdout.strip())


def video_to_wav_ffmpeg(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """Convert a single video file to WAV format using ffmpeg."""
    if not has_audio(input_path):
        logging.warning("Skipping %s (no audio track found)", input_path)
        return False

    command = [
        "ffmpeg",
        "-i",
        input_path,
        "-vn",  # no video
        "-acodec",
        "pcm_s16le",  # WAV format
        "-ar",
        str(sample_rate),  # sample rate
        "-ac",
        "1",  # mono channel
        output_path,
        "-y",  # overwrite existing output
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info("Converted %s -> %s", input_path, output_path)
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Error converting %s: %s", input_path, e)
        return False


def convert_videos_in_folders(
    root_dir: str,
    output_root: str = "Audio",
    sample_rate: int = 16000,
    recursive: bool = True,
) -> Tuple[int, int]:
    """
    Convert all videos under each emotion folder to WAV files.

    Returns:
        Tuple[int, int]: (converted_count, skipped_count)
    """
    converted, skipped = 0, 0
    os.makedirs(output_root, exist_ok=True)

    for emotion_folder in sorted(os.listdir(root_dir)): # taking happy...
        emotion_path = os.path.join(root_dir, emotion_folder) # /home/staru/.../video/happy
        if not os.path.isdir(emotion_path):
            continue

        output_emotion_path = os.path.join(output_root, emotion_folder) # /home/staru/.../Audio/happy
        os.makedirs(output_emotion_path, exist_ok=True) #

        walker = os.walk(emotion_path) if recursive else [(emotion_path, [], os.listdir(emotion_path))] # taking all files in the emotion folder
        for current_dir, _, files in walker:
            rel_subdir = os.path.relpath(current_dir, emotion_path)
            rel_subdir = "" if rel_subdir == "." else rel_subdir
            out_dir = os.path.join(output_emotion_path, rel_subdir)  
            os.makedirs(out_dir, exist_ok=True)

            for file in files:
                if not file.lower().endswith(VIDEO_EXTS):
                    continue
                input_file = os.path.join(current_dir, file) # /home/staru/.../video/happy/video1.mp4
                output_file = os.path.join(out_dir, os.path.splitext(file)[0] + ".wav") # /home/staru/.../Audio/happy/video1.wav
                logging.debug("Processing %s", input_file)
                if video_to_wav_ffmpeg(input_file, output_file, sample_rate):
                    converted += 1
                else:
                    skipped += 1
    return converted, skipped


def parse_args():
    parser = argparse.ArgumentParser(description="Extract audio tracks from a multimodal dataset.")
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Root directory containing emotion folders with video files.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Destination directory where extracted WAV files will be stored.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="Target audio sample rate for the extracted WAV files.",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Disable recursion; only convert videos located directly in each emotion folder.",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s - %(message)s")
    logging.info(
        "Starting audio extraction | dataset=%s | output=%s | sample_rate=%d",
        args.dataset_path,
        args.output_path,
        args.sample_rate,
    )

    converted, skipped = convert_videos_in_folders(
        root_dir=args.dataset_path,
        output_root=args.output_path,
        sample_rate=args.sample_rate,
        recursive=not args.no_recursive,
    )
    logging.info("Audio extraction complete | converted=%d | skipped=%d", converted, skipped)


if __name__ == "__main__":
    main()
