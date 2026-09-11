import os
import subprocess

def mp3_to_wav_ffmpeg(input_path, output_path=None, sample_rate=16000):
    """
    Convert MP3 to WAV using ffmpeg.
    """
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + ".wav"
    
    command = [
        "ffmpeg", "-i", input_path,
        "-acodec", "pcm_s16le",    # PCM 16-bit encoding
        "-ar", str(sample_rate),   # resample
        "-ac", "1",                # mono
        output_path, "-y"
    ]
    subprocess.run(command, check=True)
    #print(f"Converted {input_path} -> {output_path}")


def convert_mp3_folder(input_folder, output_folder="wav_data", sample_rate=16000):
    """
    Convert all MP3 files in a folder to WAV and save them in output_folder.
    Keeps the same filenames.
    """
    os.makedirs(output_folder, exist_ok=True)

    for root, _, files in os.walk(input_folder):
        # Recreate subfolder structure in output
        rel_path = os.path.relpath(root, input_folder)
        target_root = os.path.join(output_folder, rel_path)
        os.makedirs(target_root, exist_ok=True)

        for file in files:
            if file.lower().endswith(".mp3"):
                input_path = os.path.join(root, file)
                output_path = os.path.join(target_root, os.path.splitext(file)[0] + ".wav")
                mp3_to_wav_ffmpeg(input_path, output_path, sample_rate)


if __name__ == "__main__":
    input_dir = "./Audio Data"      
    output_dir = "./wav_data"     
    convert_mp3_folder(input_dir, output_dir, sample_rate=16000)
