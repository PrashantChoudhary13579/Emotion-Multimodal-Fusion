import os 
import shutil
import argparse
from tqdm import tqdm


emotion_map = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


def reorganize_dataset(source_folder):
    src_folder = os.path.abspath(source_folder)
    dest_folder = src_folder + "_Reorganized"
    os.makedirs(dest_folder, exist_ok=True)

    for video_folder in tqdm(os.listdir(src_folder), desc="Processing video folders"):
        video_path = os.path.join(src_folder, video_folder)
        if not os.path.isdir(video_path):
            continue
        for actor in os.listdir(video_path):
            actor_path = os.path.join(video_path , actor )
            if not os.path.isdir(actor_path):
                continue
            for file in os.listdir(actor_path):
                src_file = os.path.join(actor_path, file)
                if not os.path.isfile(src_file):
                    continue
                filename, ext = os.path.splitext(file)
                parts = filename.split('-')
                if len(parts) < 3:
                    continue

                emotion = parts[2]
                emotion = emotion_map.get(emotion, "Unknown")
                
                destination_dir = os.path.join(dest_folder, emotion)
                os.makedirs(destination_dir, exist_ok=True)
                dest_file = os.path.join(destination_dir, file)
                shutil.copy2(src_file, dest_file)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorganize dataset into emotion-based folders.")
    parser.add_argument("source_folder", help="Path to the source dataset folder.")
    args = parser.parse_args()

    reorganize_dataset(args.source_folder)