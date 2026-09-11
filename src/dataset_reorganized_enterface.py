import os
import shutil
import argparse
from tqdm import tqdm

def reorganize_dataset(source_folder):
    source_folder = os.path.abspath(source_folder)
    destination_folder = source_folder + "_Reorganized"

    os.makedirs(destination_folder, exist_ok=True)

    for subject in os.listdir(source_folder):
        subject_path = os.path.join(source_folder, subject)

        if not os.path.isdir(subject_path):
            continue

        # Anger, Disgust, Happy...
        for emotion in tqdm(os.listdir(subject_path), desc=f"Processing {subject}"):
            emotion_path = os.path.join(subject_path, emotion)

            if not os.path.isdir(emotion_path):
                continue

            dest_emotion = os.path.join(destination_folder, emotion)
            os.makedirs(dest_emotion, exist_ok=True)

            # Session folders (or any nested folders)
            for subfolder in os.listdir(emotion_path):
                subfolder_path = os.path.join(emotion_path, subfolder)

                if not os.path.isdir(subfolder_path):
                    continue

                # Walk recursively inside the session folder
                for root, dirs, files in os.walk(subfolder_path):
                    for file in files:
                        src_file = os.path.join(root, file)

                        filename, ext = os.path.splitext(file)

                        # Prevent duplicate names
                        new_name = f"{subject}_{subfolder}_{filename}{ext}"

                        shutil.copy2(
                            src_file,
                            os.path.join(dest_emotion, new_name)
                        )

    print(f"\nDone!\nNew dataset created at:\n{destination_folder}")


if __name__ == "__main__":
    folder = input("Enter dataset folder path: ").strip().strip('"')
    reorganize_dataset(folder)