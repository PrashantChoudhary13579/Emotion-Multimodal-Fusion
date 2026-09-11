import os
import random
import shutil


datasets = [
    # "/path/to/PUMAVE-D",
    "/path/to/eNTERFACE_Reorganized",
    "/path/to/RAVDESS_Reorganized",
    "/path/to/SAMVEDNA_Complete_Dataset",
]


output_folder = "/path/to/Mixed_Dataset"


# Number of files from each emotion in each dataset
samples_per_emotion = 200


allowed_emotions = {
    "anger",
    "disgust",
    "happy",
    "fear",
    "sad"
}


video_extensions = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".mpeg",
    ".webm"
}



def collect_emotion_videos(dataset):

    emotion_files = {
        emotion: []
        for emotion in allowed_emotions
    }


    for root, dirs, files in os.walk(dataset):

        folder = os.path.basename(root).lower()


        if folder not in allowed_emotions:
            continue


        for file in files:

            ext = os.path.splitext(file)[1].lower()

            if ext in video_extensions:

                emotion_files[folder].append(
                    os.path.join(root, file)
                )


    return emotion_files




def main():

    os.makedirs(
        output_folder,
        exist_ok=True
    )


    for dataset in datasets:


        dataset_name = os.path.basename(dataset)

        print("\nProcessing:", dataset_name)


        emotion_data = collect_emotion_videos(dataset)



        for emotion, files in emotion_data.items():


            print(
                emotion,
                "available:",
                len(files)
            )


            if len(files) == 0:
                continue



            # RANDOM SELECTION WITHOUT REPLACEMENT
            if len(files) >= samples_per_emotion:

                selected_files = random.sample(
                    files,
                    samples_per_emotion
                )

            else:

                selected_files = files



            emotion_output = os.path.join(
                output_folder,
                emotion.capitalize()
            )


            os.makedirs(
                emotion_output,
                exist_ok=True
            )



            for video in selected_files:


                filename = os.path.basename(video)


                # Add dataset name to avoid duplicate filenames
                new_filename = (
                    dataset_name +
                    "_" +
                    filename
                )


                destination = os.path.join(
                    emotion_output,
                    new_filename
                )


                # Extra safety: never overwrite
                if not os.path.exists(destination):

                    shutil.copy2(
                        video,
                        destination
                    )



            print(
                "Copied:",
                len(selected_files),
                emotion,
                "videos"
            )



    print("\nFinished creating mixed dataset")



if __name__ == "__main__":
    main()