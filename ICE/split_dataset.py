import os
import random
import shutil

# 원본 데이터 경로
base_dir = "C:\\ICE"
emotion_labels = ["Angry", "Happy", "Sad", "Neutral", "Surprised"]



# 비율
val_ratio = 0.2  # 20%를 validation으로

for emotion in emotion_labels:
    emotion_dir = os.path.join(base_dir, emotion)
    all_files = os.listdir(emotion_dir)
    random.shuffle(all_files)

    val_count = int(len(all_files) * val_ratio)
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    train_target_dir = os.path.join(base_dir, "train", emotion)
    val_target_dir = os.path.join(base_dir, "val", emotion)
    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(emotion_dir, file), os.path.join(train_target_dir, file))

    for file in val_files:
        shutil.copy(os.path.join(emotion_dir, file), os.path.join(val_target_dir, file))

    print(f"{emotion}: {len(train_files)} train, {len(val_files)} val")
