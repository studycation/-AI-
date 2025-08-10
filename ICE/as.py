import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    classes = os.listdir(source_dir)
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        split_idx = int(len(images) * split_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        for img in train_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(train_dir, class_name)
            os.makedirs(dst_path, exist_ok=True)
            shutil.copy(src_path, dst_path)

        for img in val_images:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(val_dir, class_name)
            os.makedirs(dst_path, exist_ok=True)
            shutil.copy(src_path, dst_path)

# 사용 예:
split_dataset(
    source_dir = 'C:/ICE/train_augmented',
    train_dir='C:/ICE/train',
    val_dir='C:/ICE/val'
)
