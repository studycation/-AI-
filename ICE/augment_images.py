from PIL import Image
import os
import random
from torchvision import transforms

# 증강 변환기 (예: 좌우 반전, 회전, 밝기 조절)
augment = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),  # 반드시 좌우 반전
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

base_dir = "C:/ICE/train"  # 원본 train 폴더 경로
augmented_dir = "C:/ICE/train_augmented"  # 증강 이미지 저장할 폴더

os.makedirs(augmented_dir, exist_ok=True)

emotion_to_augment = ["Happy", "Sad", "Neutral", "Surprised"]

for emotion in emotion_to_augment:
    src_folder = os.path.join(base_dir, emotion)
    dst_folder = os.path.join(augmented_dir, emotion)
    os.makedirs(dst_folder, exist_ok=True)

    for img_name in os.listdir(src_folder):
        img_path = os.path.join(src_folder, img_name)
        try:
        	img = Image.open(img_path).convert("RGB")  
        except:
        	print(f"이미지 열기 실패: {img_name}")
       		continue

        # 원본 이미지도 복사
        img.save(os.path.join(dst_folder, img_name))

        # 증강 이미지 2개 생성 (필요하면 숫자 조절 가능)
        for i in range(2):
            augmented_img = augment(img)
            new_name = img_name.split('.')[0] + f"_aug{i}.jpg"
            augmented_img.convert("RGB").save(os.path.join(dst_folder, new_name))

print("✅ 증강 이미지 생성 완료")
