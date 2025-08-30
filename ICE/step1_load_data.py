#이미지전처리
# step1_load_data.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_dir = 'C:/ICE/train'
val_dir = 'C:/ICE/val'

# 이미지 전처리 
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # 이미지를 Tensor로 변환
    transforms.Normalize([0.485, 0.456, 0.406],  # 이미지 정규화 ImageNet 기준으로..
                         [0.229, 0.224, 0.225])
])

# 검증 데이터
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 데이터셋 객체 
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

# 배치 단위로 불러오기
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


