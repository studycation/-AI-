#Step 1: MobileNetV2 학습용 Python 코드 만들기

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 경로 설정
train_dir = 'C:/ICE/train'
val_dir = 'C:/ICE/val'

# 전처리: 이미지 크기 맞추고, 텐서로 바꾸고, 정규화
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 데이터셋 로드
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset = ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 클래스 수
num_classes = len(train_dataset.classes)

# 모델 정의
model = mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(5):  # epoch 수 조절 가능
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1} Loss: {running_loss/len(train_loader):.4f}')

# 저장
torch.save(model.state_dict(), 'mobilenetv2_trained.pth')
print("✅ 모델 저장 완료: mobilenetv2_trained.pth")
