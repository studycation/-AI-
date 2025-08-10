import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 데이터 경로 (네가 이미 만들어둔 train/val 폴더)
train_dir = 'C:/ICE/train'
val_dir = 'C:/ICE/val'

# 2. 전처리 (학습용, 검증용은 심플하게)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. 데이터셋 로드
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. MobileNetV2 모델 불러오기 (사전학습된 모델 사용)
model = mobilenet_v2(pretrained=True)

# 5. 마지막 레이어 수정 (우리 데이터 클래스 수에 맞게)
num_classes = len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# 6. 장비 설정 (GPU 있으면 GPU 사용)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 7. 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 8. 학습 함수 정의
def train(epoch):
    model.train()  # 학습 모드로 변경
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()         # 이전 경사 초기화
        outputs = model(images)       # 모델에 입력 넣고 출력 얻기
        loss = criterion(outputs, labels)  # 손실 계산
        loss.backward()               # 역전파 (기울기 계산)
        optimizer.step()              # 가중치 업데이트

        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# 9. 검증 함수 정의
def validate():
    model.eval()   # 평가 모드
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 가장 높은 점수 클래스 선택
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

# 10. 메인 루프 (5 epoch만 학습)
for epoch in range(10):
    train(epoch)
    validate()

# 11. 학습된 모델 저장
torch.save(model.state_dict(), 'mobilenetv2_cat_emotion.pth')
print("모델 저장 완료: mobilenetv2_cat_emotion.pth")
