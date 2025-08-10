import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

# 1. 학습된 모델 불러오기
model = mobilenet_v2(pretrained=False)
num_classes = 5  # 감정 개수 (너가 분류한 클래스 수)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('mobilenetv2_cat_emotion.pth'))
model.eval()

# 2. 이미지 전처리 함수 (학습 때와 똑같이!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. 예측할 이미지 경로
img_path =r'C:\Users\USER\Desktop\cat5.jpg'  # 테스트할 고양이 사진 경로

# 4. 이미지 불러오고 전처리
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

# 5. 예측
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

# 6. 클래스 이름 매핑 (train_dataset.classes를 저장해두면 좋음)
class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

print(f"이 고양이 감정은: {class_names[predicted.item()]} 입니다!")
