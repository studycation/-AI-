import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

# 1. 학습된 모델 부르기
model = mobilenet_v2(pretrained=False)
num_classes = 5  # 감정 종류 (이번 프로젝트에서는 우선 5개로 분류)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('mobilenetv2_cat_emotion.pth'))
model.eval()

# 2. 이미지 전처리 함수 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 3. 테스트할 이미지 경로
img_path =r'C:\Users\USER\Desktop\cat5.jpg'  

# 4. 전처리
image = Image.open(img_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0) 
# 5. 예측
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

# 6. 클래스 이름 매핑 
class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

print(f"이 고양이 감정은: {class_names[predicted.item()]} 입니다!")
