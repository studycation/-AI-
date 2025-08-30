from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image

app = FastAPI()

# CORS 설정 (테스트용, 모든 도메인 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 로드
model = mobilenet_v2(pretrained=False)
num_classes = 5
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("mobilenetv2_cat_emotion.pth"))
model.eval()

# 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    return {"prediction": class_names[predicted.item()]}
