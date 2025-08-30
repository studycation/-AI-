import os
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import torch

app = FastAPI()

# static 폴더 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

# 모델 로드
model = mobilenet_v2(pretrained=False)
num_classes = 5
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load("mobilenetv2_cat_emotion.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

class_names = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

# / 접속 시 index.html 제공
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return {"emotion": class_names[predicted.item()]}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
