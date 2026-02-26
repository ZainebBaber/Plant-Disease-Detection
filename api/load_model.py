import json
import os
import torch
from torchvision import transforms
from PIL import Image
import io
from pathlib import Path
from model_architecture.PlantCnn_model import PlantCNN 

BASE_DIR= Path(__file__).resolve().parent.parent
MODEL_PATH=BASE_DIR /"saved"/"models"/"best_model.pth"
MAPPING_PATH=os.path.join(BASE_DIR, "saved/models/class_mapping.json")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(MAPPING_PATH) as f:
    class_to_idx=json.load(f)

# Reverse mapping: idx -> class
idx_to_class = {v: k for k, v in class_to_idx.items()}
num_classes=len(idx_to_class)

model=PlantCNN(num_classes=num_classes)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)
model.to(device)
print("Loaded model from:", MODEL_PATH)
print("File exists:", MODEL_PATH.exists())


model.eval()


transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], 
                             std=[0.229,0.224,0.225])
    ])



def predict_image(image_bytes: bytes):
    img=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img=transform(img)
    img=img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs=model(img)
        print(outputs)
        probs=torch.softmax(outputs, dim=1)
        print("Logits:", outputs)
        print("Probs:", probs)
        
        _,predicted=torch.max(outputs,1)

    predicted_class=idx_to_class[predicted.item()]
    confidence = probs[0, predicted.item()].item()

    return{
        "predicted_class":predicted_class,
        "confidence":round(confidence,4)
                }