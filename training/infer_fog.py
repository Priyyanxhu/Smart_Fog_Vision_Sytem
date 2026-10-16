"""Quick inference on a fog image using the trained MobileNetV2 model."""
import sys, pathlib, torch, torchvision.transforms as T
from PIL import Image
import torch.nn as nn
import torchvision.models as models

MODEL = pathlib.Path(r"D:\Smart_Fog_Vision\models\fog_classifier.pt")
LABELS = ["non-fog","fog"]
SIZE=224
MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]
tf = T.Compose([T.Resize(256), T.CenterCrop(SIZE), T.ToTensor(), T.Normalize(MEAN,STD)])

def load_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280,2))
    state = torch.load(str(MODEL), map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    return m

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: py infer_fog.py <image_path>")
        sys.exit(1)
    path = sys.argv[1]
    m = load_model()
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = m(x)
        probs = torch.softmax(logits,1).numpy()[0]
    pred = int(probs.argmax())
    print(f"Image: {path}")
    print(f"Pred: {LABELS[pred]} ({probs[pred]*100:.2f}%)")
    print(f"Probs: non-fog {probs[0]*100:.2f}% | fog {probs[1]*100:.2f}%")
    print(f"Logits: {logits.numpy()[0]}")
