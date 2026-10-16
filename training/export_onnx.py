import torch, torchvision.models as models, pathlib, torch.nn as nn
OUT = pathlib.Path(r"D:\Smart_Fog_Vision\models")
DEVICE = torch.device("cpu")
CLASS_NAMES=["non-fog","fog"]
IMG_SIZE=224
# rebuild model
try:
    mnet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
except:
    mnet = models.mobilenet_v2(pretrained=True)
# fix classifier correctly: original classifier is (dropout, linear). Replace entire classifier
in_features = mnet.classifier[1].in_features
# Simple: new classifier = Sequential(Dropout, Linear)
mnet.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features, 2)
)
# load best weights
ckpt = OUT / "best_mobilenetv2_fog.pt"
state = torch.load(str(ckpt), map_location=DEVICE)
# Handle double-nested saved earlier? Our previous save had Sequential inside, now we changed architecture; need to adapt.
# Previous state had classifier.1.1.weight, now we have classifier.1.weight.
# Map keys
new_state = {}
for k,v in state.items():
    # if key contains classifier.1.1, rename to classifier.1
    if "classifier.1.1." in k:
        new_k = k.replace("classifier.1.1.", "classifier.1.")
        new_state[new_k] = v
    elif "classifier.1.0" in k:
        # dropout weight none, skip
        continue
    else:
        new_state[k]=v
# Check if loading will fail, try direct load with strict=False
try:
    mnet.load_state_dict(new_state, strict=False)
    print("Loaded adapted state dict")
except Exception as e:
    print("Adapt load failed", e)
    mnet.load_state_dict(state, strict=False)
    print("Loaded original state dict strictly=False")

mnet.eval()
# Fix any double dropout by ensuring correct
print(mnet.classifier)

dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE, device=DEVICE)
onnx_path = OUT / "fog_classifier.onnx"
torch.onnx.export(
    mnet,
    dummy,
    str(onnx_path),
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input":{0:"batch"}, "logits":{0:"batch"}},
    opset_version=17,
    do_constant_folding=True
)
print(f"ONNX saved {onnx_path} {onnx_path.stat().st_size/1024:.1f} KB")
try:
    import onnx
    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    print("ONNX checker passed")
    print("Inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
    print("Outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
except Exception as e:
    print("onnx check failed", e)

# Also save a clean pt with new architecture
torch.save(mnet.state_dict(), OUT / "fog_classifier.pt")
print("Saved clean pt to fog_classifier.pt")
