import pathlib, onnx
p = pathlib.Path(r"D:\Smart_Fog_Vision\models\fog_classifier.onnx")
model = onnx.load(str(p), load_external_data=True)
print("Loaded, checking external")
# Count initializers with external data
cnt_ext = sum(1 for init in model.graph.initializer if init.data_location == onnx.TensorProto.EXTERNAL)
print("External inits before:", cnt_ext)
print("Graph inputs:", [i.name for i in model.graph.input][:3])
# Save without external threshold
onnx.save(model, str(p), save_as_external_data=False)
# Also need to remove .data file? Check sizes
import os
data_path = p.with_suffix(".onnx.data")
if data_path.exists():
    print("data file still exists size", data_path.stat().st_size)
    # after save without external, the .onnx should be ~9MB single
    print("onnx size now", p.stat().st_size)
    # optionally remove data file if onnx is now standalone?
    # keep but will be ignored
    # If still external, try to embed
    model2 = onnx.load(str(p), load_external_data=False)
    print("Reload check")
else:
    print("no data file")
    print("onnx size", p.stat().st_size)

# Verify
import onnx
m2 = onnx.load(str(p))
print("Checker", onnx.checker.check_model(m2))
print("OK", p.stat().st_size/1024/1024, "MB")
