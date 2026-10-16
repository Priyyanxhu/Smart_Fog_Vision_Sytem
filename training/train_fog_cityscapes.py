r"""Fog classifier retrain on FoggyCityscapes (driving viewpoint).

fog     = Dense_Fog (500) + Medium_Fog (500)  -> label 1
non-fog = No_Fog (500)                        -> label 0
Backbone: MobileNetV2 ImageNet, same head/augs as train_fog_classifier.py.
Outputs: models/fog_classifier.{pt,onnx}, best/last, metrics_*, model_card.
Old FoggyRoad artifacts are backed up to models/foggyroad_backup/.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import argparse, random, json, time, shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np

BASE = Path(r"D:\FoggyCityscapes\Foggy_Cityscapes")
OUT = Path(r"D:\Smart_Fog_Vision\models")
CLASS_NAMES = ["non-fog", "fog"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dense = sorted((BASE / "Dense_Fog").glob("*.png"))
    med = sorted((BASE / "Medium_Fog").glob("*.png"))
    clear = sorted((BASE / "No_Fog").glob("*.png"))
    print(f"Dense={len(dense)} Medium={len(med)} NoFog={len(clear)}")
    assert dense and med and clear, "FoggyCityscapes folders missing"

    fog_all = [(str(p), 1) for p in dense + med]
    clear_all = [(str(p), 0) for p in clear]
    rng = random.Random(args.seed)
    rng.shuffle(fog_all); rng.shuffle(clear_all)

    def split(lst, ratios=(0.8, 0.1, 0.1)):
        n = len(lst); ntr = int(n * ratios[0]); nv = int(n * ratios[1])
        return lst[:ntr], lst[ntr:ntr + nv], lst[ntr + nv:]

    f_tr, f_va, f_te = split(fog_all)
    c_tr, c_va, c_te = split(clear_all)
    splits = {"train": f_tr + c_tr, "val": f_va + c_va, "test": f_te + c_te}
    for k in splits: rng.shuffle(splits[k])
    for k, v in splits.items():
        cnt = Counter(l for _, l in v)
        print(f"  {k.upper():5s}: {len(v):4d} (non-fog={cnt[0]}, fog={cnt[1]})")

    SIZE = 224
    MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
    train_tf = T.Compose([T.Resize(256), T.RandomResizedCrop(SIZE, scale=(0.8, 1.0)),
                          T.RandomHorizontalFlip(), T.RandomRotation(10),
                          T.ColorJitter(0.1, 0.1, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.Resize(256), T.CenterCrop(SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])

    class DS(Dataset):
        def __init__(self, items, tf):
            self.items = items; self.tf = tf
        def __len__(self): return len(self.items)
        def __getitem__(self, i):
            p, l = self.items[i]
            return self.tf(Image.open(p).convert("RGB")), l, p

    tl = DataLoader(DS(splits["train"], train_tf), batch_size=args.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(DS(splits["val"], eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)
    sl = DataLoader(DS(splits["test"], eval_tf), batch_size=args.batch_size, shuffle=False, num_workers=0)

    try:
        mnet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        mnet = models.mobilenet_v2(pretrained=True)
    mnet.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(mnet.classifier[1].in_features, 2))
    for par in mnet.features[:14].parameters():
        par.requires_grad = False
    mnet = mnet.to(device)
    print("Trainable:", sum(p.numel() for p in mnet.parameters() if p.requires_grad))

    crit = nn.CrossEntropyLoss(label_smoothing=0.05)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, mnet.parameters()), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    def evaluate(loader):
        mnet.eval(); tl_, ta, t_ = 0, 0, 0; pr, lb = [], []
        with torch.no_grad():
            for im, y, _ in loader:
                im, y = im.to(device), y.to(device)
                lg = mnet(im); tl_ += crit(lg, y).item() * im.size(0)
                ta += (lg.argmax(1) == y).sum().item(); t_ += im.size(0)
                pr.extend(lg.argmax(1).cpu().tolist()); lb.extend(y.cpu().tolist())
        return tl_ / t_, ta / t_, pr, lb

    # backup FoggyRoad artifacts
    bk = OUT / "foggyroad_backup"
    bk.mkdir(exist_ok=True)
    for f in ["fog_classifier.pt", "fog_classifier.onnx", "best_mobilenetv2_fog.pt",
              "metrics_val.json", "metrics_test.json", "model_card.json"]:
        s = OUT / f
        if s.exists() and not (bk / f).exists():
            shutil.copy2(s, bk / f)
    print("Backed up FoggyRoad artifacts to", bk)

    best_f1, best_acc, best_state, nop, hist = 0, 0, None, 0, {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        mnet.train(); rl, ra, sn = 0, 0, 0
        for i, (im, y, _) in enumerate(tl):
            im, y = im.to(device), y.to(device)
            opt.zero_grad(); lg = mnet(im); loss = crit(lg, y)
            loss.backward(); nn.utils.clip_grad_norm_(mnet.parameters(), 1.0); opt.step()
            rl += loss.item() * im.size(0); ra += (lg.argmax(1) == y).sum().item(); sn += im.size(0)
            if (i + 1) % 10 == 0 or (i + 1) == len(tl):
                print(f"  Ep{ep:02d} batch {i+1}/{len(tl)} loss {loss.item():.4f}", flush=True)
        vl_loss, vl_acc, vp, vt = evaluate(vl)
        tp = sum(1 for p, t in zip(vp, vt) if p == t == 1); fp = sum(1 for p, t in zip(vp, vt) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(vp, vt) if p == 0 and t == 1)
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-6, prec + rec)
        sch.step()
        hist["train_loss"].append(rl / sn); hist["train_acc"].append(ra / sn)
        hist["val_loss"].append(vl_loss); hist["val_acc"].append(vl_acc); hist["val_f1"].append(f1)
        print(f"Ep{ep:02d} train {rl/sn:.4f}/{ra/sn:.3f} val {vl_loss:.4f}/{vl_acc:.3f}/F1 {f1:.3f}", flush=True)
        if f1 > best_f1 or (f1 == best_f1 and vl_acc >= best_acc):
            best_f1, best_acc = f1, vl_acc
            best_state = {k: v.cpu() for k, v in mnet.state_dict().items()}
            torch.save(mnet.state_dict(), OUT / "best_mobilenetv2_fog.pt"); nop = 0
        else:
            nop += 1
            if nop >= args.patience:
                print("Early stop"); break
        torch.save(mnet.state_dict(), OUT / "last_mobilenetv2_fog.pt")
    print(f"Done {(time.time()-t0)/60:.1f}min best acc {best_acc:.3f} F1 {best_f1:.3f}")
    if best_state is not None:
        mnet.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(mnet.state_dict(), OUT / "fog_classifier.pt")

    for name, loader in [("val", vl), ("test", sl)]:
        loss, acc, pr, lb = evaluate(loader)
        cm = [[0, 0], [0, 0]]
        for p, t in zip(pr, lb): cm[t][p] += 1
        print(f"FINAL {name.upper()} loss {loss:.4f} acc {acc*100:.2f}% TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
        with open(OUT / f"metrics_{name}.json", "w") as f:
            json.dump({"split": name, "loss": loss, "accuracy": acc, "confusion_matrix": cm,
                       "class_names": CLASS_NAMES}, f, indent=2)
    with open(OUT / "history.json", "w") as f: json.dump(hist, f, indent=2)
    with open(OUT / "labels.json", "w") as f:
        json.dump({"labels": CLASS_NAMES, "indices": {c: i for i, c in enumerate(CLASS_NAMES)}}, f, indent=2)
    card = {"model": "mobilenet_v2", "labels": CLASS_NAMES, "image_size": SIZE,
            "dataset": {"name": "FoggyCityscapes", "fog": 1000, "non_fog": 500, "total": 1500,
                        "fog_sources": ["Dense_Fog", "Medium_Fog"], "clear_source": "No_Fog"},
            "splits": {k: len(v) for k, v in splits.items()},
            "best_val_acc": best_acc, "best_val_f1": best_f1}
    with open(OUT / "model_card.json", "w") as f: json.dump(card, f, indent=2)

    mnet.eval()
    dummy = torch.randn(1, 3, SIZE, SIZE, device=device)
    onnx_path = OUT / "fog_classifier.onnx"
    torch.onnx.export(mnet, dummy, str(onnx_path), input_names=["input"], output_names=["logits"],
                      dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=18, dynamo=False, do_constant_folding=True)
    print(f"ONNX {onnx_path.stat().st_size/1024/1024:.2f} MB")
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        print("ONNX check PASSED")
    except Exception as e:
        print("ONNX check:", e)
    print("ALL DONE")

if __name__ == "__main__":
    main()
