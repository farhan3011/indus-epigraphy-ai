"""Sign classifier: cropped sign -> Mahadevan sign ID (1-417), MobileNetV2 fine-tune.

Trained on sign crops cut from the synthetic seals (boxes jittered to mimic detector error),
so its input distribution matches what the pipeline feeds it.

    python grapheme_extraction/classifier.py
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/synthetic"
WEIGHTS = ROOT / "models/classifier.pt"
N_SIGNS = 417
CROP = 96
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def crop_sign(gray, box, jitter=0.0):
    """Grayscale crop letterboxed to CROP x CROP (aspect ratio kept: it separates many signs)."""
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if jitter:
        x0, x1 = x0 + random.uniform(-jitter, jitter) * w, x1 + random.uniform(-jitter, jitter) * w
        y0, y1 = y0 + random.uniform(-jitter, jitter) * h, y1 + random.uniform(-jitter, jitter) * h
    H, W = gray.shape
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(W, int(x1 + 1)), min(H, int(y1 + 1))
    c = gray[y0:y1, x0:x1]
    if c.size == 0:
        return np.zeros((CROP, CROP), np.uint8)
    c = cv2.normalize(c, None, 0, 255, cv2.NORM_MINMAX)  # removes stone colour / lighting level
    s = (CROP - 8) / max(c.shape)
    c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((CROP, CROP), int(np.median(c)), np.uint8)
    oy, ox = (CROP - c.shape[0]) // 2, (CROP - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def to_tensor(crops):
    """(N, CROP, CROP) uint8 -> (N, 3, CROP, CROP) float in [-1, 1]."""
    x = torch.from_numpy(np.stack(crops)).float().div(127.5).sub(1)
    return x[:, None].expand(-1, 3, -1, -1)


def build_model():
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, N_SIGNS)
    return model


def load_classifier():
    model = build_model()
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    return model.to(DEVICE).eval()


def read_labels(img_path, size):
    """YOLO label file -> [(sign_id, x0, y0, x1, y1)] in pixels."""
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    out = []
    for line in lbl.read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        out.append((int(c) + 1, (x - w / 2) * size, (y - h / 2) * size, (x + w / 2) * size, (y + h / 2) * size))
    return out


def load_crops(split, variants, jitter):
    crops, labels = [], []
    for f in sorted((DATA / "images" / split).glob("*.jpg")):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        for sign, *box in read_labels(f, gray.shape[0]):
            for _ in range(variants):
                crops.append(crop_sign(gray, box, jitter))
                labels.append(sign - 1)
    return to_tensor(crops), torch.tensor(labels)


def train(epochs=25, batch=128):
    random.seed(0)
    torch.manual_seed(0)
    xtr, ytr = load_crops("train", variants=3, jitter=0.08)
    xva, yva = load_crops("val", variants=1, jitter=0.0)
    print(f"{len(ytr)} train crops, {len(yva)} val crops, {len(ytr.unique())}/{N_SIGNS} classes seen, device={DEVICE}")

    model = build_model()
    pretrained = MobileNet_V2_Weights.DEFAULT.get_state_dict(progress=False)
    model.load_state_dict({k: v for k, v in pretrained.items() if not k.startswith("classifier.1")}, strict=False)
    model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(ytr) // batch + 1))
    # No class weights: IM-417 has exactly one source image per sign and signs are sampled
    # uniformly, so the training set is balanced by construction.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    for ep in range(epochs):
        model.train()
        perm, total = torch.randperm(len(ytr)), 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model(xtr[idx].to(DEVICE)), ytr[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        top1, top5, _ = evaluate(model, xva, yva)
        history.append({"epoch": ep + 1, "train_loss": total / len(ytr), "val_top1": top1, "val_top5": top5})
        print(history[-1])

    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    top1, top5, pred = evaluate(model, xva, yva)
    pairs = {}
    for t, p in zip(yva.tolist(), pred.tolist()):
        if t != p:
            pairs[(t + 1, p + 1)] = pairs.get((t + 1, p + 1), 0) + 1
    confused = [{"true": t, "pred": p, "n": n} for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:25]]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/classifier_metrics.json").write_text(json.dumps(
        {"val_top1": top1, "val_top5": top5, "val_crops": len(yva), "history": history, "most_confused": confused}, indent=1))
    print(f"val top-1 {top1:.3f}  top-5 {top5:.3f}  -> {WEIGHTS}")


@torch.no_grad()
def evaluate(model, x, y, batch=256):
    model.eval()
    logits = torch.cat([model(x[i:i + batch].to(DEVICE)).cpu() for i in range(0, len(y), batch)])
    top5 = logits.topk(5).indices
    return ((top5[:, 0] == y).float().mean().item(), (top5 == y[:, None]).any(1).float().mean().item(), top5[:, 0])


if __name__ == "__main__":
    train()
