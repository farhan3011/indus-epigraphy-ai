"""Sign classifier: cropped sign -> top-k Mahadevan sign IDs (1-417).

Frozen DINOv2 embeddings + cosine nearest neighbour against a reference bank. IM-417 has one
drawing per sign, far too little for a softmax head; a bank works from the first example and
improves just by appending real labelled crops (no retraining).

    python grapheme_extraction/classifier.py     # build bank, tune temperature, report val accuracy
"""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/synthetic"
BANK = ROOT / "models/bank.pt"
N_SIGNS = 417
CROP = 224
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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
    s = (CROP - 16) / max(c.shape)
    c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    out = np.full((CROP, CROP), int(np.median(c)), np.uint8)
    oy, ox = (CROP - c.shape[0]) // 2, (CROP - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def to_tensor(crops):
    """(N, CROP, CROP) uint8 -> ImageNet-normalised (N, 3, CROP, CROP)."""
    x = torch.from_numpy(np.stack(crops)).float().div(255)[:, None].expand(-1, 3, -1, -1)
    return (x - MEAN) / STD


@torch.no_grad()
def embed(backbone, x, batch=64):
    """L2-normalised DINOv2 CLS embeddings, on CPU."""
    out = [backbone(pixel_values=x[i:i + batch].to(DEVICE)).last_hidden_state[:, 0].cpu() for i in range(0, len(x), batch)]
    return torch.nn.functional.normalize(torch.cat(out), dim=1)


def load_backbone():
    return AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()


def class_scores(q, bank_emb, bank_lbl):
    """Best cosine similarity per class: (N, D) queries -> (N, N_SIGNS)."""
    sims = q @ bank_emb.T
    scores = torch.full((len(q), N_SIGNS), -1.0)
    return scores.scatter_reduce(1, bank_lbl.expand(len(q), -1), sims, reduce="amax")


def load_classifier():
    """Returns f(tensor of crops) -> (N, N_SIGNS) logits; softmax gives calibrated-ish top-k probabilities."""
    backbone, bank = load_backbone(), torch.load(BANK)
    return lambda x: class_scores(embed(backbone, x), bank["emb"], bank["lbl"]) / bank["T"]


def read_labels(img_path, size):
    """YOLO label file -> [(sign_id, x0, y0, x1, y1)] in pixels."""
    img_path = Path(img_path)  # <root>/images/<split>/x.jpg -> <root>/labels/<split>/x.txt
    lbl = img_path.parents[2] / "labels" / img_path.parent.name / (img_path.stem + ".txt")
    out = []
    for line in lbl.read_text().splitlines():
        c, x, y, w, h = map(float, line.split())
        out.append((int(c) + 1, (x - w / 2) * size, (y - h / 2) * size, (x + w / 2) * size, (y + h / 2) * size))
    return out


def load_crops(split, jitter=0.0):
    """Labelled sign crops from a YOLO-format split. Real labelled seals dropped into the same
    images/labels layout are picked up the same way."""
    crops, labels = [], []
    for f in sorted((DATA / "images" / split).glob("*.jpg")):
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        for sign, *box in read_labels(f, gray.shape[0]):
            crops.append(crop_sign(gray, box, jitter))
            labels.append(sign - 1)
    return to_tensor(crops), torch.tensor(labels)


def build():
    random.seed(0)
    backbone = load_backbone()
    xtr, ytr = load_crops("train", jitter=0.08)  # jitter mimics detector box error
    xva, yva = load_crops("val")
    print(f"bank: {len(ytr)} crops, {len(ytr.unique())}/{N_SIGNS} signs; val: {len(yva)} crops; device={DEVICE}")
    emb = embed(backbone, xtr)
    scores = class_scores(embed(backbone, xva), emb, ytr)

    # Temperature only rescales scores into probabilities (ranking is unchanged): pick it by val NLL.
    T = min((0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1), key=lambda t: torch.nn.functional.cross_entropy(scores / t, yva).item())
    BANK.parent.mkdir(exist_ok=True)
    torch.save({"emb": emb, "lbl": ytr, "T": T}, BANK)

    top5 = scores.topk(5).indices
    top1 = (top5[:, 0] == yva).float().mean().item()
    top5_acc = (top5 == yva[:, None]).any(1).float().mean().item()
    pairs = {}
    for t, p in zip(yva.tolist(), top5[:, 0].tolist()):
        if t != p:
            pairs[(t + 1, p + 1)] = pairs.get((t + 1, p + 1), 0) + 1
    confused = [{"true": t, "pred": p, "n": n} for (t, p), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:25]]
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/classifier_metrics.json").write_text(json.dumps(
        {"note": "SYNTHETIC val crops: augmentations of the same 417 source drawings, not real seals",
         "val_top1": top1, "val_top5": top5_acc, "val_crops": len(yva), "temperature": T, "most_confused": confused}, indent=1))
    print(f"synthetic val top-1 {top1:.3f}  top-5 {top5_acc:.3f}  T={T}  -> {BANK}")


if __name__ == "__main__":
    q, emb, lbl = torch.eye(3), torch.eye(3)[[0, 0, 2]], torch.tensor([[5, 7, 7]])
    s = class_scores(q, emb, lbl)
    assert s[:, [5, 7]].tolist() == [[1, 1], [0, 0], [0, 1]] and s[0, 6] == -1  # max per class, -1 if class absent
    build()
