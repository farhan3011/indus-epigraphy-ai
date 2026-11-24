"""Milestone 1 extension: does a pretrained segmenter isolate Indus signs with no training at all?

    python grapheme_extraction/zero_shot.py           # about 25 minutes on a laptop (SAM runs on the CPU)
    python grapheme_extraction/zero_shot.py --quick   # a fast check, writes to reports/quick

Exploratory. Two segmenters, used as they come: FastSAM (weights models/FastSAM.pt, from huggingface.co/spaces/An-619/FastSAM,
fetched by hand because GitHub release downloads are blocked on this network) and SAM ViT-B automatic mask generation
(facebook/sam-vit-base through transformers). Masks are turned into boxes and filtered by a fixed rule (width 2-25% of the
image, aspect 0.2-5, centre in the upper half where inscriptions sit, no box containing three or more other kept boxes).
YOLO-World was planned too, but its weights are only on GitHub release assets, which this network cannot reach.

Scored against known boxes on synthetic test seals and, when the hand-labelled set is present, on the real test photographs
(data/real/manifest.csv split "test"), with the synthetic-trained YOLOv8n as the reference. Writes reports/milestone_1_zero_shot/.
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from classifier import ROOT, read_labels
from pipeline import DETECTOR, iou, load_detector

OUT = ROOT / "reports/milestone_1_zero_shot"
FASTSAM = ROOT / "models/FastSAM.pt"
REAL = ROOT / "data/real"
SYNTH = ROOT / "data/synthetic/images/test"
SAMPLES = sorted((ROOT / "data/samples").glob("met_*.jpg"))
THRESHOLD = 0.70  # recall on real photos at which a method would be worth using for label drafts


# ---------- methods ----------

def keep_sign_like(boxes, w, h):
    """The fixed filter: sign-sized, sign-shaped, in the upper half, and not a box around several kept boxes."""
    out = [b for b in boxes if 0.02 * w <= b[2] - b[0] <= 0.25 * w and 0.2 <= (b[3] - b[1]) / max(b[2] - b[0], 1) <= 5
           and (b[1] + b[3]) / 2 < 0.5 * h]
    inside = lambda a, b: a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3] and a != b
    return [a for a in out if sum(inside(a, b) for b in out) < 3]


def fastsam():
    from ultralytics import FastSAM
    model = FastSAM(str(FASTSAM))
    def detect(img):
        r = model(img, imgsz=640, conf=0.25, iou=0.7, retina_masks=False, verbose=False)[0]
        return keep_sign_like([tuple(map(float, b)) for b in r.boxes.xyxy.tolist()], img.shape[1], img.shape[0])
    return detect


def sam():
    from PIL import Image
    from transformers import pipeline
    gen = pipeline("mask-generation", model="facebook/sam-vit-base", device="cpu")
    def detect(img):
        boxes = []
        for m in gen(Image.fromarray(img[:, :, ::-1]), points_per_batch=64, pred_iou_thresh=0.86)["masks"]:
            ys, xs = np.nonzero(np.asarray(m))
            if len(xs):
                boxes.append((float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)))
        return keep_sign_like(boxes, img.shape[1], img.shape[0])
    return detect


def trained():
    det = load_detector(DETECTOR)
    return lambda img: [b[:4] for b in det(img)]


# ---------- data and scoring ----------

def synthetic_seals(n):
    out = []
    for f in sorted(SYNTH.glob("*.jpg"))[:n]:
        img = cv2.imread(str(f))
        out.append((f.stem, img, [g[1:] for g in read_labels(f, img.shape[0])]))
    return out


def real_photos():
    """Real test photographs with their hand-drawn boxes, or [] when the labelled set is not on this machine."""
    man, ann = REAL / "manifest.csv", REAL / "annotations.csv"
    if not ann.exists() or "split" not in (open(man).readline() if man.exists() else ""):
        return []
    test = {r["image_id"] for r in csv.DictReader(open(man)) if r["split"] == "test"}
    boxes = defaultdict(list)
    for a in csv.DictReader(open(ann)):
        if a["image_id"] in test:
            boxes[a["image_id"]].append(tuple(int(a[k]) for k in ("x0", "y0", "x1", "y1")))
    return [(i, cv2.imread(str(REAL / "images" / f"{i}.jpg")), boxes[i]) for i in sorted(test) if (REAL / "images" / f"{i}.jpg").exists()]


def match(pred, gold):
    used, tp = set(), 0
    for g in gold:
        best = max((k for k in range(len(pred)) if k not in used), key=lambda k: iou(g, pred[k]), default=None)
        if best is not None and iou(g, pred[best]) >= 0.5:
            used.add(best)
            tp += 1
    return tp


def score(detect, items, draw_to=None):
    tp = npred = ngt = 0
    for name, img, gold in items:
        pred = detect(img)
        tp, npred, ngt = tp + match(pred, gold), npred + len(pred), ngt + len(gold)
        if draw_to:
            cv2.imwrite(str(draw_to / f"{name}.jpg"), draw(img, pred, gold))
    return {"images": len(items), "true_boxes": ngt, "predicted": npred, "precision": tp / npred if npred else 0.0, "recall": tp / ngt if ngt else 0.0}


def draw(img, pred, gold=()):
    img = img.copy()
    for b in gold:
        cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:4])), (60, 170, 60), 2)
    for b in pred:
        cv2.rectangle(img, tuple(map(int, b[:2])), tuple(map(int, b[2:4])), (40, 40, 220), 2)
    return img


# ---------- report ----------

def write_report(res, out):
    s = res["settings"]
    L = ["# Zero-shot sign detection with pretrained segmenters (exploratory)", "",
         "Can a segmenter trained on everyday photographs isolate Indus signs with no training on them? FastSAM and SAM ViT-B produce masks; "
         "each mask becomes a box, and a fixed rule keeps the sign-like ones (width 2-25% of the image, aspect 0.2-5, centre in the upper half, "
         "not a box around three or more kept boxes). The synthetic-trained YOLOv8n is the reference. Boxes count as found at IoU 0.5.", "",
         "| Method | Synthetic seals: precision | recall | Real test photos: precision | recall |", "|---|---|---|---|---|"]
    for m, r in res["methods"].items():
        syn, real = r["synthetic"], r.get("real")
        L.append(f"| {m} | {syn['precision']:.2f} | {syn['recall']:.2f} | " + (f"{real['precision']:.2f} | {real['recall']:.2f} |" if real else "n/a | n/a |"))
    L += ["", f"Synthetic: the first {res['methods']['FastSAM']['synthetic']['images']} test seals for FastSAM and YOLOv8n, the first "
          f"{res['methods']['SAM ViT-B']['synthetic']['images']} for SAM (it takes about 25 seconds a photo on the CPU)."
          + (f" Real: the {res['real_photos']} hand-labelled test photographs ({res['methods']['FastSAM']['real']['true_boxes']} signs)." if res["real_photos"] else
             " Real: the hand-labelled set is not on this machine."), "",
          "## Reading", "", *[f"- {x}" for x in res["reading"]], "",
          "## Not run", "",
          "YOLO-World (open-vocabulary detection with prompts such as \"symbol\" or \"carved sign\") ships in the pinned ultralytics, but its weights "
          "download from GitHub release assets, which this network blocks, and no official mirror was found.", "",
          "## Figures", "",
          "`figures/` holds the four Met photographs with each method's kept boxes (red) and, for labelled photos, the hand-drawn boxes (green). "
          "The real test photographs are drawn locally but not published: some are research-use-only images.", "",
          f"Settings: run time {s['runtime_min']:.1f} minutes, quick run: {s['quick']}.", ""]
    (out / "zero_shot_report.md").write_text("\n".join(L))


def reading(res):
    real = {m: r["real"]["recall"] for m, r in res["methods"].items() if r.get("real")}
    zs = {m: v for m, v in real.items() if m != "YOLOv8n (synthetic-trained)"}
    if not zs:
        return ["No real labelled photographs were available, so only the synthetic comparison was made."]
    best = max(zs, key=zs.get)
    out = [f"On the real test photographs the best zero-shot method, {best}, finds {100 * zs[best]:.0f}% of the signs, against "
           f"{100 * real.get('YOLOv8n (synthetic-trained)', 0):.0f}% for the synthetic-trained YOLOv8n."]
    out.append("That passes the 70% bar for drafting labels." if zs[best] >= THRESHOLD else
               "That is below the 70% bar set in advance, so zero-shot segmentation does not solve sign detection here and the labelling helpers are left as they are.")
    return out


def selfcheck():
    boxes = [(10, 10, 30, 40), (0, 0, 100, 100), (40, 10, 60, 40), (10, 70, 30, 95), (5, 5, 90, 45), (12, 12, 28, 38), (42, 12, 58, 38)]
    kept = keep_sign_like(boxes, 100, 100)
    assert (10, 10, 30, 40) in kept and (0, 0, 100, 100) not in kept and (10, 70, 30, 95) not in kept  # too wide; lower half
    assert (5, 5, 90, 45) not in kept  # too wide anyway; a sign-sized box holding three others would also go
    assert match([(10, 10, 30, 40)], [(11, 11, 31, 41), (50, 50, 60, 60)]) == 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    n_fast, n_sam = (5, 2) if args.quick else (100, 30)
    out = ROOT / "reports/quick" if args.quick else OUT
    (out / "figures").mkdir(parents=True, exist_ok=True)
    real = real_photos()
    res = {"methods": {}, "real_photos": len(real)}
    for name, make, n in (("YOLOv8n (synthetic-trained)", trained, n_fast), ("FastSAM", fastsam, n_fast), ("SAM ViT-B", sam, n_sam)):
        print(name, "...", flush=True)
        detect = make()
        res["methods"][name] = {"synthetic": score(detect, synthetic_seals(n))}
        if real:
            res["methods"][name]["real"] = score(detect, real[:2] if args.quick else real)
        if name != "YOLOv8n (synthetic-trained)":
            for f in SAMPLES:
                img = cv2.imread(str(f))
                cv2.imwrite(str(out / "figures" / f"{f.stem}_{name.split()[0].lower()}.jpg"), draw(img, detect(img)))
    res["reading"] = reading(res)
    res["settings"] = {"quick": args.quick, "synthetic_fastsam": n_fast, "synthetic_sam": n_sam, "runtime_min": (time.time() - start) / 60}
    (out / "zero_shot.json").write_text(json.dumps(res, indent=1))
    write_report(res, out)
    print("\n".join(res["reading"]))
    print(f"done in {res['settings']['runtime_min']:.1f} min -> {out / 'zero_shot_report.md'}")
