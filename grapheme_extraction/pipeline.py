"""Milestone 1 pipeline: seal photograph -> ordered list of Mahadevan sign IDs.

    python grapheme_extraction/pipeline.py seal.jpg                 # JSON payload on stdout
    python grapheme_extraction/pipeline.py seal.jpg --direction LTR --annotate out.jpg
    python grapheme_extraction/pipeline.py --eval                   # metrics on the synthetic val split
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from classifier import DATA, DEVICE, ROOT, crop_sign, load_classifier, read_labels, to_tensor

DETECTOR = ROOT / "runs/yolov8n/weights/best.pt"


def detect_yolo(model, img, conf=0.25):
    r = model.predict(img, conf=conf, imgsz=512, device=DEVICE, verbose=False)[0]
    return [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def detect_opencv(img, conf=None):
    """Classical baseline (no learning): local-contrast threshold, merge strokes, take blobs."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # strokes may be darker (seal) or lighter (impression) than the stone: take both
    relief = cv2.max(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k), cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k))
    _, bw = cv2.threshold(cv2.GaussianBlur(relief, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    H, W = gray.shape
    return [(x, y, x + w, y + h, 1.0) for x, y, w, h, area in stats[1:]
            if 12 < h < H * 0.5 and 6 < w < W * 0.5 and area > 40]


def extract(img, detect, classifier, direction="RTL", artifact_id="unknown"):
    """Detect signs, classify each crop, serialise in reading order (PRD section 4 payload)."""
    boxes = detect(img)
    signs = []
    if boxes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with torch.no_grad():
            probs = classifier(to_tensor([crop_sign(gray, b[:4]) for b in boxes]).to(DEVICE)).softmax(1).cpu()
        for (x0, y0, x1, y1, det_conf), p in zip(boxes, probs):
            top = p.topk(5)
            signs.append({"sign_id": int(top.indices[0]) + 1, "confidence": round(float(top.values[0]), 3),
                          "detector_confidence": round(det_conf, 3),
                          "top5": [int(i) + 1 for i in top.indices],
                          "bbox": [round(x0), round(y0), round(x1 - x0), round(y1 - y0)]})
    # ponytail: orders by x only, i.e. assumes a single line of text. Multi-line inscriptions
    # need row clustering on y first; add when real multi-line seals are in the benchmark.
    signs.sort(key=lambda s: s["bbox"][0] + s["bbox"][2] / 2, reverse=direction == "RTL")
    return {"artifact_id": artifact_id, "reading_direction": direction, "sign_count": len(signs),
            "sequence": [s["sign_id"] for s in signs], "bounding_boxes": signs}


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - iw * ih
    return iw * ih / union if union else 0.0


def evaluate(detect, classifier, split="val"):
    """End-to-end metrics against the generator's ground truth. SYNTHETIC DATA ONLY."""
    tp = n_pred = n_gt = cls_ok = edits = gt_len = exact = count_ok = 0
    files = sorted((DATA / "images" / split).glob("*.jpg"))
    t0 = time.perf_counter()
    for f in files:
        img = cv2.imread(str(f))
        gt = sorted(read_labels(f, img.shape[0]), key=lambda g: -(g[1] + g[3]) / 2)  # RTL
        out = extract(img, detect, classifier)
        pred = [(s["sign_id"], s["bbox"][0], s["bbox"][1], s["bbox"][0] + s["bbox"][2], s["bbox"][1] + s["bbox"][3])
                for s in out["bounding_boxes"]]
        used = set()
        for g in gt:  # greedy one-to-one matching at IoU >= 0.5
            best = max((p for p in range(len(pred)) if p not in used), key=lambda p: iou(g[1:], pred[p][1:]), default=None)
            if best is not None and iou(g[1:], pred[best][1:]) >= 0.5:
                used.add(best)
                tp += 1
                cls_ok += pred[best][0] == g[0]
        gt_seq = [g[0] for g in gt]
        n_pred, n_gt = n_pred + len(pred), n_gt + len(gt)
        edits += levenshtein(gt_seq, out["sequence"])
        gt_len += len(gt_seq)
        exact += gt_seq == out["sequence"]
        count_ok += len(gt_seq) == len(pred)
    n = len(files)
    return {"images": n, "signs": n_gt,
            "det_precision@0.5": tp / max(n_pred, 1), "det_recall@0.5": tp / max(n_gt, 1),
            "sign_count_correct": count_ok / n, "cls_top1_on_matched": cls_ok / max(tp, 1),
            "sequence_edit_rate": edits / max(gt_len, 1), "sequence_exact_match": exact / n,
            "ms_per_image": (time.perf_counter() - t0) / n * 1000}


def demo():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    assert levenshtein([], [4, 5]) == 2
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    fake = lambda img: [(10, 10, 40, 60, 0.9), (300, 10, 330, 60, 0.9), (150, 10, 180, 60, 0.9)]
    clf = lambda x: torch.zeros(len(x), 417)
    xs = lambda d: [b["bbox"][0] for b in extract(np.zeros((512, 512, 3), np.uint8), fake, clf, d)["bounding_boxes"]]
    assert xs("RTL") == [300, 150, 10] and xs("LTR") == [10, 150, 300]
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--direction", choices=["RTL", "LTR"], default="RTL",
                    help="RTL for impressions (the default reading); LTR when photographing the seal itself (mirror image)")
    ap.add_argument("--detector", default=str(DETECTOR), help="detector weights, or 'opencv' for the classical baseline")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--annotate", help="write a copy of the image with boxes and sign IDs drawn")
    ap.add_argument("--eval", action="store_true", help="evaluate on the synthetic val split")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        demo()
        raise SystemExit
    if args.detector == "opencv":
        detect = detect_opencv
    else:
        model = YOLO(args.detector)  # ultralytics' YOLO() also loads RT-DETR checkpoints
        detect = lambda img: detect_yolo(model, img, args.conf)
    classifier = load_classifier()

    if args.eval:
        print(json.dumps(evaluate(detect, classifier), indent=1))
    else:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"cannot read image: {args.image}")
        out = extract(img, detect, classifier, args.direction, Path(args.image).stem)
        print(json.dumps(out, indent=1))
        if args.annotate:
            for s in out["bounding_boxes"]:
                x, y, w, h = s["bbox"]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, str(s["sign_id"]), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imwrite(args.annotate, img)
