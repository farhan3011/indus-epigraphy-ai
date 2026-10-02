"""Milestone 1 pipeline: seal photograph -> ordered list of Mahadevan sign IDs.

    python grapheme_extraction/pipeline.py seal.jpg                 # JSON payload on stdout
    python grapheme_extraction/pipeline.py seal.jpg --seal --annotate out.jpg  # photo of the seal itself

Metrics live in evaluate.py.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import RTDETR, YOLO

from classifier import DEVICE, ROOT, crop_sign, load_classifier, to_tensor

DETECTOR = ROOT / "runs/yolov8n/weights/best.pt"


def detect_yolo(model, img, conf=0.25):
    r = model.predict(img, conf=conf, imgsz=512, device=DEVICE, verbose=False)[0]
    return [(*b, c) for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]


def load_detector(spec, conf=0.25):
    """'opencv' or a weights path -> f(BGR image) -> [(x0, y0, x1, y1, conf)]."""
    if str(spec) == "opencv":
        return detect_opencv
    # RT-DETR has its own NMS-free predictor; loading it through YOLO() would post-process it wrongly
    model = (RTDETR if "rtdetr" in str(spec).lower() else YOLO)(str(spec))
    return lambda img: detect_yolo(model, img, conf)


def detect_opencv(img):
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


def group_into_lines(boxes):
    """Line index per box: a new line starts when a box's y-centre sits more than half the
    median box height below the running mean of the current line."""
    if not boxes:
        return []
    yc = [(b[1] + b[3]) / 2 for b in boxes]
    gap = 0.5 * float(np.median([b[3] - b[1] for b in boxes]))
    lines, current, line = [0] * len(boxes), [], 0
    for i in sorted(range(len(boxes)), key=lambda i: yc[i]):
        if current and yc[i] - sum(current) / len(current) > gap:
            line, current = line + 1, []
        current.append(yc[i])
        lines[i] = line
    return lines


def extract(img, detect, classifier, is_seal_matrix=False, artifact_id="unknown"):
    """Detect signs, classify each crop, serialise in reading order.

    Corpora transcribe the impression, read right-to-left. A photo of the seal itself is the
    mirror image, so it is flipped to impression view first; everything downstream (boxes
    included) is in impression-view pixels."""
    if is_seal_matrix:
        img = cv2.flip(img, 1)
    boxes = detect(img)
    signs = []
    if boxes:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        with torch.no_grad():
            probs = classifier(to_tensor([crop_sign(gray, b[:4]) for b in boxes])).softmax(1).cpu()
        for (x0, y0, x1, y1, det_conf), p, line in zip(boxes, probs, group_into_lines(boxes)):
            top = p.topk(5)
            signs.append({"line": line, "bbox_xywh": [round(x0), round(y0), round(x1 - x0), round(y1 - y0)],
                          "det_conf": round(det_conf, 3),
                          "top_k": [[int(i) + 1, round(float(v), 3)] for v, i in zip(top.values, top.indices)]})
    # lines top to bottom; within a line right-to-left
    signs.sort(key=lambda s: (s["line"], -(s["bbox_xywh"][0] + s["bbox_xywh"][2] / 2)))
    for i, s in enumerate(signs):
        s["order"] = i
    # `sequence` is the top-1 reading for convenience. bbox_xywh keeps the physical positions so
    # Milestone 2 can test reading direction instead of inheriting the RTL sort assumed here.
    return {"artifact_id": artifact_id, "is_seal_matrix": is_seal_matrix, "view": "impression",
            "sign_count": len(signs), "sequence": [s["top_k"][0][0] for s in signs], "signs": signs}


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


def annotate(img, out):
    """Copy of the (impression-view) image with boxes and top-1 sign IDs drawn."""
    img = img.copy()
    for s in out["signs"]:
        x, y, w, h = s["bbox_xywh"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, str(s["top_k"][0][0]), (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return img


def demo():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 3]) == 1
    assert levenshtein([], [4, 5]) == 2
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1 and iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0
    fake = lambda img: [(10, 10, 40, 60, 0.9), (300, 10, 330, 60, 0.9), (150, 10, 180, 60, 0.9)]
    clf = lambda x: torch.zeros(len(x), 417)
    out = extract(np.zeros((512, 512, 3), np.uint8), fake, clf)
    assert [s["bbox_xywh"][0] for s in out["signs"]] == [300, 150, 10]  # right-to-left
    # two lines: upper line is read first, each line right-to-left
    two = [(10, 10, 40, 60, 1), (100, 12, 130, 62, 1), (10, 90, 40, 140, 1), (100, 92, 130, 142, 1)]
    assert group_into_lines(two) == [0, 0, 1, 1]
    out = extract(np.zeros((512, 512, 3), np.uint8), lambda img: two, clf)
    assert [s["bbox_xywh"][:2] for s in out["signs"]] == [[100, 12], [10, 10], [100, 92], [10, 90]]
    # mirroring a seal photo reverses which physical end is read first
    lit = lambda img: [(int(x), 10, int(x) + 20, 60, 1) for x in np.flatnonzero(img[30, :, 0])[::20]]
    img = np.zeros((100, 400, 3), np.uint8)
    img[:, 50:70] = 255
    assert extract(img, lit, clf)["signs"][0]["bbox_xywh"][0] == 50
    assert extract(img, lit, clf, is_seal_matrix=True)["signs"][0]["bbox_xywh"][0] == 330
    print("ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--seal", action="store_true",
                    help="the photo shows the carved seal itself (mirror image), not an impression: flip it first")
    ap.add_argument("--detector", default=str(DETECTOR), help="detector weights, or 'opencv' for the classical baseline")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--annotate", help="write a copy of the image with boxes and sign IDs drawn")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        demo()
        raise SystemExit
    img = cv2.imread(args.image or "")
    if img is None:
        raise SystemExit(f"cannot read image: {args.image}")
    out = extract(img, load_detector(args.detector, args.conf), load_classifier(), args.seal, Path(args.image).stem)
    print(json.dumps(out, indent=1))
    if args.annotate:
        cv2.imwrite(args.annotate, annotate(cv2.flip(img, 1) if args.seal else img, out))  # boxes are in impression view
