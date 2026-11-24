"""Milestone 1 on hand-labelled real photographs: fine-tune the detector from its synthetic weights, add real crops to
the classifier bank, and score both stages on the frozen real test split against the synthetic-only models.

    python grapheme_extraction/import_labels.py export.json --package data/raw/indus_hand_labelling   # once per export
    python grapheme_extraction/train_real.py                                                           # ~10 min on a laptop

Writes runs/yolov8n-real/ (weights), models/bank_real.pt and reports/milestone_1/real/ (report, results, figures).
The test split is small: read every interval.
"""
import csv
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from classifier import BANK, DEVICE, ROOT, crop_sign, embed, load_backbone, load_classifier, to_tensor
from evaluate import bootstrap, fmt
from pipeline import annotate, extract, iou, levenshtein, load_detector

REAL = ROOT / "data/real"
YOLO_DIR = REAL / "yolo"
OUT = ROOT / "reports/milestone_1/real"
SYNTH_DET = ROOT / "runs/yolov8n/weights/best.pt"
SYNTH_MIX = 300  # synthetic seals mixed into the "mixed" fine-tune, so it does not forget them
REAL_BANK = ROOT / "models/bank_real.pt"
JITTERS = 8  # augmented copies of each real training crop in the bank


def load_real():
    """{image_id: {"split", "img", "boxes": [(sign, x0, y0, x1, y1)] in reading order}} for every used image."""
    rows = {r["image_id"]: r for r in csv.DictReader(open(REAL / "manifest.csv")) if r["split"]}
    boxes = defaultdict(list)
    for a in csv.DictReader(open(REAL / "annotations.csv")):
        boxes[a["image_id"]].append((int(a["order"]), int(a["sign_id"]), *(int(a[k]) for k in ("x0", "y0", "x1", "y1"))))
    return {i: {"split": r["split"], "img": cv2.imread(str(REAL / "images" / f"{i}.jpg")),
                "boxes": [b[1:] for b in sorted(boxes[i])]} for i, r in rows.items()}


def yolo_dataset(data):
    """Single-class YOLO layout of the real splits; every box counts as a sign, illegible ones included."""
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)
    for i, d in data.items():
        (YOLO_DIR / "images" / d["split"]).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / d["split"]).mkdir(parents=True, exist_ok=True)
        h, w = d["img"].shape[:2]
        cv2.imwrite(str(YOLO_DIR / "images" / d["split"] / f"{i}.jpg"), d["img"])
        (YOLO_DIR / "labels" / d["split"] / f"{i}.txt").write_text("\n".join(
            f"0 {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}" for _, x0, y0, x1, y1 in d["boxes"]))
    synth = sorted((ROOT / "data/synthetic/images/train").glob("*.jpg"))[:SYNTH_MIX]
    (YOLO_DIR / "images/train_synth").mkdir(parents=True)
    (YOLO_DIR / "labels/train_synth").mkdir(parents=True)
    for f in synth:  # sign-ID classes collapse under single_cls
        shutil.copy(f, YOLO_DIR / "images/train_synth" / f.name)
        shutil.copy(f.parents[2] / "labels/train" / f"{f.stem}.txt", YOLO_DIR / "labels/train_synth" / f"{f.stem}.txt")
    yamls = {}
    for name, train in (("real", "images/train"), ("mixed", "[images/train, images/train_synth]")):
        yamls[name] = YOLO_DIR / f"{name}.yaml"
        yamls[name].write_text(f"path: {YOLO_DIR.as_posix()}\ntrain: {train}\nval: images/val\ntest: images/test\nnames:\n  0: sign\n")
    return yamls


def fine_tune(yaml, name, epochs):
    """From the synthetic detector; flips stay off (a mirrored sign is another sign). Returns the best weights."""
    YOLO(str(SYNTH_DET)).train(data=str(yaml), single_cls=True, imgsz=512, epochs=epochs, patience=25, batch=8,
                               device=DEVICE, amp=torch.cuda.is_available(), fliplr=0.0, flipud=0.0, plots=False,
                               project=str(ROOT / "runs"), name=name, exist_ok=True, verbose=False)
    return ROOT / "runs" / name / "weights/best.pt"


def f1(det):
    p, r = det["precision"][0], det["recall"][0]
    return 2 * p * r / (p + r) if p + r else 0.0


def real_bank(data):
    """The synthetic bank plus jittered crops of every legible sign in the real train and val photos."""
    random.seed(0)
    crops, labels = [], []
    for d in data.values():
        if d["split"] in ("train", "val"):
            gray = cv2.cvtColor(d["img"], cv2.COLOR_BGR2GRAY)
            for sign, *box in d["boxes"]:
                if sign:
                    crops += [crop_sign(gray, box, 0.08 if k else 0) for k in range(JITTERS)]
                    labels += [sign - 1] * JITTERS
    bank = torch.load(BANK)
    emb = embed(load_backbone(), to_tensor(crops))
    torch.save({"emb": torch.cat([bank["emb"], emb]), "lbl": torch.cat([bank["lbl"], torch.tensor(labels)]), "T": bank["T"]}, REAL_BANK)
    return len(crops) // JITTERS, len(set(labels))


def detection(detect, test):
    """Per photo: true positives at IoU >= 0.5 (greedy one-to-one), predictions, true boxes."""
    rows = []
    for d in test.values():
        pred = [p[:4] for p in detect(d["img"])]
        used, tp = set(), 0
        for _, *g in d["boxes"]:
            best = max((k for k in range(len(pred)) if k not in used), key=lambda k: iou(g, pred[k]), default=None)
            if best is not None and iou(g, pred[best]) >= 0.5:
                used.add(best)
                tp += 1
        rows.append((tp, len(pred), len(d["boxes"])))
    tp, npred, ngt = (np.array(x) for x in zip(*rows))
    return {"precision": bootstrap(tp, npred), "recall": bootstrap(tp, ngt), "photos": len(rows), "true_signs": int(ngt.sum()), "predicted": int(npred.sum())}


def classification(classify, test):
    """Top-1 / top-5 on the true boxes of legible test signs (no detector involved)."""
    crops, labels, photo = [], [], []
    for k, d in enumerate(test.values()):
        gray = cv2.cvtColor(d["img"], cv2.COLOR_BGR2GRAY)
        for sign, *box in d["boxes"]:
            if sign:
                crops.append(crop_sign(gray, box)); labels.append(sign - 1); photo.append(k)
    with torch.no_grad():
        top5 = classify(to_tensor(crops)).topk(5).indices.cpu()
    y = torch.tensor(labels)
    hit1, hit5 = (top5[:, 0] == y).numpy().astype(float), (top5 == y[:, None]).any(1).numpy().astype(float)
    per = lambda h: [h[np.array(photo) == k].sum() for k in range(len(test))]
    n = [np.sum(np.array(photo) == k) for k in range(len(test))]
    return {"crops": len(labels), "top1": bootstrap(per(hit1), n), "top5": bootstrap(per(hit5), n)}


def end_to_end(detect, classify, test):
    """Sign error rate (edit distance over legible true signs, reading order) and exact match per photo."""
    edits, lens, exact, outputs = [], [], [], {}
    for i, d in test.items():
        out = extract(d["img"], detect, classify, artifact_id=i)
        gold = [s for s, *_ in d["boxes"] if s]
        edits.append(levenshtein(gold, out["sequence"])); lens.append(len(gold)); exact.append(int(gold == out["sequence"]))
        outputs[i] = out
    return {"sign_error_rate": bootstrap(edits, lens), "exact_match": bootstrap(exact, [1] * len(exact))}, outputs


def write_report(res):
    s, rows = res["data"], []
    for stage, key, label in (("detection", "precision", "Detector precision (IoU 0.5)"), ("detection", "recall", "Detector recall (IoU 0.5)"),
                              ("classification", "top1", "Classifier top-1 on true boxes"), ("classification", "top5", "Classifier top-5 on true boxes"),
                              ("end_to_end", "sign_error_rate", "Sign error rate, whole pipeline"), ("end_to_end", "exact_match", "Exact sequence match")):
        rows.append(f"| {label} | {fmt(res['synthetic_only'][stage][key])} | {fmt(res['with_real'][stage][key])} |")
    lines = [
        "# Milestone 1 on real seal photographs", "",
        f"**First measurement on real photographs.** {s['labelled']} photographs were hand-labelled ({s['boxes']} sign boxes, "
        f"{s['illegible']} marked illegible) and split by photograph, with photos of the same object kept together: "
        f"{s['split']['train']} train, {s['split']['val']} val, {s['split']['test']} test. The test photos were never used for training, "
        "bank building or model selection. With only "
        f"{res['synthetic_only']['detection']['true_signs']} test signs on {s['split']['test']} photos the intervals are wide; read them before the "
        "point estimates.", "",
        "| Measure (real test photos) | Synthetic-only models | Detector fine-tuned and bank extended on real photos |", "|---|---|---|", *rows, "",
        f"- Detector: the synthetic YOLOv8n, and two fine-tunes from it on the {s['split']['train']} real training photos (alone, and mixed with "
        f"{SYNTH_MIX} synthetic seals), flips off. The one carried into the right-hand column was chosen by F1 on the "
        f"{s['split']['val']} val photos: **{res['detector_choice']['chosen']}**.",
        f"- The extended bank adds {s['bank_signs_added']} legible real sign crops ({s['bank_classes_added']} distinct signs, {JITTERS} jittered copies each) "
        "from the train and val photos to the synthetic bank; no retraining.",
        "- Brackets are 95% bootstrap intervals over test photos. Illegible signs count for detection but are left out of the classifier "
        "and sequence measures.",
        "- Sequences are read in impression view; photos of carved seals were mirrored on import.", "",
        "| Detector | Val precision | Val recall | Test precision | Test recall |", "|---|---|---|---|---|",
        *[f"| {k} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(res['detector_choice']['test'][k]['precision'])} | "
          f"{fmt(res['detector_choice']['test'][k]['recall'])} |" for k, v in res["detector_choice"]["val"].items()], "",
        "`figures/` (local only: the photos are research-use only and not published) shows each test photo with the boxes and top-1 sign numbers of both systems (left: synthetic-only, right: with real).", ""]
    (OUT / "real_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    data = load_real()
    test = {i: d for i, d in data.items() if d["split"] == "test"}
    assert test, "no test photos: run import_labels.py first"
    counts = {k: sum(d["split"] == k for d in data.values()) for k in ("train", "val", "test")}
    yamls = yolo_dataset(data)
    val = {i: d for i, d in data.items() if d["split"] == "val"}
    detectors = {"synthetic only": SYNTH_DET, **{f"fine-tuned, {k}": fine_tune(y, f"yolov8n-real-{k}", args.epochs) for k, y in yamls.items()}}
    # the detector to carry forward is chosen on the val photos only
    val_det = {k: detection(load_detector(w), val) for k, w in detectors.items()}
    chosen = max(val_det, key=lambda k: f1(val_det[k]))
    added, classes = real_bank(data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    res = {"data": {"labelled": len(data), "boxes": sum(len(d["boxes"]) for d in data.values()),
                    "illegible": sum(s == 0 for d in data.values() for s, *_ in d["boxes"]), "split": counts,
                    "bank_signs_added": added, "bank_classes_added": classes, "epochs": args.epochs},
           "detector_choice": {"chosen": chosen, "val": val_det,
                               "test": {k: detection(load_detector(w), test) for k, w in detectors.items()}}}
    outputs = {}
    for name, det, bank in (("synthetic_only", SYNTH_DET, BANK), ("with_real", detectors[chosen], REAL_BANK)):
        detect, classify = load_detector(det), load_classifier(bank)
        e2e, outputs[name] = end_to_end(detect, classify, test)
        res[name] = {"detection": detection(detect, test), "classification": classification(classify, test), "end_to_end": e2e}
        print(name, {k: {m: round(v[0], 3) for m, v in r.items() if isinstance(v, list)} for k, r in res[name].items()}, flush=True)
    for i, d in test.items():
        a, b = (annotate(d["img"], outputs[n][i]) for n in ("synthetic_only", "with_real"))
        cv2.imwrite(str(OUT / "figures" / f"{i}.jpg"), np.hstack([a, b]))
    (OUT / "real_results.json").write_text(json.dumps(res, indent=1))
    write_report(res)
    print(f"report -> {OUT / 'real_report.md'}")
