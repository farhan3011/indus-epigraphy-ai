"""Milestone 1 evaluation on the held-out SYNTHETIC test split, with 95% bootstrap intervals.

    python grapheme_extraction/evaluate.py                          # yolov8n, rtdetr-l and the OpenCV baseline
    python grapheme_extraction/evaluate.py --runs yolov8n --out reports/quick

Writes (default reports/milestone_1/) <out>/eval_results.json, <out>/per_class_f1.csv, <out>/figures/ and <out>/milestone_1_report.md.
"""
import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from ultralytics import RTDETR, YOLO

from classifier import DATA, DEVICE, N_SIGNS, ROOT, load_classifier, load_crops, read_labels
from pipeline import annotate, extract, iou, levenshtein, load_detector, rescore
from synthetic_generator import IM77_CSV, load_glyphs, load_sequences, split_of

sys.path.insert(0, str(ROOT / "sequence_analysis"))
import ngram  # noqa: E402

# metric -> (numerator, denominator) summed over seals; "seals" is 1 per seal
METRICS = {
    "Detection precision (IoU 0.5)": ("tp", "n_pred"),
    "Detection recall (IoU 0.5)": ("tp", "n_gt"),
    "Recall on touching signs": ("touch_tp", "touch_gt"),
    "Sign count correct": ("count_ok", "seals"),
    "Top-1 sign ID on detected signs": ("cls1", "tp"),
    "Top-5 sign ID on detected signs": ("cls5", "tp"),
    "Sign error rate (edit distance / length)": ("edits", "n_gt"),
    "Exact sequence match": ("exact", "seals"),
}
BUCKETS = ["50 most frequent signs", "signs ranked 51-200", "rarer signs"]


def seal_records(detect, classifier, split, keep=None):
    """One record of counts per seal, plus mean milliseconds per image. keep: a list that receives
    (gold sequence, pipeline signs, seal source) per seal, for rescoring without re-running the pipeline."""
    sources = json.loads((DATA / f"sources_{split}.json").read_text()) if (DATA / f"sources_{split}.json").exists() else {}
    records, files = [], sorted((DATA / "images" / split).glob("*.jpg"))
    t0 = time.perf_counter()
    for f in files:
        img = cv2.imread(str(f))
        gt = sorted(read_labels(f, img.shape[0]), key=lambda g: -(g[1] + g[3]) / 2)  # one line, right-to-left
        out = extract(img, detect, classifier)
        pred = [(x, y, x + w, y + h) for x, y, w, h in (s["bbox_xywh"] for s in out["signs"])]
        r = Counter(seals=1, n_gt=len(gt), n_pred=len(pred))
        used = set()
        for g in gt:  # greedy one-to-one matching at IoU >= 0.5
            touching = any(h is not g and iou(g[1:], h[1:]) > 0 for h in gt)
            r["touch_gt"] += touching
            best = max((p for p in range(len(pred)) if p not in used), key=lambda p: iou(g[1:], pred[p]), default=None)
            if best is not None and iou(g[1:], pred[best]) >= 0.5:
                used.add(best)
                top = [i for i, _ in out["signs"][best]["top_k"]]
                r.update(tp=1, touch_tp=int(touching), cls1=int(top[0] == g[0]), cls5=int(g[0] in top))
        gold = [g[0] for g in gt]
        r.update(edits=levenshtein(gold, out["sequence"]), exact=int(gold == out["sequence"]), count_ok=int(len(gold) == len(pred)))
        records.append(r)
        if keep is not None:
            keep.append((gold, out["signs"], sources.get(f.stem, "unknown")))
    return records, (time.perf_counter() - t0) / max(len(files), 1) * 1000


def bootstrap(num, den, n_boot=1000, seed=0):
    """[point, low, high] for sum(num)/sum(den), resampling rows with replacement."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    idx = np.random.default_rng(seed).integers(0, len(num), (n_boot, len(num)))
    boots = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-9)
    return [float(num.sum() / max(den.sum(), 1e-9)), *np.percentile(boots, [2.5, 97.5]).tolist()]


def summarise(records):
    return {name: bootstrap([r[a] for r in records], [r[b] for r in records]) for name, (a, b) in METRICS.items()}


def frequency_buckets():
    """sign index (0-based) -> bucket index, by rank of corpus frequency in IM77."""
    counts = Counter()
    with open(IM77_CSV) as f:
        for row in csv.DictReader(f):
            counts.update(int(t) for t in row["signs_clean"].split() if t != "0")
    ranked = sorted(range(1, N_SIGNS + 1), key=lambda s: -counts[s])
    return {s - 1: 0 if rank < 50 else 1 if rank < 200 else 2 for rank, s in enumerate(ranked)}


def classifier_report(classifier, split, out_dir):
    """Top-1/top-5 on ground-truth crops (no detector involved), by corpus-frequency bucket."""
    x, y = load_crops(split)
    with torch.no_grad():
        top5 = torch.cat([classifier(x[i:i + 256]).topk(5).indices for i in range(0, len(y), 256)])
    hit1, hit5 = (top5[:, 0] == y).numpy(), (top5 == y[:, None]).any(1).numpy()
    bucket = frequency_buckets()
    b = np.array([bucket[int(t)] for t in y])
    ones = np.ones(len(y))
    rows = {"all signs": (hit1, hit5, ones)} | {name: (hit1[b == i], hit5[b == i], ones[b == i]) for i, name in enumerate(BUCKETS)}
    p, r, f1, support = precision_recall_fscore_support(y.numpy(), top5[:, 0].numpy(), labels=range(N_SIGNS), zero_division=0)
    with open(out_dir / "per_class_f1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sign_id", "frequency_bucket", "precision", "recall", "f1", "test_crops"])
        w.writerows([i + 1, BUCKETS[bucket[i]], round(p[i], 3), round(r[i], 3), round(f1[i], 3), int(support[i])] for i in range(N_SIGNS))
    return {name: {"crops": int(len(h1)), "top1": bootstrap(h1, n), "top5": bootstrap(h5, n)} for name, (h1, h5, n) in rows.items()}


def detector_ap(weights, split, name):
    """Single-class AP50 and AP50-95 from ultralytics' own validator."""
    model = (RTDETR if "rtdetr" in str(weights).lower() else YOLO)(str(weights))
    res = model.val(data=str(DATA / "data.yaml"), split=split, single_cls=True, imgsz=512, device=DEVICE,
                    plots=False, verbose=False, project=str(ROOT / "runs"), name=f"{name}-{split}", exist_ok=True)
    return {"AP50": float(res.box.map50), "AP50-95": float(res.box.map)}


WEIGHTS = (0, 0.25, 0.5, 1, 2, 4)


def sequence_lm():
    """Kneser-Ney trigram (the best Milestone 2 n-gram) on the corpus texts of the synthetic train split only, so no
    test-seal text is in it. Returns lm(prefix, sign) -> natural log P, sign None for the end of the text."""
    texts = [tuple(s) for s in load_sequences() if split_of(s) == "train"]
    m = ngram.KneserNey(texts, list(range(1, N_SIGNS + 1)), order=3)
    return lambda prefix, s: math.log(m.prob(ngram.EOS if s is None else s, (ngram.BOS, *prefix)))


SIZE_WEIGHTS = (0, 1, 4, 16, 32, 64, 128, 256)


def size_adjusted(signs, heights, lam):
    """Top-1 sequence after penalising candidates whose drawn height does not fit the box: the classifier letterboxes
    every crop, which hides the size that separates e.g. 99 (two short strokes) from 87 (two long). Heights are compared
    relative to the seal's median, since each seal is drawn at one scale."""
    if len(signs) < 2 or lam == 0:
        return [s["top_k"][0][0] for s in signs]
    obs = np.array([s["bbox_xywh"][3] for s in signs], float)
    ref = np.median([heights[s["top_k"][0][0]] for s in signs])
    out = []
    for o, s in zip(np.log(obs / np.median(obs)), signs):
        out.append(max(s["top_k"], key=lambda c: math.log(max(c[1], 1e-12)) - lam * (o - math.log(heights[c[0]] / ref)) ** 2)[0])
    return out


def rescoring(detect, classifier):
    """Sign error rate and exact match with and without n-gram rescoring of the top-5, weight chosen on val seals."""
    lm, kept = sequence_lm(), {}
    for split in ("val", "test"):
        kept[split] = []
        seal_records(detect, classifier, split, kept[split])
    def ser(rows, w):
        return sum(levenshtein(g, rescore(sg, lm, w)) for g, sg, _ in rows) / sum(len(g) for g, _, _ in rows)
    val = [r for r in kept["val"] if r[2] == "corpus"]  # tuned for real inscriptions, not random sign strings
    weight = min(WEIGHTS, key=lambda w: ser(val, w))
    out = {"weight": weight, "val_corpus_sign_error_rate_by_weight": {str(w): ser(val, w) for w in WEIGHTS}}
    for group in ("all", "corpus", "random"):
        rows = [r for r in kept["test"] if group == "all" or r[2] == group]
        if not rows:
            continue
        res = {}
        for name, w in (("top1", 0), ("rescored", weight)):
            seqs = [rescore(sg, lm, w) for _, sg, _ in rows]
            res[name] = {"sign_error_rate": bootstrap([levenshtein(g, q) for (g, _, _), q in zip(rows, seqs)], [len(g) for g, _, _ in rows]),
                         "exact_match": bootstrap([int(g == q) for (g, _, _), q in zip(rows, seqs)], [1] * len(rows))}
        out[group] = {"seals": len(rows), **res}
    heights = {s: g.shape[0] for s, g in load_glyphs().items()}
    size_ser = lambda rows, lam: sum(levenshtein(g, size_adjusted(sg, heights, lam)) for g, sg, _ in rows) / sum(len(g) for g, _, _ in rows)
    lam = min(SIZE_WEIGHTS, key=lambda l: size_ser(kept["val"], l))
    rows = kept["test"]
    out["size"] = {"weight": lam, "val_sign_error_rate_by_weight": {str(l): size_ser(kept["val"], l) for l in SIZE_WEIGHTS},
                   **{name: bootstrap([levenshtein(g, size_adjusted(sg, heights, l)) for g, sg, _ in rows], [len(g) for g, _, _ in rows])
                      for name, l in (("top1_sign_error_rate", 0), ("size_adjusted_sign_error_rate", lam))}}
    return out


def fmt(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def rescoring_lines(rs):
    lines = []
    lines += ["", "## Re-ranking the classifier's top 5 with a sign-sequence model", "",
              f"YOLOv8n output, each sign's top 5 re-ranked by beam search with the best Milestone 2 n-gram (Kneser-Ney trigram) trained on "
              f"the corpus texts of the training split only; synthetic train, val and test seals use disjoint corpus texts. Weight chosen on "
              f"the val seals that carry corpus texts: {rs['weight']} (0 = no re-ranking).", "",
              "| Test seals | Seals | Sign error rate, top-1 | Sign error rate, re-ranked | Exact match, top-1 | Exact match, re-ranked |",
              "|---|---|---|---|---|---|"]
    for group, label in (("all", "All"), ("corpus", "Real corpus texts"), ("random", "Random sign strings")):
        if group in rs:
            g = rs[group]
            lines.append(f"| {label} | {g['seals']} | {fmt(g['top1']['sign_error_rate'])} | {fmt(g['rescored']['sign_error_rate'])} | "
                         f"{fmt(g['top1']['exact_match'])} | {fmt(g['rescored']['exact_match'])} |")
    lines += ["", "Half the synthetic seals carry random sign strings, which a sequence model cannot help and may hurt; real inscriptions "
              "behave like the corpus-text rows."]
    if "size" in rs:
        z = rs["size"]
        lines += ["", f"**Sign size.** Penalising candidates whose drawn height does not fit the box (weight {z['weight']}, chosen on val seals) "
                  f"changes the sign error rate on all test seals from {fmt(z['top1_sign_error_rate'])} to {fmt(z['size_adjusted_sign_error_rate'])}."]
    return lines


def write_report(results, out_dir):
    dets = list(results["detectors"])
    lines = [
        "# Milestone 1 report: automated visual grapheme extraction",
        "",
        "**Every number below is measured on synthetic seals.** The test split is generated by the same "
        "program as the training data, so these scores show that the pipeline works on its own training "
        "distribution. They do not show that it works on real seals; see \"Real photographs\" below.",
        "",
        f"Test split: {results['seals']} synthetic seals, {results['signs']} signs, never used for training, "
        "model selection or temperature tuning. Brackets are 95% bootstrap intervals over seals (1,000 resamples). "
        "No cross-validation was run: the data is generated, so a fresh held-out split replaces it.",
        "",
        "## Detector comparison (end to end, same classifier)",
        "",
        "| Metric | " + " | ".join(dets) + " |",
        "|---|" + "---|" * len(dets),
    ]
    for key in ("AP50", "AP50-95"):
        lines.append(f"| {key} (single class) | " + " | ".join(
            f"{results['detectors'][d]['ap'][key]:.3f}" if results["detectors"][d].get("ap") else "n/a" for d in dets) + " |")
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(fmt(results["detectors"][d]["metrics"][m]) for d in dets) + " |")
    lines.append("| Milliseconds per image, detector + classifier | " + " | ".join(f"{results['detectors'][d]['ms_per_image']:.0f}" for d in dets) + " |")
    lines += [
        "",
        f"Timing device: `{results['device']}`. Detections are kept at confidence 0.25 for every learned detector. "
        "\"Touching signs\" are ground-truth boxes that overlap a neighbour; this is the case the original PRD "
        "expected NMS to break.",
        "",
        "## Classifier alone (ground-truth boxes, no detector)",
        "",
        "| Signs, by frequency in the IM77 corpus | Test crops | Top-1 | Top-5 |",
        "|---|---|---|---|",
    ]
    for name, row in results["classifier"].items():
        lines.append(f"| {name} | {row['crops']} | {fmt(row['top1'])} | {fmt(row['top5'])} |")
    lines += [
        "",
        "Intervals here resample crops. The reference bank and the test crops are both augmentations of the same "
        "417 source drawings (IM-417 has one drawing per sign), so this measures robustness to the generator's "
        "distortions, not recognition of real carved signs. Per-sign precision, recall and F1 are in `per_class_f1.csv`.",
    ]
    if results.get("rescoring"):
        lines += rescoring_lines(results["rescoring"])
    lines += [
        "",
        "## Real photographs",
        "",
        "`figures/` holds each learned detector's output on the real seal photographs in `data/samples/`. There are "
        "no labelled real seals, so nothing here is scored. On the Mac run that preceded this report, a "
        "synthetic-trained YOLOv8n found the inscription area but merged neighbouring signs into one box and boxed "
        "the animal motif. Compare the figures from this run against that before relying on any detector.",
        "",
        "## What is and is not delivered",
        "",
        "- Delivered: synthetic generator, YOLOv8n and RT-DETR detectors, OpenCV baseline, DINOv2 nearest-neighbour "
        "sign classifier, image-to-sign-ID pipeline, this evaluation, training curves (`*_results.csv`, `*_results.png`).",
        "- Not done: any measurement on real seals (needs labelled real photographs); Faster R-CNN (dropped: no "
        "expected gain over the two detectors compared here).",
        "",
    ]
    (out_dir / "milestone_1_report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    same = [Counter(a=3, b=4)] * 20
    assert bootstrap([r["a"] for r in same], [r["b"] for r in same]) == [0.75, 0.75, 0.75]
    lo, hi = bootstrap([1, 0] * 50, [1] * 100)[1:]
    assert lo < 0.5 < hi and hi - lo < 0.3
    assert levenshtein([342, 99], [342]) == 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=["yolov8n", "rtdetr-l"], help="run folders under runs/")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "reports/milestone_1"))
    ap.add_argument("--rescoring-only", action="store_true",
                    help="only the n-gram re-ranking check (YOLOv8n): writes rescoring.json and rescoring.md, keeps the main report")
    args = ap.parse_args()
    if args.rescoring_only:
        rs = rescoring(load_detector(ROOT / "runs/yolov8n/weights/best.pt"), load_classifier())
        Path(args.out, "rescoring.json").write_text(json.dumps(rs, indent=1))
        Path(args.out, "rescoring.md").write_text("\n".join(["# Milestone 1: re-ranking with a sign-sequence model", "",
            "Synthetic test seals regenerated with corpus texts disjoint across train, val and test (`synthetic_generator.split_of`); "
            "the detector and classifier bank are the ones from the main report.", *rescoring_lines(rs)[2:]]) + "\n")
        print(json.dumps({k: rs[k] for k in ("weight", "all")}, indent=1))
        raise SystemExit
    out_dir = Path(args.out)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    classifier = load_classifier()
    results = {"split": args.split, "device": DEVICE, "detectors": {}}
    samples = sorted((ROOT / "data/samples").glob("*.jpg"))
    for name in [*args.runs, "opencv"]:
        weights = ROOT / "runs" / name / "weights/best.pt"
        if name != "opencv" and not weights.exists():
            print(f"skipping {name}: {weights} not found")
            continue
        detect = load_detector("opencv" if name == "opencv" else weights)
        records, ms = seal_records(detect, classifier, args.split)
        entry = {"metrics": summarise(records), "ms_per_image": ms}
        if name != "opencv":
            entry["ap"] = detector_ap(weights, args.split, name)
            for f in ("results.csv", "results.png"):  # training curves, kept with the report
                if (weights.parents[1] / f).exists():
                    shutil.copy(weights.parents[1] / f, out_dir / f"{name}_{f}")
        for f in samples:  # real photographs: qualitative only, nothing to score against
            img = cv2.imread(str(f))
            cv2.imwrite(str(out_dir / "figures" / f"{f.stem}_{name}.jpg"), annotate(img, extract(img, detect, classifier)))
        results["detectors"][name] = entry
        results["seals"], results["signs"] = len(records), sum(r["n_gt"] for r in records)
        print(name, {k: round(v[0], 3) for k, v in entry["metrics"].items()})
    results["classifier"] = classifier_report(classifier, args.split, out_dir)
    if "yolov8n" in results["detectors"] and args.split == "test":
        results["rescoring"] = rescoring(load_detector(ROOT / "runs/yolov8n/weights/best.pt"), classifier)
        print("rescoring:", {k: v for k, v in results["rescoring"].items() if k in ("weight", "all")})
    (out_dir / "eval_results.json").write_text(json.dumps(results, indent=1))
    write_report(results, out_dir)
    print(f"report -> {out_dir / 'milestone_1_report.md'}")
