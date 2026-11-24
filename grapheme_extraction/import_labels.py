"""Hand-labelled real seals: Label Studio JSON export -> impression-view images, boxes and splits.

    python grapheme_extraction/import_labels.py export.json --package path/to/indus_hand_labelling

The package folder is the one handed to the labeller (images/ and sources.csv). Writes:
  data/real/images/<id>.jpg     impression view: photos of carved seals are mirrored
  data/real/annotations.csv     one row per box, in reading order, pixel coordinates of that image
  data/real/manifest.csv        one row per image, with its split; existing splits are kept (frozen test set)
  data/real/check/<id>.jpg      boxes and sign numbers drawn, for spot-checking
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from classifier import N_SIGNS, ROOT
from pipeline import group_into_lines

REAL = ROOT / "data/real"
SPLIT_SHARE = {"test": 0.4, "val": 0.15}  # rest train; test is frozen once written
MANIFEST = ["image_id", "file", "set", "source_url", "licence", "object_group", "view", "several_inscriptions", "n_signs", "n_unsure", "split", "notes"]


def file_name(image_field):
    """Label Studio prefixes uploads with 8 hex characters and a dash: '/data/upload/1/654d8bd7-wiki_064.jpg' -> 'wiki_064.jpg'."""
    return re.sub(r"^[0-9a-f]{8}-", "", image_field.rsplit("/", 1)[-1])


def parse(result):
    """One annotation's result list -> (view, several, notes, [(sign or None, unsure, x, y, w, h) in percent])."""
    boxes, sign, unsure, view, several, notes = {}, {}, set(), None, False, ""
    for r in result:
        v, name = r["value"], r["from_name"]
        if name == "box":
            boxes[r["id"]] = (v["x"], v["y"], v["width"], v["height"])
        elif name == "sign_id":
            sign[r["id"]] = " ".join(v["text"]).strip()
        elif name == "certainty":
            unsure.add(r["id"])
        elif name == "view":
            view = v["choices"][0]
        elif name == "objects":
            several = True
        elif name == "notes":
            notes = " ".join(v["text"]).strip()
    return view, several, notes, [(sign.get(i), i in unsure, *b) for i, b in boxes.items()]


def sign_number(text):
    """'342' -> 342; anything else (missing, '342?', '1000') -> None."""
    return int(text) if text and text.isdigit() and 0 <= int(text) <= N_SIGNS else None


def to_pixels(box, w, h, mirror):
    """Percent box -> (x0, y0, x1, y1) pixels, mirrored horizontally for a photo of the carved seal."""
    x, y, bw, bh = box
    if mirror:
        x = 100 - x - bw
    return round(x * w / 100), round(y * h / 100), round((x + bw) * w / 100), round((y + bh) * h / 100)


def dhash(img):
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()


def assign_splits(rows, images, old):
    """Photos of one object share a split: the package's object_group column first, then near-identical
    framing by dHash. Photos with several inscriptions only train, since a whole-image sequence would mix
    inscriptions. Non-empty splits already in the manifest never change."""
    # ponytail: object_group was filled by eye for the shipped package; an unmarked repeat photo of one
    # seal can still cross splits. Extend the column if the labeller notes catalogue numbers.
    by_row = {r["image_id"]: r for r in rows}
    ids = [r["image_id"] for r in rows if r["view"] not in ("", "skip")]
    hashes = {i: dhash(images[i]) for i in ids}
    group, named = {}, {}
    for a_i, a in enumerate(ids):
        name = by_row[a]["object_group"]
        group[a] = named.setdefault(name, a) if name else a
        for b in ids[:a_i]:
            if group[a] == a and (hashes[a] != hashes[b]).sum() <= 8:
                group[a] = group[b]
                break
    members = defaultdict(list)
    for i in ids:
        members[group[i]].append(i)
    fixed = {g: old[m] for g, ms in members.items() for m in ms if old.get(m)}
    free = [g for g in sorted(members) if g not in fixed]
    random.Random(0).shuffle(free)
    target = {k: round(share * len(free)) for k, share in SPLIT_SHARE.items()}
    split = dict(fixed)
    for g in free:
        if any(by_row[m]["several_inscriptions"] for m in members[g]):
            split[g] = "train"
        elif target["test"] > 0:
            split[g], target["test"] = "test", target["test"] - 1
        elif target["val"] > 0:
            split[g], target["val"] = "val", target["val"] - 1
        else:
            split[g] = "train"
    for r in rows:
        r["split"] = split[group[r["image_id"]]] if r["image_id"] in group else ""


def draw(img, boxes):
    img = img.copy()
    for b in boxes:
        x0, y0, x1, y1 = b["x0"], b["y0"], b["x1"], b["y1"]
        colour = (0, 160, 255) if b["unsure"] else (60, 170, 60)
        cv2.rectangle(img, (x0, y0), (x1, y1), colour, 2)
        cv2.putText(img, f"{b['order'] + 1}:{b['sign_id']}", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
    return img


def selfcheck():
    assert file_name("/data/upload/1/654d8bd7-wiki_064.jpg") == "wiki_064.jpg" and file_name("/data/local-files/?d=images/a.jpg") == "a.jpg"
    assert [sign_number(t) for t in ("342", "0", " 7", "342?", "418", None)] == [342, 0, None, None, None, None]
    assert to_pixels((10, 20, 30, 40), 200, 100, False) == (20, 20, 80, 60)
    assert to_pixels((10, 20, 30, 40), 200, 100, True) == (120, 20, 180, 60)  # left edge 10% -> right edge 90%
    result = [{"id": "a", "from_name": "box", "value": {"x": 1, "y": 2, "width": 3, "height": 4, "rectanglelabels": ["sign"]}},
              {"id": "a", "from_name": "sign_id", "value": {"text": ["342"]}},
              {"id": "a", "from_name": "certainty", "value": {"choices": ["unsure"]}},
              {"id": "b", "from_name": "box", "value": {"x": 5, "y": 6, "width": 7, "height": 8, "rectanglelabels": ["sign"]}},
              {"id": "v", "from_name": "view", "value": {"choices": ["carved seal"]}}]
    assert parse(result) == ("carved seal", False, "", [("342", True, 1, 2, 3, 4), (None, False, 5, 6, 7, 8)])


if __name__ == "__main__":
    selfcheck()
    ap = argparse.ArgumentParser()
    ap.add_argument("export", help="Label Studio export, JSON format")
    ap.add_argument("--package", required=True, help="the labelling package folder (contains images/ and sources.csv)")
    args = ap.parse_args()
    package = Path(args.package)
    sources = {r["file"]: r for r in csv.DictReader(open(package / "sources.csv"))}
    old = {r["image_id"]: r["split"] for r in csv.DictReader(open(REAL / "manifest.csv"))} if (REAL / "manifest.csv").exists() else {}
    for d in ("images", "check"):
        (REAL / d).mkdir(parents=True, exist_ok=True)

    rows, boxes_out, images, problems = [], [], {}, []
    for task in json.load(open(args.export)):
        done = [a for a in task["annotations"] if not a.get("was_cancelled")]
        if not done:
            continue
        name = file_name(task["data"]["image"])
        view, several, notes, boxes = parse(max(done, key=lambda a: a.get("updated_at", ""))["result"])
        image_id = Path(name).stem
        src = sources.get(name, {})
        row = {"image_id": image_id, "file": name, "set": src.get("set", ""), "source_url": src.get("source_url", ""),
               "licence": src.get("licence", ""), "object_group": src.get("object_group", ""), "view": view or "", "several_inscriptions": int(several), "n_signs": 0,
               "n_unsure": 0, "split": "", "notes": notes}
        rows.append(row)
        if view in (None, "skip"):
            if view is None:
                problems.append(f"{name}: no view chosen")
            continue
        img = cv2.imread(str(package / "images" / name))
        if img is None:
            problems.append(f"{name}: image not found in {package / 'images'}")
            row["view"] = "skip"
            continue
        mirror = view == "carved seal"
        img = cv2.flip(img, 1) if mirror else img
        images[image_id] = img
        h, w = img.shape[:2]
        mine = []
        for text, unsure, *box in boxes:
            s = sign_number(text)
            if s is None:
                problems.append(f"{name}: a box has " + (f"sign number {text!r}" if text else "no sign number") + "; use 0 to 417")
                continue
            x0, y0, x1, y1 = to_pixels(box, w, h, mirror)
            mine.append({"image_id": image_id, "sign_id": s, "unsure": int(unsure), "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        if not mine:
            problems.append(f"{name}: marked {view} but has no usable boxes")
        lines = group_into_lines([(b["x0"], b["y0"], b["x1"], b["y1"]) for b in mine])
        for b, line in zip(mine, lines):
            b["line"] = line
        mine.sort(key=lambda b: (b["line"], -(b["x0"] + b["x1"]) / 2))  # lines top to bottom, right to left
        for i, b in enumerate(mine):
            b["order"] = i
        row["n_signs"], row["n_unsure"] = len(mine), sum(b["unsure"] for b in mine)
        boxes_out += mine
        cv2.imwrite(str(REAL / "images" / f"{image_id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(REAL / "check" / f"{image_id}.jpg"), draw(img, mine))

    assign_splits(rows, images, old)
    with open(REAL / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r["image_id"]))
    with open(REAL / "annotations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "order", "line", "sign_id", "unsure", "x0", "y0", "x1", "y1"])
        w.writeheader()
        w.writerows(boxes_out)

    used = [r for r in rows if r["view"] not in ("", "skip")]
    print(f"{len(rows)} images labelled: {len(used)} used, {len(rows) - len(used)} skipped")
    print(f"{len(boxes_out)} sign boxes, {sum(b['unsure'] for b in boxes_out)} marked unsure, "
          f"{sum(b['sign_id'] == 0 for b in boxes_out)} illegible, {len({b['sign_id'] for b in boxes_out} - {0})} distinct signs")
    print("splits (images):", dict(Counter(r["split"] for r in used)))
    print("most common signs:", Counter(b["sign_id"] for b in boxes_out).most_common(8))
    (REAL / "import_problems.txt").write_text("\n".join(problems))
    print(f"{len(problems)} problems -> data/real/import_problems.txt" if problems else "no problems found")
    print("spot-check the drawings in data/real/check/")
