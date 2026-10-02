"""Give hand-corrected boxes their sign IDs from the catalogue text, by position.

For a seal whose M77 text number is in the manifest, the sign sequence is already known, so
boxes (impression view) sorted into reading order take the corpus signs one for one. Any
mismatch in count, or an illegible sign in the corpus text, goes to manual review instead.

    python grapheme_extraction/assign_labels.py     # rewrites class ids in data/real/labels/*.txt
"""
import csv

from classifier import ROOT
from pipeline import group_into_lines
from synthetic_generator import IM77_CSV

REAL = ROOT / "data/real"


def corpus_texts():
    """text_no -> sign IDs in reading order, lines concatenated top to bottom (0 = illegible)."""
    # ponytail: assumes the CSV's row order is the physical line order; confirm on ~10 well-known
    # multi-line seals before trusting auto-labels for multi-line texts.
    texts = {}
    with open(IM77_CSV) as f:
        for row in csv.DictReader(f):
            texts.setdefault(row["text_no"], []).extend(int(t) for t in row["signs_clean"].split())
    return texts


def assign(boxes, gold):
    """boxes: [(x0, y0, x1, y1)] in impression view. Returns a sign ID per box, or None if unsafe."""
    if len(boxes) != len(gold) or 0 in gold:
        return None
    lines = group_into_lines(boxes)
    order = sorted(range(len(boxes)), key=lambda i: (lines[i], -(boxes[i][0] + boxes[i][2]) / 2))  # RTL
    ids = [0] * len(boxes)
    for sign, i in zip(gold, order):
        ids[i] = sign
    return ids


if __name__ == "__main__":
    row = [(10, 0, 30, 40), (100, 0, 120, 40), (50, 0, 70, 40)]
    assert assign(row, [342, 99, 59]) == [59, 342, 99]  # rightmost box takes the first sign
    assert assign(row, [342, 99]) is None and assign(row, [342, 0, 59]) is None

    texts, auto, review = corpus_texts(), 0, []
    with open(REAL / "manifest.csv") as f:
        for r in csv.DictReader(f):
            lbl = REAL / "labels" / f"{r['image_id']}.txt"
            if not r["m77_text_no"] or not lbl.exists():
                continue
            rows = [line.split() for line in lbl.read_text().splitlines() if line.strip()]
            boxes = [(float(x) - float(w) / 2, float(y) - float(h) / 2, float(x) + float(w) / 2, float(y) + float(h) / 2)
                     for _, x, y, w, h in rows]
            ids = assign(boxes, texts.get(r["m77_text_no"], []))
            if ids is None:
                review.append(r["image_id"])
                continue
            lbl.write_text("\n".join(" ".join([str(s - 1), *row[1:]]) for s, row in zip(ids, rows)))
            auto += 1
    (REAL / "needs_manual_ids.txt").write_text("\n".join(review))
    print(f"auto-labelled {auto} seals; {len(review)} need manual sign IDs (data/real/needs_manual_ids.txt). Spot-check 10% of the auto ones.")
