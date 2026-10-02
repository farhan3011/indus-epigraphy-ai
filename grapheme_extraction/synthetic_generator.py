"""Synthetic seal generator: IM-417 glyphs engraved on procedural stone, with YOLO labels for free.

Label class = Mahadevan sign ID - 1, so one label file serves the single-class detector
(train with single_cls=True), the sign classifier, and the reading-order ground truth
(signs sorted by descending x = right-to-left reading).

    python grapheme_extraction/synthetic_generator.py --train 4000 --val 400 --test 400
"""
import argparse
import csv
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GLYPH_DIR = ROOT / "data/raw/im417/IM_417_150_PNG"
# Export of Mahadevan 1977 from github.com/paragorth/Indus- (see data/README.md); not redistributed here.
IM77_CSV = ROOT / "data/raw/im77/im77_corpus_lines.csv"
OUT = ROOT / "data/synthetic"
SIZE = 512
N_SIGNS = 417
MAX_LEN = 8  # longer lines do not fit one register at a legible scale


def load_glyphs():
    """sign_id -> tight-cropped uint8 mask (IM-417 ships one 150x150 white-on-black image per sign)."""
    glyphs = {}
    for i in range(1, N_SIGNS + 1):
        m = cv2.imread(str(GLYPH_DIR / f"{i}.png"), cv2.IMREAD_GRAYSCALE)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero((m > 127).astype(np.uint8)))
        glyphs[i] = m[y:y + h, x:x + w]
    return glyphs


def load_sequences():
    """Real IM77 texts as Mahadevan sign IDs in reading order (illegible-sign marker 0 dropped)."""
    seqs = []
    with open(IM77_CSV) as f:
        for row in csv.DictReader(f):
            seq = [int(t) for t in row["signs_clean"].split() if t != "0"]
            if 1 <= len(seq) <= MAX_LEN:
                seqs.append(seq)
    return seqs


def stone_texture():
    # ponytail: procedural noise stands in for the PRD's 30-50 photographed substrates;
    # replace with real steatite/terracotta/copper plates when a texture bank exists.
    base = np.float32(random.choice([(200, 215, 225), (90, 120, 180), (150, 150, 150), (110, 140, 120)]))  # BGR
    tex = np.zeros((SIZE, SIZE), np.float32)
    for s in (4, 16, 64):
        noise = np.random.randn(SIZE // s + 1, SIZE // s + 1).astype(np.float32)
        tex += cv2.resize(noise, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC) * s ** 0.5
    ramp = np.linspace(-1, 1, SIZE, dtype=np.float32)
    a = random.uniform(0, 2 * np.pi)
    light = np.cos(a) * ramp[None, :] + np.sin(a) * ramp[:, None]  # uneven raking light
    return base + (tex / tex.std() * random.uniform(5, 16) + light * random.uniform(0, 30))[..., None]


def engrave(img, mask, depth):
    """Intaglio relief: darken (depth>0, seal) or lighten (depth<0, impression) under the mask,
    with an offset rim highlight so strokes read as carved rather than painted."""
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255, (0, 0), random.uniform(0.6, 1.4))
    rim = np.clip(np.roll(m, (2, 2), axis=(0, 1)) - m, 0, 1)
    return img * (1 - depth * m[..., None]) + 45 * abs(depth) * rim[..., None]


def vary_glyph(mask, scale):
    """Scale, rotate +-6 deg and randomly thin/thicken strokes (chisel width, wear)."""
    h, w = mask.shape
    mask = cv2.resize(mask, (max(2, round(w * scale)), max(2, round(h * scale))), interpolation=cv2.INTER_AREA)
    mask = cv2.copyMakeBorder(mask, 12, 12, 12, 12, cv2.BORDER_CONSTANT)
    h, w = mask.shape
    mask = cv2.warpAffine(mask, cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-6, 6), 1), (w, h))
    op = random.choice([None, cv2.erode, cv2.dilate, cv2.dilate])
    if op and not (op is cv2.erode and scale < 1.2):  # eroding small glyphs deletes them
        mask = op(mask, np.ones((2, 2), np.uint8))
    x, y, w, h = cv2.boundingRect(cv2.findNonZero((mask > 60).astype(np.uint8)))
    return mask[y:y + h, x:x + w]


def motif(top):
    """Unlabelled carving below the text (stand-in for the animal motif) so the detector
    learns that not every engraved shape is a sign."""
    m = np.zeros((SIZE, SIZE), np.uint8)
    if top < SIZE - 80:
        for _ in range(random.randint(1, 4)):
            c = (random.randint(60, SIZE - 60), random.randint(top + 40, SIZE - 30))
            axes = (random.randint(40, 150), random.randint(20, max(21, (SIZE - top) // 3)))
            cv2.ellipse(m, c, axes, random.uniform(0, 180), 0, random.uniform(120, 360), 255, random.randint(3, 9))
        m[:top] = 0
    return m


def make_seal(glyphs, seqs, degrade):
    """Returns (BGR uint8 image, [(sign_id, x0, y0, x1, y1), ...])."""
    # Half real texts (real sign frequencies and neighbours), half uniform random signs of a real
    # length: the corpus is Zipfian (112 signs occur once), and the classifier needs every sign.
    ids = random.choice(seqs)[::-1]  # reading order is right-to-left, layout below is left-to-right
    if random.random() < 0.5:
        ids = random.choices(range(1, N_SIGNS + 1), k=len(ids))
    # One scale per seal (+-8% per sign): relative sign size is the only thing separating
    # e.g. short strokes (99) from long strokes (87), so it must survive compositing.
    scale = min(random.uniform(0.9, 1.9), (SIZE - 70) / sum(glyphs[i].shape[1] + 10 for i in ids))
    masks = [vary_glyph(glyphs[i], scale * random.uniform(0.92, 1.08)) for i in ids]
    gaps = [max(-2, round(random.gauss(6, 2) * scale)) for _ in masks]  # negative = touching signs
    total = sum(m.shape[1] for m in masks) + sum(gaps[1:])
    line_h = max(m.shape[0] for m in masks)
    x = random.randint(8, max(8, SIZE - 8 - total))
    yc = random.randint(line_h // 2 + 8, max(line_h // 2 + 8, int(SIZE * 0.45)))

    canvas, boxes = np.zeros((SIZE, SIZE), np.uint8), []
    for sign, m, gap in zip(ids, masks, gaps):
        h, w = m.shape
        y = yc - h // 2 + random.randint(-3, 3)
        if x < 0 or y < 0 or x + w > SIZE or y + h > SIZE:  # check BEFORE drawing: no unlabelled signs
            break
        canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], m)
        boxes.append((sign, max(0, x - 2), max(0, y - 2), min(SIZE, x + w + 2), min(SIZE, y + h + 2)))
        x += w + gap

    stone = stone_texture()
    depth = random.uniform(0.3, 0.75) * random.choice([1, 1, -0.6])
    if stone.mean() > 150:  # lightening an already pale stone saturates the signs away
        depth = abs(depth)
    img = engrave(stone, canvas, depth)
    img = engrave(img, motif(yc + line_h // 2 + 10), depth)
    img = np.clip(img, 0, 255).astype(np.uint8)

    out = degrade(image=img, bboxes=[b[1:] for b in boxes], signs=[b[0] for b in boxes])
    return out["image"], [(int(s), *b) for s, b in zip(out["signs"], out["bboxes"])]


def degradation():
    return A.Compose([
        A.Perspective(scale=(0.01, 0.04), border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.GaussianBlur(sigma_limit=(0.4, 1.0), p=0.4),
        A.MotionBlur(blur_limit=5, p=0.15),
        A.GaussNoise(std_range=(0.01, 0.06), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.35, p=0.8),
        A.ToGray(p=0.2),
        A.ImageCompression(quality_range=(40, 90), p=0.6),
    ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=["signs"], min_visibility=0.6))


def generate(n, split, glyphs, seqs):
    img_dir, lbl_dir = OUT / "images" / split, OUT / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    degrade = degradation()
    for i in range(n):
        img, boxes = make_seal(glyphs, seqs, degrade)
        assert all(0 <= x0 < x1 <= SIZE and 0 <= y0 < y1 <= SIZE for _, x0, y0, x1, y1 in boxes), boxes
        cv2.imwrite(str(img_dir / f"{split}_{i:05d}.jpg"), img)
        (lbl_dir / f"{split}_{i:05d}.txt").write_text("\n".join(
            f"{s - 1} {(x0 + x1) / 2 / SIZE:.6f} {(y0 + y1) / 2 / SIZE:.6f} {(x1 - x0) / SIZE:.6f} {(y1 - y0) / SIZE:.6f}"
            for s, x0, y0, x1, y1 in boxes))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--test", type=int, default=300, help="held out: never used for model selection or tuning")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    glyphs, seqs = load_glyphs(), load_sequences()
    generate(args.train, "train", glyphs, seqs)
    generate(args.val, "val", glyphs, seqs)
    generate(args.test, "test", glyphs, seqs)
    names = "\n".join(f"  {i}: '{i + 1}'" for i in range(N_SIGNS))
    (OUT / "data.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")
    print(f"wrote {args.train} train + {args.val} val + {args.test} test seals to {OUT}")
