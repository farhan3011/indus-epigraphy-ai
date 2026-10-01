"""Draft sign boxes for hand correction: manifest -> Label Studio pre-annotation tasks.

Images are written to data/real/images/ in impression view (seal photos mirrored), so every
label made afterwards is in the same view as the corpus. Drafts come from the current detector;
retrain it on the first ~50 corrected seals and rerun for better drafts.

    python grapheme_extraction/bootstrap_boxes.py            # -> data/real/label_studio_tasks.json
    Label Studio: import that file, labelling config = one RectangleLabels "indus_sign",
    export as YOLO into data/real/labels/.
"""
import csv
import json

import cv2
from ultralytics import YOLO

from classifier import ROOT
from pipeline import DETECTOR, detect_yolo

REAL = ROOT / "data/real"


def to_task(image_url, boxes, w, h):
    """Label Studio task with boxes as percentages of the image size."""
    result = [{"type": "rectanglelabels", "from_name": "label", "to_name": "image",
               "original_width": w, "original_height": h,
               "value": {"x": 100 * x0 / w, "y": 100 * y0 / h, "width": 100 * (x1 - x0) / w,
                         "height": 100 * (y1 - y0) / h, "rectanglelabels": ["indus_sign"]}}
              for x0, y0, x1, y1, _ in boxes]
    return {"data": {"image": image_url}, "predictions": [{"model_version": "draft", "result": result}]}


if __name__ == "__main__":
    t = to_task("u", [(10, 20, 60, 120, 0.9)], 200, 400)["predictions"][0]["result"][0]["value"]
    assert (t["x"], t["y"], t["width"], t["height"]) == (5, 5, 25, 25)

    model = YOLO(str(DETECTOR))
    (REAL / "images").mkdir(parents=True, exist_ok=True)
    tasks = []
    with open(REAL / "manifest.csv") as f:
        for row in csv.DictReader(f):
            assert row["licence"], f"{row['image_id']}: no licence recorded, not allowed in the benchmark"
            img = cv2.imread(str(REAL / "source" / row["file"]))
            assert img is not None, f"cannot read data/real/source/{row['file']}"
            if row["is_seal_matrix"].strip().lower() in ("1", "true", "yes"):
                img = cv2.flip(img, 1)
            cv2.imwrite(str(REAL / "images" / f"{row['image_id']}.jpg"), img)
            h, w = img.shape[:2]
            # served by Label Studio local storage rooted at data/real/images
            tasks.append(to_task(f"/data/local-files/?d=images/{row['image_id']}.jpg", detect_yolo(model, img, conf=0.15), w, h))
    (REAL / "label_studio_tasks.json").write_text(json.dumps(tasks))
    print(f"{len(tasks)} tasks -> {REAL / 'label_studio_tasks.json'}")
