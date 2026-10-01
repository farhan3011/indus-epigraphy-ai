"""Train a single-class `indus_sign` detector on the synthetic seals.

    python grapheme_extraction/train_detector.py                       # YOLOv8n
    python grapheme_extraction/train_detector.py --model rtdetr-l.pt   # RT-DETR (NMS-free)
"""
import argparse
from pathlib import Path

import torch
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    model = (RTDETR if "rtdetr" in args.model else YOLO)(args.model)
    # single_cls collapses the 417 sign-ID labels into one class: the detector only localises,
    # the classifier names the sign (PRD section 4, Option A).
    model.train(data=str(ROOT / "data/synthetic/data.yaml"), single_cls=True, imgsz=512,
                epochs=args.epochs, batch=args.batch, device=DEVICE, amp=False,
                project=str(ROOT / "runs"), name=Path(args.model).stem, exist_ok=True)
