"""Train a single-class `indus_sign` detector on the synthetic seals.

    python grapheme_extraction/train_detector.py                       # YOLOv8n
    python grapheme_extraction/train_detector.py --model rtdetr-l.pt   # RT-DETR (NMS-free transformer)
"""
import argparse
from pathlib import Path

import torch
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "mps" if torch.backends.mps.is_available() else 0 if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained .pt, or a .yaml to train from scratch")
    ap.add_argument("--epochs", type=int, help="default 30 for YOLO, 60 for RT-DETR (transformers converge more slowly)")
    ap.add_argument("--batch", type=int, help="default 32 for YOLO, 8 for RT-DETR (much more GPU memory per image)")
    ap.add_argument("--name", help="run folder under runs/ (default: model file stem)")
    ap.add_argument("--device", default=DEVICE)
    args = ap.parse_args()

    rtdetr = "rtdetr" in args.model.lower()
    model = (RTDETR if rtdetr else YOLO)(args.model)
    # single_cls collapses the 417 sign-ID labels into one class: the detector only localises,
    # the classifier names the sign.
    model.train(data=str(ROOT / "data/synthetic/data.yaml"), single_cls=True, imgsz=512,
                epochs=args.epochs or (60 if rtdetr else 30), batch=args.batch or (8 if rtdetr else 32),
                patience=15, device=args.device,
                amp=torch.cuda.is_available(),  # off elsewhere: the AMP self-check downloads a model and MPS gains nothing
                fliplr=0.0, flipud=0.0,  # a mirrored sign is a different sign (or none): never flip
                project=str(ROOT / "runs"), name=args.name or Path(args.model).stem, exist_ok=True)
