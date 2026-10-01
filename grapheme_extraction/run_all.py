"""Run all of Milestone 1 in order, on synthetic data: generate, train both detectors, build the
classifier bank, evaluate, write the report.

    python grapheme_extraction/run_all.py            # full run (needs a GPU to be practical)
    python grapheme_extraction/run_all.py --quick    # minutes-long install check; OVERWRITES data/synthetic and models/bank.pt
    python grapheme_extraction/run_all.py --from 3   # resume at step 3 after a crash
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step to run (1-5)")
    ap.add_argument("--rtdetr-model", default="rtdetr-l.pt", help="use rtdetr-l.yaml if the pretrained weights cannot be downloaded")
    args = ap.parse_args()

    q = args.quick
    sizes = ["--train", "80", "--val", "20", "--test", "20"] if q else ["--train", "4000", "--val", "400", "--test", "400"]
    yolo, rtdetr = ("quick-yolov8n", "quick-rtdetr-l") if q else ("yolov8n", "rtdetr-l")
    quick_train = ["--epochs", "1", "--batch", "4"] if q else []
    steps = [
        ("1/5 generate synthetic seals", ["synthetic_generator.py", *sizes]),
        ("2/5 train YOLOv8n detector", ["train_detector.py", "--name", yolo, *quick_train]),
        ("3/5 train RT-DETR detector", ["train_detector.py", "--model", args.rtdetr_model, "--name", rtdetr, *quick_train]),
        ("4/5 build classifier bank", ["classifier.py"]),
        ("5/5 evaluate and write report", ["evaluate.py", "--runs", yolo, rtdetr, *(["--out", str(HERE.parent / "reports/quick")] if q else [])]),
    ]
    for i, (title, cmd) in enumerate(steps, 1):
        if i < args.start:
            continue
        print(f"\n=== {title} ===", flush=True)
        subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]], check=True, cwd=HERE.parent)
    print("\ndone: see reports/quick/milestone_1_report.md" if q else "\ndone: see reports/milestone_1_report.md")
