# Run info

- Machine: Linux, 2x NVIDIA GeForce RTX 5090 (32 GB each); run used one GPU
- Software: Python 3.10.21, torch 2.14.1+cu130 (CUDA 13.0), ultralytics 8.4.171
- Branch: `milestone-1-pipeline` at commit e0a9b50, default settings via `python grapheme_extraction/run_all.py`
- Pretrained weights `yolov8n.pt` and `rtdetr-l.pt` downloaded fine (not trained from scratch)
- `pipeline.py --selftest` printed `ok`

| Step | Time |
|---|---|
| 4. Synthetic generation (4000 / 400 / 400) | ~1 min |
| 5. YOLOv8n, 30 epochs, batch 32 | ~1.7 min (0.029 h) |
| 6. RT-DETR-L, 60 epochs, batch 8 (no early stop) | ~39.5 min (0.658 h), ~40 s/epoch, ~6.5 GB GPU memory |
| 7. Classifier bank | ~16 s |
| 8. Evaluation + report | ~20 s |

`reports/quick/` is from the `--quick` install check that ran first, so ignore its numbers. `full_run.log` is the full console output.
