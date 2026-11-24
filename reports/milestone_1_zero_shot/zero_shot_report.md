# Zero-shot sign detection with pretrained segmenters (exploratory)

Can a segmenter trained on everyday photographs isolate Indus signs with no training on them? FastSAM and SAM ViT-B produce masks; each mask becomes a box, and a fixed rule keeps the sign-like ones (width 2-25% of the image, aspect 0.2-5, centre in the upper half, not a box around three or more kept boxes). The synthetic-trained YOLOv8n is the reference. Boxes count as found at IoU 0.5.

| Method | Synthetic seals: precision | recall | Real test photos: precision | recall |
|---|---|---|---|---|
| YOLOv8n (synthetic-trained) | 1.00 | 1.00 | 0.65 | 0.53 |
| FastSAM | 0.73 | 0.78 | 0.55 | 0.41 |
| SAM ViT-B | 0.36 | 0.48 | 0.12 | 0.18 |

Synthetic: the first 100 test seals for FastSAM and YOLOv8n, the first 30 for SAM (it takes about 25 seconds a photo on the CPU). Real: the 12 hand-labelled test photographs (68 signs).

## Reading

- On the real test photographs the best zero-shot method, FastSAM, finds 41% of the signs, against 53% for the synthetic-trained YOLOv8n.
- That is below the 70% bar set in advance, so zero-shot segmentation does not solve sign detection here and the labelling helpers are left as they are.

## Not run

YOLO-World (open-vocabulary detection with prompts such as "symbol" or "carved sign") ships in the pinned ultralytics, but its weights download from GitHub release assets, which this network blocks, and no official mirror was found.

## Figures

`figures/` holds the four Met photographs with each method's kept boxes (red) and, for labelled photos, the hand-drawn boxes (green). The real test photographs are drawn locally but not published: some are research-use-only images.

Settings: run time 18.7 minutes, quick run: False.
