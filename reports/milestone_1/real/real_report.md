# Milestone 1 on real seal photographs

**First measurement on real photographs.** 31 photographs were hand-labelled (159 sign boxes, 32 marked illegible) and split by photograph, with photos of the same object kept together: 14 train, 5 val, 12 test. The test photos were never used for training, bank building or model selection. With only 68 test signs on 12 photos the intervals are wide; read them before the point estimates.

| Measure (real test photos) | Synthetic-only models | Detector fine-tuned and bank extended on real photos |
|---|---|---|
| Detector precision (IoU 0.5) | 0.655 [0.522, 0.774] | 0.690 [0.565, 0.816] |
| Detector recall (IoU 0.5) | 0.529 [0.434, 0.615] | 0.588 [0.448, 0.714] |
| Classifier top-1 on true boxes | 0.276 [0.163, 0.390] | 0.241 [0.132, 0.370] |
| Classifier top-5 on true boxes | 0.466 [0.300, 0.615] | 0.483 [0.327, 0.644] |
| Sign error rate, whole pipeline | 1.017 [0.838, 1.286] | 1.017 [0.886, 1.143] |
| Exact sequence match | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

- Detector: the synthetic YOLOv8n, and two fine-tunes from it on the 14 real training photos (alone, and mixed with 300 synthetic seals), flips off. The one carried into the right-hand column was chosen by F1 on the 5 val photos: **fine-tuned, mixed**.
- The extended bank adds 69 legible real sign crops (40 distinct signs, 8 jittered copies each) from the train and val photos to the synthetic bank; no retraining.
- Brackets are 95% bootstrap intervals over test photos. Illegible signs count for detection but are left out of the classifier and sequence measures.
- Sequences are read in impression view; photos of carved seals were mirrored on import.

| Detector | Val precision | Val recall | Test precision | Test recall |
|---|---|---|---|---|
| synthetic only | 0.550 [0.235, 0.783] | 0.440 [0.190, 0.652] | 0.655 [0.522, 0.774] | 0.529 [0.434, 0.615] |
| fine-tuned, real | 0.765 [0.583, 0.846] | 0.520 [0.238, 0.760] | 0.707 [0.500, 0.804] | 0.426 [0.196, 0.612] |
| fine-tuned, mixed | 0.692 [0.476, 0.857] | 0.720 [0.476, 0.897] | 0.690 [0.565, 0.816] | 0.588 [0.448, 0.714] |

`figures/` shows each test photo with the boxes and top-1 sign numbers of both systems (left: synthetic-only, right: with real).
