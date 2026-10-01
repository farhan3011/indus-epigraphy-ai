# Milestone 1: Automated Visual Grapheme Extraction — PRD

**Project:** Computational Epigraphy of the Indus Valley Script
**Milestone:** 1 of 4
**Status:** Planning / pre-implementation

---

## 1. Objective

Parse raw Indus seal artifact images to extract, crop, and categorize individual script symbols (graphemes), producing a mapping from raw seal photographs to an ordered list of standardized numerical sign IDs (per Mahadevan's 417-sign concordance).

This milestone is a prerequisite for all downstream milestones. Milestone 2 (Markov chain / positional bias modeling) depends entirely on receiving a clean, ordered sequence of sign IDs per seal.

## 2. Background & Constraints

- The Indus script is undeciphered; this project treats it as a structured notational system, not an attempt at translation.
- Average inscription length is ~5 signs per artifact — short sequences, low redundancy, high sensitivity to extraction errors.
- The task decomposes into two distinct sub-problems that the source proposal treats as one:
  1. **Detection** — locating where each sign sits on a full seal photo (bounding boxes).
  2. **Classification** — mapping a cropped sign image to one of 417 standardized sign IDs.
- **Key data constraint:** the primary available dataset (IM-417) contains pre-segmented, individually cropped sign images mapped to sign IDs. It does **not** contain bounding-box-annotated full seal photographs. This makes the detection sub-problem non-trivial to solve with standard supervised object detection.
- Published prior work (Dixit et al., 2025, *Journal of Computer Applications in Archaeology*; Palaniappan & Adhikari, 2017) confirms this is a known, real constraint in the field — both had to manually annotate small custom sets (~245 and ~350 images respectively) because no larger public bbox-annotated Indus dataset exists.

## 3. Proposed Technical Approach

### 3.1 Two-track strategy

**Track A — Classifier (primary, well-supported by available data)**
Train a CNN (MobileNetV2, fine-tuned, or a small custom CNN if class counts are very low) on the IM-417 dataset to classify a cropped sign image into one of ~417 sign ID classes.

- Expect long-tailed class distribution (some signs occur far more often in the real corpus than others).
- Mitigate with class-weighted loss and augmentation (rotation, slight warping/erosion to simulate weathering, noise injection).
- Hold out a validation split from IM-417; report top-1 **and** top-5 accuracy (visually similar signs are a known confusion source).

**Track B — Detector (the constrained part)**
Since no large bbox-labeled dataset of full seal photos exists, generate one synthetically:

1. Take individual sign crops from IM-417.
2. Composite 2–8 crops (matching realistic ~5-sign inscription lengths) onto background images at randomized positions, scales, and rotations, with a plausible spacing/line arrangement.
3. Since crop placement is controlled programmatically, ground-truth bounding boxes are known automatically — no manual annotation required.
4. Train a lightweight single-class detector (YOLO-style head, or a simpler anchor-based CNN regressing box coordinates + objectness) on this synthetic set.
5. Validate against any small real annotated set that can be obtained (e.g. checking whether the DM-BiCLab GitHub repo — see References — includes usable XML bounding-box annotations from their 245-image manually annotated set) to sanity-check real-world generalization, acknowledging the synthetic-to-real domain gap explicitly in the report.

**Fallback / comparison baseline:** classical CV segmentation (OpenCV contour detection + morphological operations) as a non-deep-learning region proposal method, kept as a documented point of comparison against the trained detector — not as the primary detection mechanism, since the milestone spec calls for a deep learning detection backbone.

### 3.2 Pipeline (end to end)

```
Full seal photo
   -> [Track B: trained detector] -> candidate bounding boxes
   -> crop each box
   -> [Track A: classifier] -> sign ID per crop
   -> order crops spatially (per Milestone 2's directionality findings, or top-to-bottom/right-to-left default)
   -> ordered list of sign IDs (Milestone 1 deliverable)
```

### 3.3 Risk decoupling

If the detector (Track B) underperforms by the internal checkpoint (see Timeline), fall back to using clean sign sequences directly from the IM77/CISI corpora for Milestones 2–4, and report the classifier + synthetic detector as a standalone, separately-evaluated component rather than blocking the rest of the semester on it.

## 4. Datasets

| Dataset | Use |
|---|---|
| IM-417-150 Grapheme Dataset (Kaggle) | Classifier training (Track A); source crops for synthetic compositing (Track B) |
| Indus Script IM77 Corpus (RMRL) | Ground-truth sign sequences for validation / fallback |
| Digital Corpus of Indus Valley Script (CISI, Parpola et al.) | Ground-truth sign sequences for validation / fallback |
| Small manually-annotated bbox sets from published research (if accessible) | Real-world validation set for the synthetic-trained detector |

## 5. Architecture Choices & Rationale

- **MobileNetV2** for classification: lightweight, well-suited to a moderate-sized dataset, transfer learning reduces data requirements.
- **Synthetic compositing** for detection training data: standard technique for low-resource object detection when only segmentation-level source data exists; produces exact ground-truth boxes without manual labeling cost.
- **OpenCV segmentation** retained only as a documented baseline/comparison, not the primary detector, to satisfy the "deep learning backbone" requirement in the milestone spec.

## 6. Deliverables

1. Trained classifier (Track A) with reported top-1/top-5 accuracy and confusion matrix.
2. Trained detector (Track B) with reported precision/recall on held-out synthetic data and (if available) a small real-image validation set.
3. End-to-end image pipeline script: input = seal photograph, output = ordered list of standardized sign IDs.
4. Written comparison: synthetic-trained detector vs. OpenCV segmentation baseline.
5. Documented training notebooks (learning curves, cross-validation results).

## 7. Evaluation Plan

- Classifier: top-1 and top-5 accuracy, per-class F1 (to surface long-tail underperformance), confusion matrix.
- Detector: IoU-based precision/recall on synthetic held-out set; qualitative + quantitative check against any real annotated images obtained.
- End-to-end: sign-sequence accuracy against known IM77/CISI ground truth for the same seals, where overlap exists.

## 8. Timeline (within overall 3-month project schedule)

- Weeks 1–2: Data setup, environment, synthetic compositing pipeline built.
- Weeks 3–4: Classifier (Track A) trained and validated.
- Week 5: Detector (Track B) trained on synthetic data; internal checkpoint — decide whether to proceed with Track B or fall back to corpus-direct sign sequences for later milestones.
- Week 5 (end): Pipeline integration (Track A + Track B chained), initial end-to-end evaluation.

## 9. References

- Dixit, V., Hussain, N., Basak, S., Atturu, D., Mitra, D., Bhattacharya, U. (2025). *Deep Learning in Archiving Indus Script and Motif Information*. Journal of Computer Applications in Archaeology, 8(1), 156–169. Code/data: github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information
- Palaniappan, S., Adhikari, R. (2017). *Deep Learning Indus Script*. arXiv:1702.00523.
- tpsatish95/OCR-on-Indus-Seals (GitHub) — crawled seal dataset + region-proposal text/no-text classification.

---

### ⚠️ Open items to confirm with the professor before finalizing this approach

*\*Whether a synthetic-composite-trained detector (using programmatically generated bounding boxes) satisfies the milestone's requirement for a "deep learning vision backbone to detect bounding boxes," given that no public dataset of bbox-annotated full seal photographs exists at scale.\**

*\*Whether classical CV segmentation may be submitted as a documented comparison baseline alongside the deep learning detector, or whether it must be excluded entirely from the graded deliverable.\**

*\*Whether obtaining and using the small (~245-image) manually-annotated bounding box set referenced in published research (if accessible via their GitHub repo) for validation purposes is acceptable, or if only self-generated/self-annotated data should be used.\**