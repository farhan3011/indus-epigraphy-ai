# Milestone 1: decision and status (1 Oct 2026)

**Decision: follow the revised plan ("Milestone 1 Plan: Grapheme Extraction (Revised)"). Real labelled seals come first; synthetic data is an optional extra.** That plan is not in this repository yet. Until real labelled seals exist, Milestone 1 is completed on synthetic data only: see `docs/RUN_GUIDE.md` and `reports/milestone_1/milestone_1_report.md`.

## Why

A YOLOv8n detector trained only on 4,000 synthetic seals scores AP50 0.995 on synthetic validation and fails on real photographs. On three real Wikimedia/Met photos it boxed whole inscriptions, the animal motif, or an entire seal, and isolated only a few individual signs. Real signs are deep relief with cast shadows; the synthetic ones are thin engraved lines on noise. The classical OpenCV segmentation was worse still (it boxed animal bodies and seal edges).

## Results

Milestone 1 is complete on synthetic data. The full run (YOLOv8n, RT-DETR-L, OpenCV baseline, held-out synthetic test split) is in `reports/milestone_1/milestone_1_report.md`, with conclusions in `reports/milestone_1/milestone_1_findings.md`. In short: the two learned detectors are tied (AP50 0.995), the classifier is the bottleneck (top-1 0.77 on detected signs, exact sequence match about 0.43), and neither detector works on the two real photographs.

## Adopted from the revised plan (implemented)

- Two-stage pipeline: single-class detector, then sign classifier. Faster R-CNN dropped.
- Classifier is a frozen DINOv2 embedding + cosine nearest neighbour over a reference bank (`classifier.py`); it replaced the MobileNetV2 softmax. Rebuild the bank with `python grapheme_extraction/classifier.py` (no training).
- Seal photos are mirrored to impression view before detection (`pipeline.py --seal`); reading order is lines top to bottom, right to left.
- No flip augmentation (`fliplr=0, flipud=0` in `train_detector.py`).
- JSON output carries `line`, `order`, `bbox_xywh`, `det_conf` and `top_k` with probabilities.
- Labelling tools: `data/real/manifest.csv`, `bootstrap_boxes.py` (detector drafts to Label Studio), `assign_labels.py` (catalogue text to sign IDs by position).

## Kept from the earlier work

- `synthetic_generator.py`. It now supplies the crops for the classifier bank (the plan's Level 3) and is the plan's Level 4 if real-data recall stays below 0.85.
- Synthetic evaluation (`evaluate.py`), on a held-out synthetic test split.

## Not done

- Collecting and hand-labelling real seals: 50-100 frozen test seals first, then 150-250 train/val. This is manual work.
- Retraining the detector on real data. (The YOLOv8n vs RT-DETR comparison on synthetic data is done.) The current `best.pt` was trained on synthetic data with the default `fliplr=0.5`; retrain it.
- Evaluation on a real test split. `evaluate.py` already reports bootstrap intervals and frequency buckets, on the synthetic test split.
- Copy-paste augmentation (plan Level 2).
- Checking which way the IM-417 drawings face relative to impressions (plan step 1).

## Corrections to the revised plan

- **One object per image.** Two of the three real photos show the seal and its impression side by side, which a per-image `is_seal_matrix` flag cannot describe. Crop each object into its own image before it enters the manifest.
- **Catalogue matching is harder than section 3.3 assumes.** The IM77 export is keyed by Mahadevan text number, not CISI number, so a photo captioned "M-42" does not join to it directly. The manifest therefore has a separate `m77_text_no` column. The only ready link is the mayig corpus (179 Mohenjo-daro seals, Parpola IDs, about 77% of tokens convertible to Mahadevan IDs unambiguously); the yajnadevam `cisi_number` column is unchecked.
- **Illegible signs.** 781 corpus tokens are the illegible marker `0`; `assign_labels.py` sends any text containing one to manual review.
- **Temperature.** The first tuning run chose the lowest value in its grid (0.01); the grid now extends to 0.001. Re-tune on real validation crops.
- **Synthetic texts.** The generator samples from all IM77 texts. Before any real evaluation, restrict it to texts not in the test split.
