# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A student computational-epigraphy project on the undeciphered Indus script. It treats inscriptions as a notational system and makes no decipherment claims. Four milestones are defined in `Indus_Epigraphy_AI_Project_Proposal.md`; only Milestone 1 (seal photo -> ordered list of Mahadevan sign IDs) has code, in `grapheme_extraction/`.

Read `docs/milestone1_status.md` before changing direction: it records which design was adopted and why. `docs/PRD_Milestone_1_Grapheme_Extraction.md` and `milestone1.md` are earlier plans that were superseded (three-way detector benchmark, softmax classifier, flip augmentation); do not implement from them.

## Commands

Environment is `uv` with Python 3.10 in `AI/` (gitignored). `venv/` is a stale Python 3.14 env; ignore it.

```bash
uv venv AI --python 3.10 && source AI/bin/activate && uv pip install -r requirements.txt
```

Raw data is not in git. The two download commands are in `data/README.md` (IM-417 glyphs via `kagglehub`, IM77 corpus CSV via `curl` at a pinned commit).

All scripts are run from the repo root as files, not as a package (they import each other as top-level modules, e.g. `from classifier import ...`):

```bash
python grapheme_extraction/run_all.py            # everything, in order (steps below)
python grapheme_extraction/run_all.py --quick    # tiny install check; OVERWRITES data/synthetic and models/bank.pt
python grapheme_extraction/run_all.py --from 3   # resume at a step

python grapheme_extraction/synthetic_generator.py --train 4000 --val 400 --test 400
python grapheme_extraction/train_detector.py                      # YOLOv8n -> runs/yolov8n/
python grapheme_extraction/train_detector.py --model rtdetr-l.pt  # RT-DETR -> runs/rtdetr-l/
python grapheme_extraction/classifier.py                          # builds models/bank.pt (no training)
python grapheme_extraction/evaluate.py                            # writes reports/milestone_1_report.md
python grapheme_extraction/pipeline.py photo.jpg [--seal] [--detector runs/rtdetr-l/weights/best.pt] [--annotate out.jpg]
```

There is no test framework or linter. Each script carries `assert`-based self-checks for its non-trivial logic:

```bash
python grapheme_extraction/pipeline.py --selftest   # reading order, line grouping, mirroring, edit distance
```

`classifier.py`, `evaluate.py`, `bootstrap_boxes.py` and `assign_labels.py` run their asserts at the top of `__main__` before doing real work.

`docs/RUN_GUIDE.md` is the step-by-step guide for running on a GPU machine.

## Architecture

Two-stage pipeline: a **single-class** detector finds signs, then a separate classifier names each crop.

- `synthetic_generator.py` composites IM-417 glyph drawings onto procedural stone textures. It writes YOLO labels whose class is `sign_id - 1` (417 classes). That one label file serves three consumers: the detector (trained with `single_cls=True`, so classes collapse), the classifier (crops + sign IDs), and evaluation (signs sorted by descending x give the ground-truth reading order). Half the generated inscriptions are real IM77 texts, half are uniform random signs so rare signs get coverage.
- `train_detector.py` wraps ultralytics; the model class (`YOLO` vs `RTDETR`) is chosen by whether the name contains "rtdetr". `pipeline.load_detector` uses the same rule; loading RT-DETR weights through `YOLO()` is wrong.
- `classifier.py` is a frozen DINOv2 (`facebook/dinov2-small`) embedding with cosine nearest-neighbour over a reference bank (`models/bank.pt`), returning per-class max similarity divided by a temperature tuned on the val split. `load_classifier()` returns a callable giving logits; adding labelled crops to the bank needs no retraining.
- `pipeline.py` `extract()` produces the output JSON: `sequence` (top-1 IDs in reading order) plus per-sign `line`, `order`, `bbox_xywh`, `det_conf`, `top_k` with probabilities. Later milestones depend on `top_k` (sequence rescoring) and on `bbox_xywh` (testing reading direction from physical positions).
- `evaluate.py` scores each detector end to end on the held-out `test` split with bootstrap intervals over seals, and generates the report markdown itself (`write_report`). Edit the report text there, not in `reports/`.
- `bootstrap_boxes.py` and `assign_labels.py` are helpers for hand-labelling real seals (`data/real/manifest.csv`); no real labelled data exists yet.

Module dependencies run one way: `classifier` <- `pipeline` <- `evaluate`; `synthetic_generator` is imported only for its `IM77_CSV` path. Path constants (`ROOT`, `DATA`, `BANK`, `DETECTOR`) are defined in `classifier.py` and `pipeline.py`.

## Domain rules that are easy to break

- **Label space is Mahadevan (M77) sign numbers 1-417 everywhere.** The CISI JSON corpus (`data/raw/cisi`) uses Parpola `P###` IDs; its `features/P###.json` files carry a `mahadevan_graphemes` cross-reference.
- **Reading order is right-to-left in impression view.** A photo of the carved seal is the mirror image: `extract(..., is_seal_matrix=True)` flips the image before detection, and all output boxes are in impression-view pixels.
- **Never flip-augment** (`fliplr=0, flipud=0`): a mirrored sign is a different sign.
- **Relative sign size carries identity** (e.g. short strokes 99 vs long strokes 87), which is why the generator uses one scale per seal.
- **Splits:** `val` is for model selection and temperature tuning; `test` is only read by `evaluate.py`.
- **IM-417 has one drawing per sign** (417 PNGs, white on black, 150x150). Classifier accuracy on synthetic crops compares augmentations of the same drawings.

## State of the evidence

Every metric in the repo is measured on synthetic data. A synthetic-trained YOLOv8n transferred poorly to three real seal photos (merged neighbouring signs, boxed the animal motif). Report synthetic numbers as synthetic, and do not describe pipeline output on a real photo as a reading. The full run has been done once on a CUDA machine; its results are in `reports/` (`milestone_1_report.md`, `milestone_1_findings.md`, `RUN_INFO.md`).

## Data and licensing

`data/raw/`, `data/synthetic/`, `data/real/source|images/`, `runs/`, `models/` and `*.pt` are gitignored. `data/synthetic/data.yaml` stores an absolute path, so regenerate the dataset on each machine instead of copying it. The IM77 CSV is a third-party export with no clear licence for the underlying data: download per user, never commit. Only CC0 images go in `data/samples/`. Background on data sources and licences: `reports/Indus sign detection data gaps.md`.

## Network quirk seen during development

On one network, GitHub release-asset downloads were reset, which breaks ultralytics' automatic weight download. `yolov8n.pt` can be fetched from `https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt` and passed with `--model <path>`; RT-DETR can be built from `rtdetr-l.yaml` (no pretrained weights).
