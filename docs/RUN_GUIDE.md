# Run guide: all of Milestone 1 on a GPU machine (synthetic data)

Run the steps in this order. Each one needs the output of the one before it. Steps 1 to 3 are setup; steps 4 to 8 can be run with one command.

## 1. Get the code

```bash
git clone -b milestone-1-pipeline https://github.com/farhan3011/-indus-epigraphy-ai-.git
cd -- -indus-epigraphy-ai-
```

## 2. Set up Python

```bash
# install uv first if needed: https://docs.astral.sh/uv/getting-started/installation/
uv venv AI --python 3.10
source AI/bin/activate            # Windows: AI\Scripts\activate
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

The last line must print `CUDA: True`. If it prints `False` (common on Windows, where the default torch build is CPU-only), reinstall torch for your CUDA version, for example:

```bash
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 3. Download the raw data (about 20 MB)

```bash
# 417 sign drawings, one per Mahadevan sign
python -c "import kagglehub, shutil; shutil.copytree(kagglehub.dataset_download('storesource/im-417-150'), 'data/raw/im417')"

# 2,906 real inscription texts as Mahadevan sign numbers
mkdir -p data/raw/im77
curl -sSL -o data/raw/im77/im77_corpus_lines.csv https://raw.githubusercontent.com/paragorth/Indus-/ed56b9dc293f26a4e6bddea36c9a394e6b211ff4/data/im77/im77_corpus_lines.csv
```

Check: `data/raw/im417/IM_417_150_PNG/` holds 417 PNG files, and the CSV has 3,917 lines. If the Kaggle download asks for credentials, create a free Kaggle API token and put `kaggle.json` in `~/.kaggle/`.

## Shortcut: run steps 4 to 8 with one command

```bash
python grapheme_extraction/run_all.py --quick   # optional install check, a few minutes, tiny dataset and 1 epoch
python grapheme_extraction/run_all.py           # the real run
```

`--quick` overwrites `data/synthetic/` and `models/bank.pt` with tiny versions and writes to `reports/quick/`; the full run regenerates everything. If a step crashes, fix it and resume with `--from N` (N = 1 to 5, matching steps 4 to 8 below). The steps below are what the runner does, in case you want to run them one at a time.

## 4. Generate the synthetic seals (about 1 minute, CPU only)

```bash
python grapheme_extraction/synthetic_generator.py --train 4000 --val 400 --test 400
```

Writes images and labels to `data/synthetic/` plus `data/synthetic/data.yaml`. Always generate on the machine you train on: `data.yaml` stores an absolute path, so a copied dataset will not train. The test split is held out: it is only read in step 8.

## 5. Train the YOLOv8n detector (GPU)

```bash
python grapheme_extraction/train_detector.py
```

30 epochs, batch 32. Downloads `yolov8n.pt` on first run. Weights land in `runs/yolov8n/weights/best.pt`, training curves in `runs/yolov8n/results.csv`. Ten epochs on an Apple M2 took about 25 minutes and reached AP50 0.995 on synthetic validation.

## 6. Train the RT-DETR detector (GPU, the heavy step)

```bash
python grapheme_extraction/train_detector.py --model rtdetr-l.pt
```

60 epochs, batch 8, stops early if validation has not improved for 15 epochs. RT-DETR-L is about ten times the size of YOLOv8n and is the longest step by far; it has not been run to completion anywhere yet, so there is no timing to quote. Weights land in `runs/rtdetr-l/weights/best.pt`.

- Out of GPU memory: add `--batch 4`.
- Plenty of GPU memory (16 GB or more): `--batch 16` is faster.
- `rtdetr-l.pt` will not download: use `--model rtdetr-l.yaml` (trains from scratch, expect worse results; say so when you report back).

## 7. Build the classifier bank (GPU, a few minutes, no training)

```bash
python grapheme_extraction/classifier.py
```

Downloads DINOv2-small from Hugging Face on first run, embeds the synthetic training crops and writes `models/bank.pt`. On the Mac this gave top-1 0.790 and top-5 0.943 on synthetic validation crops.

## 8. Evaluate everything and write the report

```bash
python grapheme_extraction/pipeline.py --selftest    # must print: ok
python grapheme_extraction/evaluate.py               # YOLOv8n, RT-DETR and the OpenCV baseline
```

`evaluate.py` scores all three detectors on the held-out synthetic test split with the same classifier and writes, under `reports/milestone_1/`:

- `milestone_1_report.md`: the comparison table (AP50, precision, recall, recall on touching signs, sign error rate, exact match, speed), each with a 95% bootstrap interval;
- `eval_results.json` and `per_class_f1.csv`;
- `yolov8n_results.csv/.png` and `rtdetr-l_results.csv/.png` (training curves);
- `figures/`: each detector's boxes drawn on the two real Met seal photos in `data/samples/`.

Open `notebooks/milestone1_results.ipynb` afterwards to see the curves and tables together.

Rough reference from an earlier Mac run with YOLOv8n (validation split, older code): detection recall 1.000, sign error rate 0.232, exact sequence match 0.493.

## 9. Run on one photo

```bash
python grapheme_extraction/pipeline.py path/to/photo.jpg --annotate out.jpg                                         # impression, YOLO
python grapheme_extraction/pipeline.py path/to/photo.jpg --seal --annotate out.jpg                                  # the carved seal itself
python grapheme_extraction/pipeline.py path/to/photo.jpg --detector runs/rtdetr-l/weights/best.pt --annotate out.jpg  # RT-DETR
```

## What to send back

- the whole `reports/milestone_1/` folder
- `runs/yolov8n/weights/best.pt` and `runs/rtdetr-l/weights/best.pt`
- `models/bank.pt`
- which GPU was used and roughly how long steps 5 and 6 took

## Reference run

The whole chain has been run once on Linux with an RTX 5090 (see `reports/milestone_1/RUN_INFO.md`): about 45 minutes in total, 40 of them RT-DETR. Windows is untested.

## Read before trusting any number

Every metric above is measured on synthetic images. On three real seal photos the synthetic-trained detector merged neighbouring signs and boxed the animal motif, so these scores say nothing yet about real seals. See `docs/milestone1_status.md`.

## Optional: retrain on relief-shaded seals (GPU)

The generator can render signs as shaded carved relief instead of flat darkened shapes, which should transfer better to real photographs. It is off by default because the current detectors were trained on flat shapes.

```bash
mv data/synthetic data/synthetic_flat          # keep the flat set for comparison
python grapheme_extraction/synthetic_generator.py --train 4000 --val 400 --test 400 --relief
python grapheme_extraction/train_detector.py --name yolov8n-relief
python grapheme_extraction/train_detector.py --model rtdetr-l.pt --name rtdetr-l-relief
python grapheme_extraction/classifier.py
python grapheme_extraction/evaluate.py --runs yolov8n-relief rtdetr-l-relief --out reports/milestone_1_relief
```

Send back `reports/milestone_1_relief/` and both `runs/*-relief/weights/best.pt`. Look at the figures of the real Met photos first: the question is whether the boxes on real seals improve.

## Milestones 3 and 4 and their extensions (CPU only)

None of this needs a GPU. Run from the repository root with the environment from step 2 active.

```bash
# Milestone 3: masked-sign prediction. Writes reports/milestone_3/ and models/masked_sign_model.pt (about 15 minutes)
python sequence_analysis/masked_model.py

# Milestone 3 extension: several gaps searched together, damage of unknown length, miscopied signs (about 7 minutes)
python sequence_analysis/restore.py
python sequence_analysis/restore.py fill "267 ? ? 342"    # ? = one missing sign, * = a damaged stretch of 0-3 signs
python sequence_analysis/restore.py check "267 99 342"    # how expected each sign is, given the others

# Milestone 4: fetch the pinned keys and corpora once, then run the pre-registered test (about 5 minutes)
python decipherment/data.py
python decipherment/harness.py

# Milestone 4 extension, exploratory and not pre-registered: what a blind search can reach (about 7 minutes)
python decipherment/overfit.py

# Milestone 4 extension, exploratory: dictionary match rate of a table's output (about 4 minutes)
python decipherment/lexicon.py

# Milestone 4 extension, exploratory: Linear B read as Greek, a positive control on a real decipherment (about 1 minute)
python decipherment/linear_b.py
```

Order matters in two places. `restore.py` compares against the neural model, so it needs `reports/milestone_3/results.json` from `masked_model.py`. `overfit.py` and the Milestone 4 tab of the app need the files `data.py` downloads.

Every script takes `--quick` for a fast check; quick outputs go to `reports/quick/`, which is not committed.

If `python decipherment/data.py` fails with a connection reset, the network is blocking `raw.githubusercontent.com`. The script then falls back to the jsDelivr mirror and, for the Sanskrit corpus, to a sparse git checkout of the same pinned commit. Every file is still checked against its SHA-256.

Tests and the app self-check:

```bash
python -m pytest -q sequence_analysis    # if an unrelated system pytest plugin breaks collection: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python app.py                            # asserts, then runs every tab once headless
```

The app lists a detector only when its weights file exists and is not empty. Without trained Milestone 1 weights the first tab runs on the OpenCV baseline and the other three tabs work as normal.
