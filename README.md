# Indus Epigraphy AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Computational epigraphy of the undeciphered **Indus Valley script** (~2600–1900 BCE). The project treats the inscriptions as a structured notational system and studies their visual and sequential regularities. It does not attempt decipherment and makes no claim that the inscriptions encode a language.

Scope and milestones: [Indus_Epigraphy_AI_Project_Proposal.md](Indus_Epigraphy_AI_Project_Proposal.md).

## Status

| Milestone | State | Code | Results |
|---|---|---|---|
| 1. Seal photo to ordered sign IDs | Done on synthetic seals; does not yet work on real photographs | `grapheme_extraction/` | [report](reports/milestone_1/milestone_1_report.md), [findings](reports/milestone_1/milestone_1_findings.md) |
| 2. Sign-sequence statistics | Done on the IM77 corpus | `sequence_analysis/` | [report](reports/milestone_2/milestone_2_report.md) |
| 3. Masked-sign prediction | Done: a small transformer is compared with the Milestone 2 bigram (the bigram wins) | `sequence_analysis/masked_model.py` | [report](reports/milestone_3/milestone_3_report.md) |
| 4. Hallucination benchmark | Not started | | |

## Setup

Uses [`uv`](https://github.com/astral-sh/uv) and Python 3.10.

```bash
git clone https://github.com/farhan3011/indus-epigraphy-ai.git
cd indus-epigraphy-ai
uv venv AI --python 3.10
source AI/bin/activate          # Windows: AI\Scripts\activate
uv pip install -r requirements.txt
```

Raw data is not in the repository. The two download commands are in [data/README.md](data/README.md).

## Demo app

```bash
streamlit run app.py     # opens http://localhost:8501
python app.py            # self-check
```

One page, two tabs. Milestone 1: pick a synthetic seal with a known answer, a real Met Museum photo, or upload a JPEG; the app draws the boxes, names each sign and shows the top five guesses. Milestone 2: guess a hidden sign in an unseen inscription, compare a real inscription with its reorderings, and explore where each sign sits and what its neighbours are. It needs the raw data, `runs/` and `models/` on the machine, so it runs locally only.

## Milestone 1: grapheme extraction

A single-class detector (YOLOv8n or RT-DETR) finds the signs, and a DINOv2 nearest-neighbour classifier names each one with a Mahadevan sign number. One command generates the synthetic seals, trains both detectors, builds the classifier and writes the report; it needs a GPU to be practical.

```bash
python grapheme_extraction/run_all.py
python grapheme_extraction/pipeline.py photo.jpg --annotate out.jpg    # run on one image
```

Step-by-step instructions: [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md). Design decisions and open items: [docs/milestone1_status.md](docs/milestone1_status.md).

## Milestone 2: sign-sequence analysis

N-gram models with Witten-Bell smoothing, transition matrices, sign-pair statistics and writing-direction measures on the IM77 corpus, plus distributional syntax: positional sign classes, class-order rules and recurring segments, checked on held-out texts. Robustness checks cover:
- comparison with sign systems of known type (English, DNA, Python code, Linear B, Sumerian cuneiform and cuneiform seal inscriptions, folk melodies, Rao's controls)
- the direction test with the likely direction-deciding signs removed
- a position baseline that counts from both ends
- repeated cross-validation and the effect of unseen signs
- a second, independent transcription (Wells/ICIT), with sign variants merged and split
- corpus variants with doubtful readings, duplicate copies and multi-line texts
- differences by site, object type and field symbol
- line breaks in multi-line and boustrophedon texts
- bias-corrected entropy estimators and sample-size curves
- missing-sign restoration with trigram and Kneser-Ney models
- simulated re-segmentation of compound signs
- the two replication gaps
- every p-value in the report corrected together (Benjamini-Hochberg and Holm)

CPU only, a few minutes (`--quick` for a fast check). Every random step uses a fixed seed, and `results.json` records the Python and package versions and the SHA-256 of every input. Tests: `python -m pytest -q sequence_analysis`; CI runs them on every push. The extra inputs are listed in [data/README.md](data/README.md); without them the affected checks are skipped.

```bash
python sequence_analysis/run_milestone2.py
```

Planned once Milestone 1 is properly hand-labelled on real seals:
- an error-propagation test that uses Milestone 1's real confusion matrix instead of uniform errors
- a direction test that uses sign positions measured on photographs instead of Mahadevan's direction labels

## Milestone 3: masked-sign prediction

```bash
python sequence_analysis/masked_model.py                          # 5-fold evaluation, 3 seeds, about 6 minutes on a laptop CPU
python sequence_analysis/masked_model.py predict "267 99 ? 342"    # fill gaps; reading order, ? for each missing sign
```

A small transformer reads the whole inscription and ranks every sign for a hidden position. It is scored on exactly the trials Milestone 2 used for the bigram (every sign of every held-out text, 5-fold), with calibrated confidences. Result: first guess right 30.0% for the transformer against 38.0% for the bigram; averaging the two ties the bigram. With about 1,500 short texts, the two neighbours carry nearly everything a model can use. The final model is also run on the 337 corpus texts with one illegible sign.

## Layout

```
grapheme_extraction/   Milestone 1 code
sequence_analysis/     Milestone 2 code
reports/milestone_1/   Milestone 1 results (synthetic data)
reports/milestone_2/   Milestone 2 results
notebooks/             Milestone 1 results notebook
data/                  download instructions, sample photos, labelling manifest
docs/                  run guide and Milestone 1 status
```
