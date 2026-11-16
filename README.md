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
| 3. Masked-sign prediction | Done: neural models against the best n-gram (Kneser-Ney trigram, 42.2%); the n-gram wins alone and a mixture only ties it | `sequence_analysis/masked_model.py` | [report](reports/milestone_3/milestone_3_report.md) |
| 4. Decipherment-claim test harness | Done: the one runnable Sanskrit key is not distinguishable from shuffled versions of itself | `decipherment/` | [report](reports/milestone_4/milestone_4_report.md) |

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

One page, four tabs. Milestones 3 and 4 and their extensions are described in their sections below. Milestone 1: pick a synthetic seal with a known answer, a real Met Museum photo, or upload a JPEG; the app draws the boxes, names each sign and shows the top five guesses. Milestone 2: guess a hidden sign in an unseen inscription, compare a real inscription with its reorderings, and explore where each sign sits and what its neighbours are. It needs the raw data, `runs/` and `models/` on the machine, so it runs locally only.

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

### Extension: do the stroke signs behave like numerals? (exploratory)

```bash
python sequence_analysis/numerals.py    # about 30 seconds on a laptop CPU
```

The 25 signs drawn as rows of equal bars are found from the glyph drawings by a stated rule (contact sheet in the report folder) and five numeral predictions are tested against pseudo sets of frequency-matched signs. Two hold after correction: text pairs differing only in which stroke sign stands in one slot are about 3 times as common as for matched signs (116 against 35), and frequency falls with bar count (Spearman -0.60). Three do not: stroke signs are not followed by a narrower set of signs, different stroke signs are not followed by the same signs, and their position in a text and their tendency to run together do not differ from matched signs. Report: [reports/milestone_2/numerals_report.md](reports/milestone_2/numerals_report.md).

### Extension: do the stroke signs behave like numerals? (exploratory)

```bash
python sequence_analysis/numerals.py    # about 30 seconds on a laptop CPU
```

The 25 signs drawn as rows of equal bars are found from the glyph drawings by a stated rule (contact sheet in the report folder) and five numeral predictions are tested against pseudo sets of frequency-matched signs. Two hold after correction: text pairs differing only in which stroke sign stands in one slot are about 3 times as common as for matched signs (116 against 35), and frequency falls with bar count (Spearman -0.60). Three do not: stroke signs are not followed by a narrower set of signs, different stroke signs are not followed by the same signs, and their position in a text and their tendency to run together do not differ from matched signs. Report: [reports/milestone_2/numerals_report.md](reports/milestone_2/numerals_report.md).

### Extension: does the text predict the object? (exploratory)

```bash
python sequence_analysis/predict_object.py    # about 1 minute on a laptop CPU
```

A logistic regression on sign and sign-pair counts is scored by 5-fold cross-validation grouped by text. Balanced accuracy: site 31% (chance 20%), object type 50% (chance 20%; text length alone gives 42%), field symbol 46% (chance 33%). All beat labels permuted within the training folds, but modestly, and adding sign pairs to single signs changes little, so what predicts the object is mostly which signs occur, not their order. Report: [reports/milestone_2/predict_object_report.md](reports/milestone_2/predict_object_report.md).

## Milestone 3: masked-sign prediction

```bash
python sequence_analysis/masked_model.py                          # 5-fold evaluation, 3 seeds, about 15 minutes on a laptop CPU
python sequence_analysis/masked_model.py predict "267 99 ? 342"    # fill gaps; reading order, ? for each missing sign
```

Small neural models (transformer or bidirectional LSTM, chosen per fold on an inner split) read the whole inscription and rank every sign for a hidden position. They are scored on exactly the trials Milestone 2 used (every sign of every held-out text, 5-fold) against the best n-gram, an interpolated Kneser-Ney trigram, with calibrated confidences and conformal candidate sets. Result: the neural model alone gets the first guess right 32.2% of the time against the n-gram's 42.2%; a fitted mixture of the two (42.4%) and a residual model are statistically tied with the n-gram. The neighbouring signs carry almost everything a model can use. The final model is also run on the 337 corpus texts with one illegible sign. On Yadav et al.'s measure (the true sign inside the set holding 90% of the predicted probability), every calibrated model reaches 87-91% against their published 74% for a bigram, but with sets of 61-91 signs, so the figures are not like for like: their split is unstated and a larger set always hits more often (table in the report).

### Extension: several gaps, unknown-length damage, miscopies

```bash
python sequence_analysis/restore.py                       # evaluation + report, about 7 minutes on a laptop CPU
python sequence_analysis/restore.py fill "267 ? ? 342"    # ? = one missing sign, * = a damaged stretch of 0-3 signs
python sequence_analysis/restore.py check "267 99 342"    # how expected each sign is, given the others
```

Milestone 3 filled several gaps independently of each other. `restore.py` searches all gaps together on the Kneser-Ney trigram (both of two hidden signs right in 19.3% of held-out trials, against 7.5% for the neural model filling each gap on its own and 10.3% for a greedy search), handles a damaged stretch whose length is not known (length right in 53.6% of trials against a 32.6% baseline), flags a sign that does not fit its text (a swapped sign is located in 61.7% of texts, chance 22.1%) and retrieves parallel corpus texts for a gap. The demo app shows all four, plus which context signs the neural model leaned on. Report: [reports/milestone_3/restoration_extensions_report.md](reports/milestone_3/restoration_extensions_report.md).

## Milestone 4: testing claimed decipherments

```bash
python decipherment/data.py       # fetch the pinned keys and corpora into data/raw/decipherment (never committed)
python decipherment/harness.py    # controls + every claim in decipherment/prereg.json; about 5 minutes on a laptop CPU
```

A claimed decipherment is a sign-to-sound table. The harness applies a table to the inscriptions, scores how Sanskrit-like the order of the output is, and compares it with 10,000 tables made by shuffling the claim's own values, under rules committed in advance (`decipherment/prereg.json`). A positive control (Sanskrit enciphered with a known table) shows the test detects true tables 95-100% of the time; a negative control shows it does not fire falsely. Of the published claims, only Yajnadevam's Sanskrit key can be run in full; it is not distinguishable from shuffled versions of itself (frequency-matched null, Holm p about 0.65). Most other claims publish no fixed table and are listed as untestable. The test can reject a specific table; it cannot show whether the script records a language.

### Extension: what a blind search can reach (exploratory, not pre-registered)

```bash
python decipherment/overfit.py    # about 7 minutes on a laptop CPU
```

The permutation test compares a claimed table with random tables, but a decipherer searches for a table. `overfit.py` searches too: random tables are improved by swapping the values of two signs whenever the score does not fall. On the Indus inscriptions the search passes the claimed table within about 50 swaps and ends 0.81 bits per phoneme above it; tables searched on half the texts still score far above the claim on the other half. The same search on real Sanskrit enciphered with a known table also passes the true table, with tables that disagree with it, so this score cannot pick out the right table on a corpus of this size and scoring below the search ceiling is not evidence against a claim. The demo app runs a search live and tests any uploaded sign-to-sound table with the Milestone 4 test. Report: [reports/milestone_4/overfitting_report.md](reports/milestone_4/overfitting_report.md).

### Extension: dictionary match rate (exploratory, not pre-registered)

```bash
python decipherment/lexicon.py    # about 4 minutes on a laptop CPU
```

A second score for a claimed table: how much of its output, reduced to consonant skeletons, is made of Sanskrit words (seen at least twice in the DCS training split), compared with shuffled versions of the same table. A table with correct values is detected 95-100% of the time by both scores (positive control), and the false-alarm rate is 4-9%. Yajnadevam's table beats fully shuffled tables on whole-text matches, but not tables shuffled within frequency bins (percentile 58-62), so the excess is how often each sign occurs, not which sound it carries; coverage is not significant either way. Report: [reports/milestone_4/lexicon_report.md](reports/milestone_4/lexicon_report.md).

### Extension: a positive control on a real decipherment (exploratory, not pre-registered)

```bash
python decipherment/linear_b.py    # about 1 minute on a laptop CPU after the first download
```

The harness is run, unchanged, on Linear B read as Greek: 918 Linear B words (a word list from the NeuroDecipher repository, not tablet text) with the accepted Mycenaean sound values, against Homer. The true table is detected only against frequency-matched shuffles (percentile 99, p 0.010), not against fully shuffled tables (p 0.32); a table with 10% of its values replaced is still detected 90% of the time, 30% replaced 35%. Scored against Sanskrit instead of Greek the same table is also significant, so a significant result from this test is not evidence that a table is right for its language. Report: [reports/milestone_4/linear_b_report.md](reports/milestone_4/linear_b_report.md).

### Extension: the same claim against four languages (exploratory, not pre-registered)

```bash
python decipherment/languages.py    # about 1 minute on a laptop CPU
```

Yajnadevam's tables are compared with shuffled versions of themselves under four language models (Sanskrit, Old Tamil, English, Sumerian), each reduced to the consonant skeleton and cut to the same 336,466 letters, and a blind search is run under each. Against frequency-matched shuffles the table sits at percentile 94 for Sanskrit, 87 for Tamil, 76 for English and 22 for Sumerian, and none is significant after Holm correction, so the data do not favour Sanskrit over the others. Search ceilings differ between languages by more than their spread, but part of that comes from the table's Sanskrit-shaped values. Report: [reports/milestone_4/languages_report.md](reports/milestone_4/languages_report.md).

### Extension: how large would the corpus have to be? (exploratory)

```bash
python decipherment/corpus_size.py    # about 55 minutes on a laptop CPU
```

Sanskrit is enciphered with the claimed sign-to-sound relation into corpora from 0.5 to 20 times the Indus size, so the true table is known. The permutation test detects the true table at every size, but a blind search finds tables scoring above it on held-out texts at every size too (the true table beats every search in 0% of replicates from the Indus size up to 20 times it), and those tables agree with the true one on only 3-5% of tokens. A larger corpus does not fix this under this score: the true table is not what the score is maximised by. Report: [reports/milestone_4/corpus_size_report.md](reports/milestone_4/corpus_size_report.md).

## To do

- Retrain the Milestone 1 detectors; see [docs/RUN_GUIDE.md](docs/RUN_GUIDE.md).

## Layout

```
grapheme_extraction/   Milestone 1 code
sequence_analysis/     Milestone 2 and 3 code (masked_model.py, restore.py)
decipherment/          Milestone 4 code (harness.py, overfit.py)
reports/milestone_1/   Milestone 1 results (synthetic data)
reports/milestone_2/   Milestone 2 results
reports/milestone_3/   Milestone 3 results and the restoration extension
reports/milestone_4/   Milestone 4 results and the search extension
notebooks/             Milestone 1 results notebook
data/                  download instructions, sample photos, labelling manifest
docs/                  run guide and Milestone 1 status
```
