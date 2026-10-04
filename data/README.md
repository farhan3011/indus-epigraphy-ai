# Data Directory Guide

Raw artifact data and large corpora are not committed directly to this repository. Download the verified academic datasets from the following sources:

- **The Indus Script IM77 Corpus (RMRL):** Digitized web concordance of Iravatham Mahadevan's foundational corpus.
  - URL: [https://indusscript.in/](https://indusscript.in/)
- **IM-417-150 Grapheme Dataset (Kaggle):** Segmented individual glyph crops mapped to Mahadevan's 417 concordance signs.
  - URL: [https://www.kaggle.com/datasets/storesource/im-417-150](https://www.kaggle.com/datasets/storesource/im-417-150)
- **Digital Corpus of Indus Valley Script (CISI):** Machine-readable JSON serialization of Asko Parpola's corpus.
  - URL: [https://github.com/mayig/indus-valley-script-corpus](https://github.com/mayig/indus-valley-script-corpus)
- **Indus Script Synthetic Corpus (Hugging Face):** 5,000 synthetic symbol sequences cross-validated against historical seals.
  - URL: [https://huggingface.co/datasets/hellosindh/indus-script-synthetic](https://huggingface.co/datasets/hellosindh/indus-script-synthetic)

Place all downloaded raw files under `data/raw/` locally — this directory path is excluded via `.gitignore`.

## What Milestone 1 actually uses

Everything below lands in `data/raw/` (gitignored). Fetch it with:

```bash
git clone --depth 1 https://github.com/mayig/indus-valley-script-corpus data/raw/cisi
python -c "import kagglehub, shutil; shutil.copytree(kagglehub.dataset_download('storesource/im-417-150'), 'data/raw/im417')"
mkdir -p data/raw/im77 && curl -L -o data/raw/im77/im77_corpus_lines.csv \
  https://raw.githubusercontent.com/paragorth/Indus-/ed56b9dc293f26a4e6bddea36c9a394e6b211ff4/data/im77/im77_corpus_lines.csv
```

- **`im417/`** — 417 images in total: one 150x150 white-on-black drawing per Mahadevan sign (the "150" is the resolution, not a per-class sample count).
- **`im77/im77_corpus_lines.csv`** — third-party export of Mahadevan's 1977 concordance in Mahadevan sign numbers, reading order, `0` = illegible sign. The underlying data belongs to Mahadevan / RMRL and carries no stated open licence: use it locally, do not commit it.
- **`cisi/`** — 179 Mohenjo-daro artefacts in Parpola numbering; each `features/Pnnn.json` has a `mahadevan_graphemes` field mapping to Mahadevan IDs. Not used by the generator now that IM77 sequences are available.

## Extra inputs for the Milestone 2 robustness checks

```bash
# second transcription: yajnadevam/indus-website (GPL-3.0), Wells/ICIT sign numbers, pinned commit
mkdir -p data/raw/yajnadevam && curl -sSL -o data/raw/yajnadevam/population-script.sql \
  https://raw.githubusercontent.com/yajnadevam/indus-website/2434249618b1e74de3f26d28b90dfde093c3dd94/population-script.sql
# reference systems: an English novel (Project Gutenberg #1342) and the phage lambda genome (NCBI NC_001416.1)
mkdir -p data/raw/reference
curl -sSL -o data/raw/reference/pride_and_prejudice.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt
curl -sSL -o data/raw/reference/lambda_phage.fasta \
  "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001416.1&rettype=fasta&retmode=text"
```

Python code tokens come from the local Python 3.10 standard library, so nothing is downloaded for them.

## What Milestone 1 actually uses

```bash
# 1. Mahadevan sign glyphs: 417 PNGs, ONE 150x150 white-on-black drawing per sign (not 150 samples per class)
python -c "import kagglehub, shutil; shutil.copytree(kagglehub.dataset_download('storesource/im-417-150'), 'data/raw/im417')"

# 2. IM77 texts as Mahadevan sign IDs (2,906 texts; third-party export of indusscript.in, pinned commit)
mkdir -p data/raw/im77 && curl -sSL -o data/raw/im77/im77_corpus_lines.csv \
  https://raw.githubusercontent.com/paragorth/Indus-/ed56b9dc293f26a4e6bddea36c9a394e6b211ff4/data/im77/im77_corpus_lines.csv
```

The IM77 export carries no clear licence for the underlying Mahadevan/RMRL data, so it is downloaded per user and never committed. The CISI JSON corpus uses Parpola numbering (`P###`); its `features/P###.json` files hold a `mahadevan_graphemes` cross-reference if a P-to-M mapping is needed later.
