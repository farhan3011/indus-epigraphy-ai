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
