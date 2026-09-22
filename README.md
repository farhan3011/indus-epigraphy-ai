# Indus Epigraphy AI

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Package Manager](https://img.shields.io/badge/uv-fast_packaging-purple.svg)](https://github.com/astral-sh/uv)
[![Framework](https://img.shields.io/badge/PyTorch-2.x-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Computational epigraphy framework for the undeciphered **Indus Valley Script** (~2600–1900 BCE). Treating inscriptions as a structured, notational information system, this project deploys deep-learning vision backbones for grapheme extraction, directional Markov modeling, and masked-token language encoders over archaeological seal corpora.

---

## Getting Started (Quickstart)

This project standardizes on [`uv`](https://github.com/astral-sh/uv) for fast, reproducible virtual environment and package management.

### Option A: Ubuntu (Linux)

#### 1. Clone the Repository
```bash
git clone https://github.com/farhan3011/indus-epigraphy-ai.git
cd indus-epigraphy-ai
```

#### 2. Install `uv`
If you do not already have `uv` installed, install it via the official standalone script (recommended) or snap:
```bash
# Recommended standalone installer
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via snap
sudo snap install uv --classic
```

#### 3. Create the Virtual Environment
Create a dedicated virtual environment named `AI` using Python 3.10:
```bash
uv venv AI --python 3.10
```

#### 4. Activate the Environment
```bash
source AI/bin/activate
```

#### 5. Install Dependencies
```bash
uv pip install -r requirements.txt
```

#### 6. (Optional) Install Headless Browser Tools
For running automated scraping or dataset acquisition pipelines:
```bash
playwright install chromium
```

#### 7. Verify the Setup & GPU Acceleration
```bash
python -c "import torch, torchvision, ultralytics, cv2; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Ultralytics:', ultralytics.__version__)"
```

---

### Option B: macOS (Apple Silicon & Intel)

#### 1. Clone the Repository
```bash
git clone https://github.com/farhan3011/indus-epigraphy-ai.git
cd indus-epigraphy-ai
```

#### 2. Install `uv`
Using [Homebrew](https://brew.sh/) (recommended) or the standalone installer:
```bash
# Via Homebrew
brew install uv

# Or via standalone installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### 3. Create the Virtual Environment
```bash
uv venv AI --python 3.10
```

#### 4. Activate the Environment
```bash
source AI/bin/activate
```

#### 5. Install Dependencies
```bash
uv pip install -r requirements.txt
```

#### 6. (Optional) Install Headless Browser Tools
```bash
playwright install chromium
```

#### 7. Verify the Setup & Apple Silicon Acceleration (MPS)
```bash
python -c "import torch, torchvision, ultralytics, cv2; print('PyTorch:', torch.__version__, '| Apple MPS Available:', torch.backends.mps.is_available(), '| Ultralytics:', ultralytics.__version__)"
```

---

## Project Documentation & Milestone Roadmap

- 📖 **[Milestone 1 Product Requirements Document (PRD)](docs/PRD_Milestone_1_Grapheme_Extraction.md)**:
  Detailed documentation explaining the computational logic behind **YOLO vs. Faster R-CNN vs. RT-DETR**, why and how **synthetic seal generation** is constructed (Poisson blending, intaglio relief modeling, and zero-cost bounding-box generation), and the reality-check holdout evaluation protocol.
- 📋 **[Project Proposal & Academic Scope](Indus_Epigraphy_AI_Project_Proposal.md)**:
  The complete four-milestone epigraphy curriculum (visual extraction, Markov chain modeling, masked-token encoders, and hallucination benchmarking).
- 🗄️ **[Data Sources & Directory Guide](data/README.md)**:
  Download references for the IM77 concordance, Kaggle IM-417-150 glyph dataset, and CISI Parpola corpus.

---

## Repository Layout

```
indus-epigraphy-ai/
├── AI/                                # Local virtual environment (gitignored)
├── data/                              # Data directories & acquisition guides
│   └── README.md
├── docs/                              # Technical specifications & PRDs
│   └── PRD_Milestone_1_Grapheme_Extraction.md
├── .gitignore                         # Environment, checkpoint, and data exclusions
├── Indus_Epigraphy_AI_Project_Proposal.md # Academic course proposal & milestones
├── LICENSE                            # MIT License
├── README.md                          # Project overview & cross-platform setup guide
└── requirements.txt                   # Pinned dependency manifest
```
