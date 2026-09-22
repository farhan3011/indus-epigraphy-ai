# Product Requirements Document (PRD): Milestone 1
## Automated Visual Grapheme Extraction for Indus Epigraphy AI

---

### Document Information
- **Project Name:** Computational Epigraphy of the Indus Valley Script
- **Milestone:** Milestone 1 — Automated Visual Grapheme Extraction
- **Author/Lead:** Epigraphy AI Team
- **Virtual Environment:** Python 3.10 via `uv` (`AI/`)
- **Version:** 1.0.0
- **Status:** Approved for Implementation

---

## 1. Executive Summary & Problem Formulation

### 1.1 Archaeological & Epigraphic Context
The Indus Valley Script (~2600–1900 BCE) remains one of antiquity's greatest undeciphered scripts. Unlike Egyptian hieroglyphs (deciphered via the multilingual Rosetta Stone) or Mesopotamian cuneiform (unlocked via the Behistun Inscription), the Indus script has **no bilingual texts**. Furthermore, the surviving corpus consists of approximately 4,000 inscribed artifacts (steatite seals, terracotta tablets, pottery sherds, and copper plates) with an **average inscription length of merely 5 signs**. 

This extreme brevity limits grammatical and contextual syntax. Rather than attempting speculative linguistic decipherment, this project pursues **computational epigraphy**—treating the inscriptions as an unknown, structured notational information system.

### 1.2 Objective of Milestone 1
The objective of Milestone 1 is to construct an end-to-end computer vision pipeline capable of:
1. Ingesting raw artifact imagery (photographs of seals and impressions).
2. Segmenting and localizing individual script signs with precise bounding boxes.
3. Classifying each detected glyph into its standardized concordance index (from Iravatham Mahadevan’s foundational IM-417 concordance or Asko Parpola's CISI system).
4. Serializing the signs into an ordered sequential reading array (supporting directional reading order, predominantly Right-to-Left).

### 1.3 The Core Computational Bottlenecks
- **Absence of Real Annotated Bounding Boxes:** No public dataset provides seal photos paired with bounding-box annotations around multi-sign sequences.
- **The Kaggle IM-417 Isolation:** The Kaggle IM-417-150 dataset offers only tightly cropped, segmented glyphs in isolation—divorced from seal bodies, background mineral grains, and multi-symbol linear arrangements.
- **Severe Class Imbalance:** The 417 concordance signs follow a steep Zipfian power-law distribution. A few dozen signs appear thousands of times, while dozens of rare signs appear fewer than 5 times across the entire archaeological record.
- **Artifact Degradation:** Physical seals exhibit intaglio relief (carved inward), erosion, salt encrustation, uneven lighting, edge chipping, and mineral grain textures (steatite, faience, terracotta, copper). Naive detection models fail catastrophically when presented with real archaeological noise.

---

## 2. In-Depth Comparative Analysis: YOLO vs. Faster R-CNN vs. RT-DETR

To provide empirical benchmarking for the academic milestone and select the optimal primary vision backbone, three distinct object detection architectures are implemented and compared.

```
                           ┌───────────────────────────────┐
                           │      Raw Seal Photograph      │
                           └───────────────┬───────────────┘
                                           │
             ┌─────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│     YOLO (v8 / v11)     │   │      Faster R-CNN       │   │         RT-DETR         │
│  Single-stage CNN Grid  │   │ Two-stage RPN + RoIAlign│   │ End-to-End Transformer  │
├─────────────────────────┤   ├─────────────────────────┤   ├─────────────────────────┤
│ • Speed: Ultra-fast     │   │ • Speed: Slower/Heavy   │   │ • Speed: Real-time (GPU)│
│ • Post-processing: NMS  │   │ • Post-processing: NMS  │   │ • Post-processing: NONE │
│ • Local context only    │   │ • Tailored RPN Anchors  │   │ • Global Self-Attention │
│ • Failure: Suppresses   │   │ • Failure: Suppresses   │   │ • Preserves touching    │
│   adjacent signs        │   │   adjacent signs        │   │   linear glyph tokens   │
├─────────────────────────┤   ├─────────────────────────┤   ├─────────────────────────┤
│ Role: Fast Testbed &    │   │ Role: High-Precision    │   │ Role: Primary Epigraphy │
│ Synthetic Debugger      │   │ Academic Baseline       │   │ Extraction Backbone     │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
```

### 2.1 Architecture 1: YOLO (YOLOv8 / YOLOv11)
- **Architectural Anatomy:** Single-stage detector with a CSPDarknet backbone, Path Aggregation Network (PANet) feature pyramid, and an anchor-free decoupled head separating object classification from bounding box coordinate regression. Coordinates are optimized via Distribution Focal Loss (DFL) and Complete Intersection-over-Union (CIoU) loss.
- **Strengths for the Project:**
  - **Blazing Fast Training & Inference:** YOLO models converge in a fraction of the time required by two-stage or transformer architectures.
  - **Rapid Prototyping & Synthetic Debugging:** YOLO serves as the **domain-shift quality discriminator** in our synthetic data generation feedback loop. If a YOLO model trained on synthetic seals achieves 98% mAP on synthetic validation but fails on real Harappa.com seal photos, the synthetic generation engine is flagged for domain shift (e.g., poor blending, unrealistic lighting, or inaccurate stone grain).
- **Critical Weakness (The NMS Epigraphic Failure):**
  - YOLO fundamentally relies on **Non-Maximum Suppression (NMS)** to eliminate overlapping candidate boxes.
  - On Indus seals, inscriptions average ~5 signs packed densely in a single line. Signs frequently touch, share stroke boundaries, or possess ligature modifiers.
  - Conventional IoU-based NMS frequently mistakes closely situated adjacent glyphs as duplicate predictions and suppresses the neighboring sign. Dropping even a single sign completely breaks downstream n-gram transition modeling (Milestone 2) and masked token reconstruction (Milestone 3).

### 2.2 Architecture 2: Faster R-CNN
- **Architectural Anatomy:** Two-stage detector. Stage 1 uses a Feature Pyramid Network (FPN) and a Region Proposal Network (RPN) that slides over convolutional feature maps to propose candidate Regions of Interest (RoIs) using predefined anchor boxes. Stage 2 applies RoIAlign to pool candidate regions into fixed spatial grids, feeding them into dense classification and bounding-box regression heads.
- **Strengths for the Project:**
  - **High Localization Accuracy:** RoIAlign avoids feature quantization, preserving exact boundary delineations for subtle, eroded stone relief.
  - **Custom Anchor Tuning:** RPN anchor aspect ratios and scales can be customized to match the morphological distributions of the 417 Mahadevan signs (e.g., 1:1 for squarish signs, 1:2 and 1:3 for elongated vertical signs like "fish" variants, 2:1 for horizontal accents).
  - **Benchmark Rubric Requirement:** Fulfills the project requirement of showcasing documented cross-validation accuracies across distinct paradigms.
- **Critical Weakness:**
  - Computationally heavy with slower inference latency.
  - RPN and RoI heads still depend on NMS post-processing heuristics, carrying the identical risk of dropping adjacent linear glyphs.

### 2.3 Architecture 3: RT-DETR (Real-Time DEtection TRansformer)
- **Architectural Anatomy:**
  - **Backbone & Efficient Hybrid Encoder:** Employs an Attention-based Intra-scale Feature Interaction (**AIFI**) module exclusively on high-level feature maps (where global semantic self-attention is most effective) while using Cross-scale Feature Fusion Modules (**CCFM**) based on convolutional blocks for lower scales, bypassing the prohibitive computational cost of vanilla DETR.
  - **Query-Based Decoder:** A Transformer decoder utilizes a fixed number of learnable positional queries to interact directly with multi-scale image features.
  - **Direct Set Prediction & Hungarian Loss:** Replaces NMS entirely with the **Hungarian Bipartite Matching Algorithm**, computing an optimal 1-to-1 matching between predicted bounding boxes and ground-truth targets via an explicit permutation loss:
    $$\mathcal{L}_{\text{match}}(y, \hat{y}) = -\sum_{i=1}^{N} \left[ \mathbb{1}_{\{c_i \neq \emptyset\}} \log \hat{p}_{\sigma(i)}(c_i) + \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)}) \right]$$
- **Why RT-DETR is the Strategic Choice for Indus Epigraphy:**
  1. **Completely NMS-Free:** By predicting a set of distinct tokens simultaneously through 1-to-1 bipartite matching, RT-DETR never eliminates adjacent glyphs based on IoU overlap thresholds. Touching or ligatured Indus signs remain preserved.
  2. **Global Relational Context:** The multi-head self-attention mechanism processes the entire seal simultaneously. Unlike localized CNN kernels, RT-DETR understands the spatial relationship between tokens across a linear register, treating the inscription as a unified notational sequence rather than disconnected objects.
  3. **Architectural Symmetry with Downstream Milestones:** Milestones 2 and 3 treat the Indus script as a sequential token grammar (Markov transition matrices and TinyBERT masked sequence encoders). Employing a Transformer vision backbone in Milestone 1 establishes a coherent, end-to-end Transformer architecture from pixel-level grapheme extraction to semantic guardrails.

---

## 3. Why & How Synthetic Data is Required

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    THE ARCHAEOLOGICAL DATA BOTTLENECK                          │
│                                                                                │
│  1. IM77 & CISI: Pure text concordance strings (NO bounding boxes)             │
│  2. IM-417 (Kaggle): Isolated single-glyph crops (NO seal bodies/backgrounds)  │
│  3. Harappa.com: Raw artifact photographs (NO annotations)                     │
│  4. Zipfian Distribution: 417 signs, high rarity in tail classes               │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                   SYNTHETIC DATA GENERATION ENGINE (HOW)                       │
│                                                                                │
│  ┌─────────────────────────┐       ┌─────────────────────────┐                 │
│  │ Authentic Texture Bank  │       │ CISI/IM77 N-Gram Corpus │                 │
│  │ (Steatite/Clay/Copper)  │       │ (Real 3-7 sign strings) │                 │
│  └────────────┬────────────┘       └────────────┬────────────┘                 │
│               │                                 │                              │
│               └───────────────┬─────────────────┘                              │
│                               ▼                                                │
│                 [Register & Kerning Placement]                                 │
│                 (Upper third horizontal line)                                  │
│                               │                                                │
│                               ▼                                                │
│                 [Photometric Relief Simulation]                                │
│                 - Morphological chisel erosion/dilation                        │
│                 - Directional emboss & bump shading                            │
│                 - Poisson Seamless Cloning (cv2.seamlessClone)                 │
│                               │                                                │
│                               ▼                                                │
│                 [Geometric & Weathering Artifacts]                             │
│                 - Affine shear, perspective warp, ±5° rotation                 │
│                 - Surface noise, fractures, stone scratches                    │
│                               │                                                │
│                               ▼                                                │
│                 [Automated Ground-Truth Export]                                │
│                 - Zero-cost COCO JSON & YOLO label formats                     │
│                 - 2-4px edge safety padding                                    │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                      THE "REALITY CHECK" VALIDATION LOOP                       │
│                                                                                │
│  1. Train YOLO on 1,000 synthetic seals (fast convergence)                     │
│  2. Evaluate strictly on Real Benchmark Split (50-100 real Harappa.com seals)   │
│  3. Tune synthetic parameters (lighting, erosion, blur) to close domain gap    │
│  4. Scale to 3,000-5,000 images & train RT-DETR and Faster R-CNN               │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Why Synthetic Data is Mandatory
1. **The Ground-Truth Void:** While archaeological catalogs (Mahadevan IM77, Parpola CISI) document thousands of inscriptions, they exist solely as textual indices or drawn iconographic concordances. No large-scale dataset exists containing multi-sign real seal photographs paired with bounding-box annotations.
2. **Kaggle IM-417-150 Limitations:** This dataset contains isolated, segmented glyph images. If trained directly on these crops, a model can only perform classification on pre-cropped patches—it cannot detect or localize signs on an unprocessed seal photograph.
3. **Data Volume Demands of Transformers:** RT-DETR requires thousands of diverse training instances to properly optimize its attention weights and avoid severe memorization/overfitting.
4. **Intaglio Archaeological Physics:** Indus inscriptions were carved (intaglio) into steatite, fired to create a glaze, or stamped onto wet clay. A naive composite (e.g., pasting white-background glyphs onto stone) produces stark edge discontinuities. Deep models immediately exploit these boundary artifacts and fail completely when evaluated on authentic, eroded artifacts.

### 3.2 How the Synthetic Generation Engine Works
The synthetic pipeline (`grapheme_extraction/synthetic_generator.py`) generates photorealistic, structurally valid seal images through four modular stages:

#### Step 1: Base Texture Bank Acquisition
- Curate 30–50 high-resolution texture plates representing real artifact substrates:
  - White-glazed carved steatite.
  - Terracotta and fired red clay.
  - Grey unbaked clay sealings.
  - Weathered copper tablet surfaces.

#### Step 2: Inscription Grammar & Kerning Placement
- Instead of placing signs randomly, the engine extracts verified character sequences (3 to 7 signs long) directly from the **CISI (Parpola JSON)** or **IM77 Concordance**.
- Glyphs are rendered horizontally across the upper third of the seal canvas (the standard historical register above animal motifs like unicorns, bulls, or elephants).
- Inter-character kerning is sampled from normal distributions ($\mu = 6\text{px}, \sigma = 2\text{px}$) to simulate realistic stone-carving spacing, including occasionally touching characters.

#### Step 3: Photometric & Intaglio Relief Simulation
To bridge the synthetic-to-real domain gap:
- **Morphological Chisel Variation:** Apply random morphological erosion ($k \in [1, 3]$) and dilation to simulate variable chisel depth, line width, and physical wear.
- **Directional Emboss / Bump Shading:** Convolve glyph masks with directional Sobel kernels to generate highlights and shadows corresponding to oblique archaeological raking light.
- **Poisson Seamless Blending:** Embed the glyph into the stone substrate using OpenCV's `cv2.seamlessClone` (MIXED_CLONE mode) or adaptive alpha blending ($0.75 \le \alpha \le 0.95$), allowing the stone's underlying grain and cracks to show through the incision strokes.

#### Step 4: Geometric Distortion & Weathering
- Apply random affine transformations: rotation ($\pm 3^\circ$ to $\pm 6^\circ$), slight perspective skew ($< 4\%$), and elastic deformation.
- Overlay artifact noise: synthetic hairline fractures, salt pitting, and subtle Gaussian blur ($\sigma \in [0.4, 1.0]$) to emulate focal depth variations in museum photography.

#### Step 5: Automated Ground-Truth Annotation Export
- Because the placement coordinate $(x_i, y_i, w_i, h_i)$ of every glyph crop is deterministically recorded during compositing, annotations are generated automatically with zero human labeling cost.
- Apply a **2–4 pixel safety padding** around bounding boxes to ensure stroke terminals and accents are not clipped.
- Automatically serialize targets into:
  - **COCO JSON format** (`[x_min, y_min, width, height]`) for Faster R-CNN and PyTorch evaluation.
  - **YOLO text format** (`class_id x_center y_center width height` normalized to $[0, 1]$) for YOLO and RT-DETR.

### 3.3 The "Reality Check" Holdout Benchmark
- **The Golden Rule:** Never validate or report model accuracy solely on synthetic data.
- **Holdout Set:** Curate 50–100 authentic seal photographs from Harappa.com and CISI.
- **Annotation:** Bootstrap annotations using an automated proposal pipeline (OpenCV contour thresholding / zero-shot Grounding DINO) followed by manual refinement in CVAT or Label Studio.
- **Domain Gap Feedback Loop:**
  1. Train YOLOv8 on 1,000 synthetic seals.
  2. Measure mean Average Precision ($m\text{AP}@50$) strictly against the Real Benchmark Split.
  3. If detection recall on real seals is low, adjust synthetic generation hyperparameters (increase stone noise, alter emboss direction, adjust Poisson blending opacity).
  4. Once YOLO verifies high transferability, scale synthetic production to 3,000–5,000 images and train RT-DETR and Faster R-CNN.

---

## 4. Detection Pipeline Formulation: 2-Stage Modular Pipeline vs. Top-K

| Architectural Strategy | Description | Advantages | Disadvantages | Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **Option A: 2-Stage Modular Pipeline** | Stage 1: Detector localizes all signs as a single class (`indus_sign`). Stage 2: Cropped glyphs are classified into 417 IDs by a ResNet/ViT trained on IM-417. | Bypasses severe 417-class imbalance; detector focuses purely on intaglio boundary localization; classifier leverages full IM-417 Kaggle data. | Slight latency overhead from two sequential model passes. | **RECOMMENDED** |
| **Option B: Top-K Direct Multi-Class** | Train detector to classify the Top 25–50 most frequent signs directly, mapping all others to `unknown_sign`. | Single-pass end-to-end inference. | Fails on rare signs; ignores ~30% of valid concordance tokens; complicates downstream sequence modeling. | Comparative alternative |

### Reading Order & Sequence Serialization
Once bounding boxes $[b_1, b_2, \dots, b_k]$ and concordance sign IDs $[s_1, s_2, \dots, s_k]$ are extracted from an image:
1. Coordinates are sorted horizontally by centroid:
   - For **Right-to-Left (RTL)** reading direction (the dominant Indus script direction established by Lal, Parpola, and Mahadevan): sort descending by $x_{\text{center}}$.
   - For **Left-to-Right (LTR)** (e.g., seal impressions vs. seal matrices): sort ascending by $x_{\text{center}}$.
2. The pipeline outputs a standardized JSON payload:
   ```json
   {
     "artifact_id": "seal_harappa_042",
     "reading_direction": "RTL",
     "sign_count": 5,
     "sequence": [48, 342, 17, 99, 211],
     "bounding_boxes": [
       {"sign_id": 48, "confidence": 0.94, "bbox": [512, 45, 68, 92]},
       {"sign_id": 342, "confidence": 0.91, "bbox": [438, 48, 71, 90]},
       {"sign_id": 17, "confidence": 0.88, "bbox": [362, 50, 65, 88]},
       {"sign_id": 99, "confidence": 0.93, "bbox": [290, 46, 69, 93]},
       {"sign_id": 211, "confidence": 0.87, "bbox": [220, 52, 64, 85]}
     ]
   }
   ```

---

## 5. System Architecture & Operational Environment

### 5.1 Environment Configuration (`uv` + `AI/`)
Per engineering standards, all dependencies are managed inside the dedicated `AI/` virtual environment using `uv`:
```bash
# Activation
source AI/bin/activate

# Package management via uv
uv pip install -r requirements.txt
```

### 5.2 Dependency Manifest (`requirements.txt`)
- `torch` & `torchvision`: Deep learning backbones and CUDA execution.
- `ultralytics`: Provides optimized implementations of both YOLO (v8/v11) and RT-DETR.
- `opencv-python`: Image manipulation, morphological filters, Poisson seamless cloning (`cv2.seamlessClone`).
- `albumentations`: Specialized photometric and geometric augmentation pipelines.
- `scikit-learn`, `numpy`, `scipy`: Statistical evaluations and matrix operations.
- `transformers`: Downstream token encoders and vision transformer backbones.
- `matplotlib`, `seaborn`: Loss curves, PR-curves, and confusion matrix visualizations.

### 5.3 Repository File Layout for Milestone 1
```
aiproj/
├── AI/                                # uv virtual environment
├── data/
│   ├── raw/                           # Gitignored raw artifacts
│   │   ├── textures/                  # 30-50 stone/clay backgrounds
│   │   ├── im417/                     # Kaggle IM-417-150 sign crops
│   │   └── real_benchmark/            # 50-100 real Harappa.com seal photos + annotations
│   ├── synthetic/
│   │   ├── images/                    # Generated synthetic seal images
│   │   ├── labels_coco/               # COCO JSON annotations
│   │   └── labels_yolo/               # YOLO txt annotations
│   └── README.md
├── docs/
│   └── PRD_Milestone_1_Grapheme_Extraction.md
├── grapheme_extraction/
│   ├── synthetic_generator.py         # Synthetic engine (Poisson blend + COCO/YOLO export)
│   ├── train_yolo.py                  # YOLOv8/v11 training & synthetic discriminator
│   ├── train_faster_rcnn.py           # Faster R-CNN baseline training script
│   ├── train_rtdetr.py                # RT-DETR primary model training script
│   ├── classifier.py                  # IM-417 crop sign classifier (ResNet/ViT)
│   ├── pipeline.py                    # Unified extraction pipeline (Image -> Sign IDs)
│   └── utils.py                       # Bounding box drawing, format conversions, RTL sorting
├── notebooks/
│   ├── 01_synthetic_data_validation.ipynb
│   ├── 02_model_benchmarking_yolo_rcnn_rtdetr.ipynb
│   └── 03_real_seal_inference_demo.ipynb
├── reports/
│   └── milestone_1_comparative_report.md
├── requirements.txt
└── README.md
```

---

## 6. Quantitative Evaluation Metrics & Success Criteria

| Evaluation Metric | Target Benchmark | Justification & Epigraphic Significance |
| :--- | :--- | :--- |
| **mAP@50 (Real Benchmark)** | $\ge 0.85$ | Ensures high sign detection precision on unseen authentic archaeological artifacts. |
| **mAP@50-95 (Real Benchmark)** | $\ge 0.65$ | Measures tight bounding-box localization without clipping stroke ascenders/descenders. |
| **Sign Separation Recall (SSR)** | $\ge 0.90$ | Percentage of contiguous/touching adjacent signs correctly isolated into separate boxes without NMS deletion. |
| **Sequence Edit Distance (Levenshtein)** | $\le 0.15$ | Measures sequential token error rate when comparing extracted sign IDs against ground-truth concordance sequence. |
| **Inference Latency** | $< 50\text{ms / image}$ | Real-time interactive processing speed on modern GPU hardware. |

---

## 7. Risk Analysis & Mitigation Strategies

| Risk Description | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Synthetic-to-Real Domain Shift** (Model overfits to artificial edges) | High | Critical | Use Poisson seamless cloning (`cv2.seamlessClone`), stone-grain alpha blending, random blur, and use YOLO as an early domain-gap discriminator. |
| **Touching Sign Merging / Suppression** | High | Critical | Adopt RT-DETR (NMS-free bipartite matching); tune IoU thresholds on baseline models; use 2-stage crop separation. |
| **Class Imbalance on Rare Concordance Signs** | High | Moderate | Employ 2-stage architecture: single-class `indus_sign` localization + separate IM-417 classifier with class-weighted cross-entropy / focal loss. |
| **Inverted Inscriptions (Seal Matrix vs. Impression)** | Medium | Moderate | Add explicit reading-direction flag (`RTL` vs. `LTR`) in the pipeline; train models on horizontally flipped variants to maintain invariance. |

---

## 8. Milestone 1 Acceptance Checklist

- [ ] Synthetic generator creates photorealistic composite seals with authentic CISI sequence strings and intaglio relief simulation.
- [ ] Real benchmark dataset (50+ seal photos from Harappa.com) curated and annotated with ground truth.
- [ ] YOLOv8/v11 trained and evaluated on real benchmark to validate synthetic realism.
- [ ] Faster R-CNN trained with customized RPN anchor ratios as comparative baseline.
- [ ] RT-DETR trained and verified to resolve contiguous/touching glyphs without NMS suppression.
- [ ] Comprehensive comparative benchmark table (YOLO vs. Faster R-CNN vs. RT-DETR) generated across mAP, SSR, and latency.
- [ ] Unified Python pipeline (`pipeline.py`) accepts raw seal image, extracts glyphs, classifies IDs, and outputs ordered concordance array.
