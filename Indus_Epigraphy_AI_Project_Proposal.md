**Project Overview & Context**

The Indus Valley Script, which dates back to the Bronze Age (~2600–1900 BCE), remains one of the most enduring undeciphered writing systems in historical linguistics. Unlike scripts that were unlocked by multilingual records—such as Egyptian hieroglyphs via the Rosetta Stone or Mesopotamian cuneiform—the Indus script has no verified bilingual text to bridge it to a known language. Furthermore, the average inscription length is extremely short, averaging only five signs per artifact, which severely limits the contextual syntax available for traditional linguistic decoding.

Rather than attempting an unverified "decipherment," this project focuses on **computational epigraphy**. Students treat the Indus inscriptions as a structured, notational information system. By applying modern computer vision pipelines, statistical n-gram modeling, and masked sequence evaluation, student teams will analyze the underlying sequential regularities, directional biases, and structural syntax governing the script's composition.

By completing this computational epigraphy framework, students will demonstrate the ability to:

- **Deploy Deep Learning Vision Pipelines** to accurately segment, detect, and archive individual graphemes from archaeological artifact imagery.

ResearchGate +1

- **Model Sequential Transition Probabilities** using statistical n-grams and Markov models to identify directional syntax and positional constraints.

ResearchGate +1

- **Implement Character-Level Language Architectures** to quantify the structural entropy and perplexity of an unknown notation system.

ResearchGate +1

- **Construct Rigorous Verification Guardrails** using machine learning classifiers to isolate synthetic reconstructions from historical replicas.

Hugging Face +1

- **Manage Cascading Pipeline Engineering Frameworks**, analyzing how image digitization errors propagate down into downstream sequence analysis layers.

**Dataset & Library Resources**

Students must develop their computational models and extraction blocks using the following verified academic datasets and interactive script tools:

- **The Indus Script IM77 Corpus (RMRL):** The digitized web concordance of Iravatham Mahadevan's foundational corpus, capturing text indices, sign combinations, and initial pairing statistics. Clean tabular references are available via the [Indus Script IM77 Platform](https://indusscript.in/).
- **IM-417-150 Grapheme Dataset:** A specialized image repository containing segmented individual characters mapped directly to Mahadevan's standardized 417 sign concordance list. Image arrays can be cloned via the [Kaggle IM-417 Archive](https://www.kaggle.com/datasets/storesource/im-417-150).
- **Digital Corpus of Indus Valley Script (CISI):** A machine-readable JSON serialization of the Corpus of Indus Seals and Inscriptions compiled by Asko Parpola et al., designed for immediate consumption by python dictionary layers. Scripts are accessible via the [Parpola JSON GitHub Repository](https://github.com/mayig/indus-valley-script-corpus).
- **Indus Script Synthetic Corpus:** An advanced ensemble-verified dataset containing 5,000 novel synthetic symbol sequences cross-validated against historical seals using Transformer classifiers. The training configurations are hosted on the [Hugging Face Synthetic Script hub](https://huggingface.co/datasets/hellosindh/indus-script-synthetic).
- **AI-EPIGRAPHY Computational Toolkit:** An interactive architectural framework following Model-View-Controller patterns designed specifically for frequency distribution, collocation analysis, and transitional probabilities testing. The core open-source codebase is hosted via the AI-Epigraphy GitHub Source.

**Technical Architecture & Milestones**

**Milestone 1: Automated Visual Grapheme Extraction**

- **Objective:** Parse raw artifact images to extract, crop, and categorize individual script symbols.
- **Technical Task:** Students implement a deep learning vision backbone (such as a lightweight MobileNet or custom CNN layer in PyTorch) to detect bounding boxes around individual script symbols on seal photographs. The model will map visual characters directly to their standardized numerical index IDs from the concordance.
- **Deliverable:** An image pipeline script that inputs a photo of a seal inscription and outputs an ordered list of standardized sign IDs.

YouTube·matsciencechannel +1

**Milestone 2: Positional Bias & Markov Chain Modeling**

- **Objective:** Establish the direction of writing and calculate sign pairing constraints.
- **Technical Task:** Students write scripts to map sequential transition frequencies across the parsed text corpus. They construct a Markov chain matrix and evaluate directional asymmetries by comparing symbol distribution rules at text beginners versus text enders. Students will apply Witten-Bell smoothing algorithms to handle zero-probability values for unobserved pairings.
- **Deliverable:** A directional statistical matrix verifying right-to-left syntax rules and tracking sign collocation indices.

ResearchGate +2

**Milestone 3: Sequential Entropy & Masked Token Encoders**

- **Objective:** Ingest incomplete strings and accurately predict hidden tokens based on structural context.
- **Technical Task:** Students configure character-level sequence encoders (such as a masked n-gram block or a TinyBERT framework) trained directly on the text corpora. The network will evaluate conditional token probabilities to automatically predict missing signs within damaged sequences.
- **Deliverable:** A sequence-to-sequence inference module capable of executing context-aware token completions with validation confidence ratings.

Hugging Face +1

**Milestone 4: Benchmarking Semantic Hallucinations**

- **Objective:** Build validation guardrails to test and statistically disprove absolute language claims.
- **Technical Task:** Students evaluate proposed readings using a multinomial Naive Bayes classifier or a controlled verification harness. They construct an evaluation benchmark that plots generative perplexities against random noise baselines to prove the limits of AI semantic mapping over low-resource, short-text corpora.
- **Deliverable:** An empirical research report disproving arbitrary linguistic translations by evaluating confidence scoring metrics against empirical structural controls.

**Expected Deliverables**

**Core Software Artifacts**

1. **Epigraphic Parsing Engine Source Code:** A unified Python script processing an input seal image all the way through visualization panels to output positional n-gram probability reports.
2. **Model Notebook Files:** Documented training notebooks showcasing cross-validation accuracies, learning curves, and token prediction loss charts.