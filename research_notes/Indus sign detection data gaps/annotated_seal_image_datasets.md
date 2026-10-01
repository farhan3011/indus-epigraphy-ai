# Annotated image datasets of real Indus seal photographs (state checked 2026-10-01)

Method note: GitHub repos were inspected through the GitHub REST API (`/repos/{r}` and `/git/trees/{branch}?recursive=1`), so file counts below are exact counts of the tree on the check date, not README claims. Hugging Face, Zenodo, figshare, Internet Archive, Wikimedia Commons and the Met were queried through their public APIs. Kaggle and Roboflow Universe were only checked through web search (no CLI/API access), so absence there is weaker evidence.

## 1. Does DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information (Dixit et al. 2025, JCAA 8(1)) contain the ~245 annotated images and annotation files?

### Takeaway
No. The repo holds code plus exactly three sample images and zero annotation files; the README says the full data must be requested from the authors. The 245 box-annotated images (232 train + 13 validation, XML) are described in the paper only.

### Cited Findings
- The whole repo is 10 files (about 590 KB), last pushed 2025-02-23, licence CC0-1.0. Full file list: `Grapheme_Identification/Source Code/Cross_Validation.ipynb`, `Motif_Identification/Source Code/MotifIdentification.ipynb`, `Training and Validation Phase/MA.py`, `MI.py`, `mobileNet.ipynb`, `LICENSE`, `README.md`, and three sample images — [GitHub tree API](https://api.github.com/repos/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/git/trees/main?recursive=1)
- The only image files are `Sample Data/Sample_Grapheme/Sample_Grapheme.jpg` (1.7 KB), `Sample Data/Sample_IVC_Seal/Sample_Seal.PNG` (285 KB) and `Sample Data/Sample_Motif/Sample_Swatik.PNG` (25 KB). There is no `.xml`, `.txt` label, `.json` or YOLO/COCO file anywhere in the tree — [GitHub tree API](https://api.github.com/repos/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/git/trees/main?recursive=1)
- README, verbatim: "The full data sets may be obtained from authors upon request. To facilitate understanding and reproducibility, we have provided sample data of each data type." — [README](https://github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/blob/main/README.md)
- README also says YOLOv3 is "a pre-trained model ... available at the reference provided"; no YOLO weights or config are in the repo — [README](https://github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/blob/main/README.md)
- Claimed in the paper, absent from the repo: 1,004 images collected from the CISI volumes and Mahadevan's compendium, 963 after deduplication; bounding-box annotation on 232 training + 13 validation images, stored as XML; the paper's data-availability statement points to this GitHub repo — [JCAA article](https://journal.caa-international.org/articles/10.5334/jcaa.175) (extracted through an automated page summary; re-check the 232/13 and 963 figures against the PDF before quoting)
- Classes: the grapheme classifier (MobileNet) has 40 classes labelled with Mahadevan numbers — M8, M12, M15, M17, M19, M28, M48, M51, M53, M59, M102, M104, M141, M162, M173, M174, M176, M204, M205, ... plus "Other". The label list is hard-coded on line 25 of `MA.py` — [MA.py](https://github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information/blob/main/Training%20and%20Validation%20Phase/MA.py); full list also in the [JCAA article](https://journal.caa-international.org/articles/10.5334/jcaa.175)
- Motif classes (11): buffalo, bull, elephant, horned ram, man holding tigers, Pashupati, sharp horn with long trunk, short-horned bull, swastika, tiger-man on tree, unicorn — [JCAA article](https://journal.caa-international.org/articles/10.5334/jcaa.175)
- The grapheme-classifier training set is 12,264 augmented crops across the 40 classes (more than 300 per class); also not in the repo — [JCAA article](https://journal.caa-international.org/articles/10.5334/jcaa.175)
- Harappa.com hosts a summary page for the paper — [Harappa.com](https://www.harappa.com/content/deep-learning-archiving-indus-script-and-motif-information)

### Inferences
- The YOLOv3 detector in the paper appears to be single-class (a generic "grapheme" box) with sign identity assigned afterwards by MobileNet; this matches the two-stage description and the fact that only 40 sign classes exist. I did not confirm the detector's class list from the paper text.
- Even if the authors share the data, it would give at most 245 boxed images with class-agnostic boxes and Mahadevan IDs for only about 39 frequent signs, not the full 417-sign inventory.
- The data is obtainable only by emailing the authors; nothing here can be scripted into a pipeline today.

### Gaps
- Annotation tool (LabelImg or other) and XML schema (Pascal VOC presumed) are not stated in what I retrieved.
- Whether the authors actually respond to data requests is unknown.
- A separate repo, `dixitsandeep/indus_seal_extraction`, belongs to a "Sandeep Dixit"; I could not confirm whether this is the same Dixit as the JCAA first author (see question 3).

## 2. What does tpsatish95/OCR-on-Indus-Seals (Palaniappan & Adhikari 2017, arXiv:1702.00523) contain?

### Takeaway
Code, trained Caffe models, and pipeline output images for about 64 seals; the 1,150-image "Indus Seals Dataset" and the region-label images are not in the repo. There are no ground-truth per-sign boxes at all, in the repo or in the paper's method.

### Cited Findings
- Repo is 1,299 tree entries, about 166 MB, last pushed 2017-02-04, licence Apache-2.0 — [GitHub repo API](https://api.github.com/repos/tpsatish95/OCR-on-Indus-Seals)
- Claimed in the paper, absent from the repo: the "Indus Seals Dataset" of 1,150 images = about 800 high-resolution scans of seals CISI M0101 to M0620 from the Roja Muthiah Research Library, plus 350 web images kept from 1,000 crawled through Google Custom Search (Harappa.com and the general web) — [paper draft in repo](https://github.com/tpsatish95/OCR-on-Indus-Seals/blob/master/paper/paper_draft.md); [arXiv:1702.00523](https://arxiv.org/abs/1702.00523)
- Claimed in the paper: a "Text/No-Text Dataset" of 2,091 selective-search region crops manually sorted into Text (652), NoText (1,055) and Both (384) — [paper draft](https://github.com/tpsatish95/OCR-on-Indus-Seals/blob/master/paper/paper_draft.md)
- Claimed in the paper: a "Symbols Dataset" that is only a binary image-level label on the 350 crawled images, 168 with a Jar sign and 182 without — [paper draft](https://github.com/tpsatish95/OCR-on-Indus-Seals/blob/master/paper/paper_draft.md)
- What is present for those labels: list files only. `code/ROIs_Indus/RMRL_model_update/Train.txt` (1,670 lines) and `Test.txt` (417 lines), plus an older `code/ROIs_Indus/Train.txt` (607) and `Test.txt` (257). Each line is `path label`, e.g. `Data_U_Full/Train/Both/1_sub_1065.jpg 2`; label counts in the 1,670-line file are 842 / 520 / 308 for labels 0 / 1 / 2. The `Data_U_Full/` images they point to are not in the repo — [Train.txt](https://github.com/tpsatish95/OCR-on-Indus-Seals/blob/master/code/ROIs_Indus/RMRL_model_update/Train.txt); [tree API](https://api.github.com/repos/tpsatish95/OCR-on-Indus-Seals/git/trees/master?recursive=1)
- Trained weights are present: `code/ROIs_Indus/Models/bvlc_googlenet_indusnet_iter_{8000,16600,20000}.caffemodel` (about 41 MB each) and `code/ROIs_Indus/bvlc_googlenet.caffemodel` (54 MB) — [tree API](https://api.github.com/repos/tpsatish95/OCR-on-Indus-Seals/git/trees/master?recursive=1)
- Image content actually present: `code/Test/Regions/<id>/` for 24 seal ids (bare numbers such as 28, 290, 1000) (2 to 6 jpg and 1 to 3 png each), `code/Test/Result/` (25 png + 5 in `Partial/`), `code/Test/patches/` (23 png), `code/Test/temp/` (48 png), and `paper/Results/{Perfect_Text,Partial_Text}/{Perfect_Symbols,Partial_Symbols,No_Symbols}/<seal>/` for 50 seal folders, most named by CISI id (e.g. `H-139 A`, `M-152 A`, `K 40 a_crop`, `L 1 a`, `C 2 a`) and ten by the same bare numbers used in `code/Test/Regions`, giving about 64 distinct seals overall, each with 1 to 6 jpg and 6 to 31 png — [tree API](https://api.github.com/repos/tpsatish95/OCR-on-Indus-Seals/git/trees/master?recursive=1)
- No annotation format exists in the tree: no XML, JSON, CSV or YOLO txt of box coordinates — [tree API](https://api.github.com/repos/tpsatish95/OCR-on-Indus-Seals/git/trees/master?recursive=1)
- The pipeline segments symbols with selective search plus a text/no-text CNN and heuristics, not with a detector trained on sign boxes — [paper draft](https://github.com/tpsatish95/OCR-on-Indus-Seals/blob/master/paper/paper_draft.md)
- A sibling repo `tpsatish95/OCR-On-Indus-Seals-Research-Paper` (about 23.5 MB, pushed 2017-02-19) holds the paper source — [GitHub search API](https://api.github.com/search/repositories?q=indus+seal)

### Inferences
- The images under `code/Test/Regions`, `code/Test/Result` and `paper/Results` are model outputs (proposed regions and cropped symbols), sorted by the authors into quality buckets ("Perfect_Symbols", "Partial_Symbols", "No_Symbols"). The "Perfect_Symbols" folders (about 34 seals) could be mined as weakly verified sign crops, most with CISI ids, but they are not box annotations and carry no sign ids.
- Label 2 is "Both" (shown by the path). Labels 0 and 1 are probably NoText and Text: 842 and 520 are close to 80 percent of the paper's 1,055 and 652. The 1,670 + 417 = 2,087 lines nearly match the paper's 2,091.
- No sign-numbering scheme is used anywhere in the released material; the paper mentions Mahadevan text mappings held at RMRL, but those were not released.

### Gaps
- I did not open the individual jpg/png files under `paper/Results`, so the exact content of each (crop vs overlay) is unconfirmed.
- The RMRL scans appear never to have been published anywhere; I found no mirror.

## 3. Are there other datasets (Kaggle, Hugging Face, Roboflow Universe, Zenodo, Mendeley Data, figshare, other GitHub repos) with Indus seal photographs and/or sign-level annotations?

### Takeaway
I found no public dataset anywhere that pairs real Indus seal photographs with per-sign bounding boxes or masks. The closest real-photo asset is `akksshhaay/Indus-Seal-Dataset` (135 seals, photographs plus a cropped text strip, no boxes); everything else is isolated sign drawings, text corpora, or code.

### Cited Findings
- `akksshhaay/Indus-Seal-Dataset`: 135 numbered folders, 139 seal photographs, 97 `*text.jpg` crops of the inscription strip, 91 `*symbolic.jpg` crops of the motif, plus `ExhaustiveDatabaseOfSeals.xlsx`; about 29 MB; last pushed 2021-02-02; no licence file. The README calls them "about 135 different processed Indus Seals"; "processed" here means cropped and re-saved (EXIF shows paint.net), not binarised: the sample I opened, `1/1.jpg` (438x428), is a colour photograph of a unicorn seal impression with a clear 12-or-so-sign inscription, and `1/1text.jpg` (429x90) is the inscription row cropped from it. Only folder 1 was viewed — [repo](https://github.com/akksshhaay/Indus-Seal-Dataset); [tree API](https://api.github.com/repos/akksshhaay/Indus-Seal-Dataset/git/trees/master?recursive=1)
- The xlsx columns are Image, Description, Text Image, Symbolic Image, Link, Image No; the Link values are Harappa.com page URLs (e.g. `https://www.harappa.com/seal/14.html`, `https://www.harappa.com/indus2/156.html`). It contains prose descriptions, not sign sequences or coordinates — [ExhaustiveDatabaseOfSeals.xlsx](https://github.com/akksshhaay/Indus-Seal-Dataset/blob/master/ExhaustiveDatabaseOfSeals.xlsx)
- `akksshhaay/Indus-Alphabates-Dataset`: 1,840 `.bmp` sign images in `Indus Dataset/`, about 5 MB, 2021, no licence; isolated signs, not seals — [tree API](https://api.github.com/repos/akksshhaay/Indus-Alphabates-Dataset/git/trees/master?recursive=1)
- `HimanshuAttri/IndusValleyScriptDataset`: 418 PNGs of single script characters "sourced from a book", about 1.3 MB, pushed 2025-07-07, no licence — [repo](https://github.com/HimanshuAttri/IndusValleyScriptDataset)
- `ejhong/rongo`: 417 Indus glyph PNGs in `data/glyphs/indus/` (and normalised copies) extracted from 12 Mahadevan sign-list page scans in `data/raw/mahadevan/`; sign drawings only — [tree API](https://api.github.com/repos/ejhong/rongo/git/trees/main?recursive=1)
- `dixitsandeep/indus_seal_extraction`: about 873 MB, pushed 2025-03-22, no licence. Contains `input/CorpusVol1.pdf` (30,027,528 bytes) and `input/CorpusVol2.pdf` (35,870,617 bytes), scripts that cut individual seal photos out of the PDF pages and OCR their catalogue ids, and 36 jpg + 36 Tesseract `.box` files in `training_data/training_input_images/` — [repo](https://github.com/dixitsandeep/indus_seal_extraction); [tree API](https://api.github.com/repos/dixitsandeep/indus_seal_extraction/git/trees/master?recursive=1)
- Those `.box` files are character boxes for the printed catalogue caption, not for Indus signs: e.g. one file reads `K`, `-`, `9`, `3`, `A`, `(`, `1`, `0`, `0`, `%`, `)` with coordinates, another `M - 1 B`; the matching jpg is a 281x73 caption crop — [example .box](https://github.com/dixitsandeep/indus_seal_extraction/blob/master/training_data/training_input_images/seal_page_356_seal_2_ocr_attempt_1.box)
- `epachayan/indus-script`: machine-readable Mohenjo-daro seal catalogues (Mackay 1938, Marshall 1931) as CSV/JSON with "photo-coded text features"; no image files; MIT; pushed 2026-09-20 — [repo](https://github.com/epachayan/indus-script)
- `piusryan/Indus-Keeladi-CNN-Deep-Learning-for-Ancient-Script-Analysis-`: classification pipeline with only 30 PNGs in `data/processed/train`; MIT; pushed 2026-08-27 — [repo](https://github.com/piusryan/Indus-Keeladi-CNN-Deep-Learning-for-Ancient-Script-Analysis-)
- `dmahesh2869/harappan-seal-analysis`: network-science analysis, 39 CSVs and 13 plot PNGs, no seal photographs — [repo](https://github.com/dmahesh2869/harappan-seal-analysis)
- Hugging Face: a dataset search for "indus", "harappa", "indus script", "indus valley", "indus seal" returns two Indus-script items plus one unrelated name match. `hellosindh/indus-script-synthetic` is text only (5,000 synthetic sign sequences in `synthetic_indus_5k.jsonl` and a 641-sign `sign_index.json`, CC-BY-4.0). `Zer0pa/Indus-Valley-lane-state` states "no image bytes admitted". `nbrahme/Indus_V3_DS` is gated and its file list is LLM benchmark CSVs, unrelated to the script — [HF API search](https://huggingface.co/api/datasets?search=indus); [hellosindh card](https://huggingface.co/datasets/hellosindh/indus-script-synthetic); [Zer0pa card](https://huggingface.co/datasets/Zer0pa/Indus-Valley-lane-state)
- Zenodo: API searches for "indus script", "indus seal", "harappan seal" return publications (PDF papers) and a font package ("The ida fonts", record 7804585); no image dataset of seals appeared in the top results. The only dataset-type hits were unrelated (a river-drought dataset, a 3D "Harappan pot") — [Zenodo API](https://zenodo.org/api/records?q=indus+seal)
- Kaggle: web search surfaces only `storesource/im-417-150` (417 sign drawings from Mahadevan's concordance) — [Kaggle IM-417-150](https://www.kaggle.com/datasets/storesource/im-417-150)
- Roboflow Universe: a web search for Indus script or Indus seal detection projects returned only marine-mammal "seal" datasets — [example unrelated hit](https://universe.roboflow.com/ali-mustofa-cxrgi/seal-detection)

### Inferences
- A real-photo validation set with per-sign boxes will have to be hand-annotated; no shortcut download exists.
- `akksshhaay/Indus-Seal-Dataset` is the fastest starting point: about 139 real photographs, already cloneable with `git clone`, and 97 of them come with a pre-cropped inscription strip that narrows where boxes need to be drawn. Sign ids would still need to be assigned by hand against the Mahadevan list.
- The `dixitsandeep` scripts are a ready-made way to cut CISI plate pages into individual seal photographs tagged with their CISI id (e.g. `M-1 B`), which is the join key to Mahadevan-numbered transcriptions.

### Gaps
- Kaggle and Roboflow were checked only through a general web search; both sites need a logged-in search to be sure. Small student uploads may exist that search engines do not index.
- Mendeley Data was not searched. The figshare API query returned no results, which I could not distinguish from a malformed query.
- A Springer chapter titled "Automated Identification of Allographs Among the Indus Valley Script Signs" (https://link.springer.com/chapter/10.1007/978-3-032-06744-9_19) exists but redirected to a login page; its dataset, year and data-availability statement are unread.
- Zenodo record 19597147 ("Multi-Model Deep Generative Restoration of Harappan Seals Using Advanced Architectures", listed as a publication) was not opened; it may reference a seal image set.
- Several 2026 GitHub repos (`SoylentAquamarine/indus-script-collective`, `joyboseroy/indus_decipher`, `sriharideveloper/indus`, `isramos/indus-script-sign-merging`) were seen in search results but not opened; their descriptions indicate text/statistics work, not images.

## 4. Which sources offer unannotated real seal photographs in bulk that could be hand-annotated?

### Takeaway
The CISI volume scans on the Internet Archive are by far the largest freely downloadable pool (thousands of seal photographs, each printed with its CISI id). Wikimedia Commons (about 135 files) and the Met (a handful) are small; Harappa.com has good photographs but blocks scripted access.

### Cited Findings
- Internet Archive item `corpus-vol-2` ("Corpus of Indus seals and Inscriptions Vol 1"): `CorpusVol1.pdf` 30.0 MB and `CorpusVol1_jp2.zip` 32.8 MB of page images; uploaded 2024-09-20; not access-restricted — [archive.org item](https://archive.org/details/corpus-vol-2); [metadata API](https://archive.org/metadata/corpus-vol-2)
- Internet Archive item `corpus-vol-2_202409` ("... Vol 2"): `CorpusVol2.pdf` 35.9 MB and `CorpusVol2_jp2.zip` 38.7 MB; not access-restricted — [archive.org item](https://archive.org/details/corpus-vol-2_202409); [metadata API](https://archive.org/metadata/corpus-vol-2_202409)
- Internet Archive item `TheIndusScript.TextConcordanceAndTablesIravathanMahadevan` (about 1.76 GB) bundles three scans: "Collections in India" (PDF 31.5 MB, `_jp2.zip` 194 MB), "Collections in Pakistan" (PDF 35,870,617 bytes, byte-identical in size to `CorpusVol2.pdf`; `_jp2.zip` 236 MB), and Mahadevan's "The Indus Script: Texts, Concordance and Tables" (PDF 36 MB, `_jp2.zip` 423 MB); not access-restricted — [archive.org item](https://archive.org/details/TheIndusScript.TextConcordanceAndTablesIravathanMahadevan); [metadata API](https://archive.org/metadata/TheIndusScript.TextConcordanceAndTablesIravathanMahadevan)
- Other Archive items exist for CISI Vol 3 part 1 (marked "Incomplete Volumes") and a "MASI 86 ... Collections in India Vol 1" scan — [archive.org search](https://archive.org/advancedsearch.php?q=%22indus+seals%22+AND+mediatype%3Atexts&fl%5B%5D=identifier&fl%5B%5D=title&rows=15&output=json)
- The `CorpusVol1.pdf` / `CorpusVol2.pdf` in `dixitsandeep/indus_seal_extraction` have byte sizes identical to the Archive files (30,027,528 and 35,870,617), and that repo has scripts to cut seals out of the pages — [tree API](https://api.github.com/repos/dixitsandeep/indus_seal_extraction/git/trees/master?recursive=1)
- Wikimedia Commons: `Category:Indus civilization seals` has 51 files and 6 subcategories; `Category:Indus script` has 58 files and 2 subcategories. Walking both trees gives 135 unique files (109 jpg, 3 jpeg, 20 png, 3 svg) across 9 categories, including `Indus civilization seal impressions` (14), `Unicorns on seals of the Indus Valley Civilization` (16), `Indus script tablets` (8), `Pashupati seal` (8) — [Commons category](https://commons.wikimedia.org/wiki/Category:Indus_civilization_seals); [Commons API](https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Indus_civilization_seals&cmlimit=500&cmtype=file%7Csubcat&format=json)
- Met Museum open-access API: `search?hasImages=true&q=Indus seal` returns 42 object ids; checked objects 324063 ("Stamp seal: buffalo with incense burner (?)", culture Indus) and 324064 ("Stamp seal fragment: unicorn and incense-burner (?), inscription", culture Indus) are flagged public domain with full-resolution `primaryImage` URLs; other hits are not Indus (e.g. 324070 is Gandhara) — [Met search API](https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=Indus+seal); [object 324064](https://collectionapi.metmuseum.org/public/collection/v1/objects/324064)
- Harappa.com has a "Seals" slide-subject listing with at least 5 pages and 28 scholar slideshows — [Harappa seals listing](https://www.harappa.com/category/slide-subject/seals); [slideshows](https://www.harappa.com/slideshows)
- Harappa.com returned a Cloudflare "Attention Required" block page to plain `curl` requests (with a browser user-agent) on 2026-10-01, for the seals listing and for `robots.txt` — [Harappa seals listing](https://www.harappa.com/category/slide-subject/seals)
- British Museum: Indus stamp seals from Mohenjo-daro and Harappa are photographed and listed (e.g. reg. no. 1947,0416.2, 1947,0416.5) on the BM Images picture-library site — [BM Images example](https://www.bmimages.com/preview.asp?image=00031576001); [BM Images Mohenjo-daro results](https://www.bmimages.com/results.asp?txtkeys1=mohenjo-daro)
- One British Museum Mohenjo-daro seal is also on Sketchfab as a 3D model — [Sketchfab](https://sketchfab.com/3d-models/seal-mohenjo-daro-british-museum-ea4f061f0a4e4d08b67ac1926877347c)

### Inferences
- Fetch recipes: Archive files download directly at `https://archive.org/download/<identifier>/<filename>`; Commons files can be listed with the `categorymembers` API and fetched via `Special:FilePath/<name>`; Met images come from the `primaryImage` field of each object record. All three need no login.
- CISI volumes 1 and 2 are photographic plates of several thousand inscribed objects, and each photo is captioned with the CISI id that Mahadevan/ICIT-style transcriptions key on. That makes them the only bulk source where a photo can be paired with a known sign sequence, which turns box annotation into "draw N boxes in a known order" rather than open-ended labelling. The JP2 page sets in the larger Archive item (194 to 236 MB) are several times bigger than those in the two single-volume items (33 to 39 MB) although the Pakistan PDF is the same file, so they are probably the higher-resolution page images; this was not checked by opening them.
- The Commons set mixes photographs of seals, modern impressions, museum display shots and a few drawings (png/svg), so the usable photo count is somewhat below 135.
- The Met's genuinely Indus seals number roughly ten (object ids clustered around 324062 to 324070), based on the two checked records plus the id pattern; I did not open every record.
- Harappa.com images would have to be collected manually or through a real browser session. The `akksshhaay` repo already contains about 139 images taken from Harappa.com pages (its xlsx links back to them).

### Gaps
- Exact number of seal photographs in CISI vols 1 to 3 was not counted; I did not download the PDFs.
- Harappa.com total seal-image count is unknown because the site blocked scripted access.
- British Museum collection-online counts and download terms were not retrieved; only the commercial BM Images site appeared in search.
- National Museum New Delhi (and museumsofindia.gov.in) and other museum APIs were not checked.
- Whether CISI scans on the Internet Archive are authorised uploads is a licensing question left to the other researcher; I only note that no licence is stated on those items.

## 5. For each source: directly downloadable without login, and how large?

### Takeaway
Everything that actually exists is downloadable without login except the two paper datasets that were never released (Dixit et al. on request only; Palaniappan and Adhikari not available). Sizes run from under 1 MB to about 1.8 GB.

### Cited Findings
| Source | Real photos | Per-sign boxes | Sign scheme | Login needed | Size | Stated licence |
|---|---|---|---|---|---|---|
| [DM-BiCLab repo](https://github.com/DM-BiCLab/Deep-Learning-in-Archiving-Indus-Script-and-Motif-Information) | 1 sample seal | none in repo (245 XML-annotated images claimed in [paper](https://journal.caa-international.org/articles/10.5334/jcaa.175)) | Mahadevan, 40 classes | no (full data: email authors) | 590 KB | CC0-1.0 |
| [tpsatish95/OCR-on-Indus-Seals](https://github.com/tpsatish95/OCR-on-Indus-Seals) | pipeline outputs for about 64 seals | none | none | no | 166 MB | Apache-2.0 |
| [akksshhaay/Indus-Seal-Dataset](https://github.com/akksshhaay/Indus-Seal-Dataset) | 139 photos, 97 text-strip crops | none | none | no | 29 MB | none stated |
| [dixitsandeep/indus_seal_extraction](https://github.com/dixitsandeep/indus_seal_extraction) | CISI vol 1 and 2 PDFs + extraction scripts | none (boxes are for caption text) | CISI ids only | no | 873 MB | none stated |
| [akksshhaay/Indus-Alphabates-Dataset](https://github.com/akksshhaay/Indus-Alphabates-Dataset) | no (1,840 sign bitmaps) | n/a | unstated | no | 5 MB | none stated |
| [HimanshuAttri/IndusValleyScriptDataset](https://github.com/HimanshuAttri/IndusValleyScriptDataset) | no (418 sign PNGs) | n/a | unstated | no | 1.3 MB | none stated |
| [ejhong/rongo](https://github.com/ejhong/rongo) | no (417 Mahadevan glyphs) | n/a | Mahadevan | no | 98 MB | none stated |
| [Kaggle IM-417-150](https://www.kaggle.com/datasets/storesource/im-417-150) | no (417 sign drawings) | n/a | Mahadevan | Kaggle account | small | see page |
| [HF hellosindh/indus-script-synthetic](https://huggingface.co/datasets/hellosindh/indus-script-synthetic) | no (text) | n/a | own 641-sign index | no | small | CC-BY-4.0 |
| [Archive: CISI vol 1](https://archive.org/details/corpus-vol-2) | yes, plates | none | CISI ids | no | 110 MB item | none stated |
| [Archive: CISI vol 2](https://archive.org/details/corpus-vol-2_202409) | yes, plates | none | CISI ids | no | 130 MB item | none stated |
| [Archive: Mahadevan + CISI bundle](https://archive.org/details/TheIndusScript.TextConcordanceAndTablesIravathanMahadevan) | yes, plates | none | CISI ids; Mahadevan concordance | no | 1.76 GB item | none stated |
| [Wikimedia Commons](https://commons.wikimedia.org/wiki/Category:Indus_civilization_seals) | about 135 files | none | none | no | not measured | per file |
| [Met Open Access](https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=Indus+seal) | about 10 Indus seals | none | none | no | not measured | public-domain flag per object |
| [Harappa.com](https://www.harappa.com/category/slide-subject/seals) | yes | none | none | no login, but bot-blocked | unknown | site terms |

- Repo sizes come from the GitHub API `size` field (KB); Archive sizes from the `item_size` field of each metadata record — [GitHub API example](https://api.github.com/repos/akksshhaay/Indus-Seal-Dataset); [Archive metadata example](https://archive.org/metadata/corpus-vol-2)

### Inferences
- Cheapest route to a real-image detection validation set: annotate about 100 to 150 images drawn from `akksshhaay/Indus-Seal-Dataset` (colour museum-style photos) plus CISI plate crops (black-and-white catalogue photos, with known CISI ids), then request the 245-image XML set from the Dixit et al. authors as a possible supplement.
- Because no released set uses per-sign boxes, any reported detector metric on real images will be on self-annotated data; that should be stated in the project write-up.

### Gaps
- Kaggle download size and licence for IM-417-150 were not re-fetched (the project already holds this set).
- Download sizes for Commons and Met image sets were not measured.
