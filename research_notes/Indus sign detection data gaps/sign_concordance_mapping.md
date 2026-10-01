# Machine-readable concordance between Mahadevan (M77), Parpola (P) and Wells/Fuls (ICIT) Indus sign numbers — state as of 1 Oct 2026

Headline: a downloadable, machine-readable Parpola -> Mahadevan (and Parpola -> Wells) mapping **already exists inside the repo the student uses** (`mayig/indus-valley-script-corpus`, `features/P*.json`, field `mahadevan_graphemes`). It is hand-made by one author, unsourced and incomplete (334 of 396 P signs mapped), but it covers 97% of the tokens in that repo's corpus. The scholarly concordances (Fuls & Wells 2023 catalogue, ICIT database) exist only in print or behind a login. Separately, a full Mahadevan-numbered corpus (2,906 texts) is now on GitHub, so the P->M step can be avoided altogether.

Figures marked "(computed)" are my own counts over a clone of `mayig/indus-valley-script-corpus` at commit `ad2f1e2` (16 Apr 2025), not numbers published by anyone.

## 1. Does mayig/indus-valley-script-corpus contain a Mahadevan cross-reference? How many artefacts does it cover?

### Takeaway
Yes. Every `features/Pnnn.json` file has `parpola_graphemes`, `wells_graphemes` and `mahadevan_graphemes` arrays, so the repo is itself a P <-> M <-> W lookup table in JSON. The corpus is small: 179 artefact files, all Mohenjo-daro (M-1 to about M-184), 1,003 sign tokens.

### Cited Findings
- Each feature file carries the three lists. Example `P086`: `parpola_graphemes` V126, V127, V151, V508; `wells_graphemes` W390, W392, W405, W406, W407, W408; `mahadevan_graphemes` M161, M162, M167, M168, M169 — [features/P086.json](https://github.com/mayig/indus-valley-script-corpus/blob/main/features/P086.json)
- The Rust struct documents the field as "a correspondence list of graphemes from Mahadevan's 1977 sign list matching this Parpola sign" (`pub mahadevan_graphemes: Vec<String>`) — [indus_corpus/src/lib.rs](https://github.com/mayig/indus-valley-script-corpus/blob/main/indus_corpus/src/lib.rs)
- The README mentions only the Wells mapping, not the Mahadevan one: "I have also included Wells's sign list from Wells (2015), matching as closely as I could with the Parpola sign list. Where there were multiple signs from Wells that mapped to a single sign from Parpola, I ensured that there was an associated feature to distinguish between them." No source or method is given for the Mahadevan column — [README](https://github.com/mayig/indus-valley-script-corpus/blob/main/README.md)
- The P numbering is "the most inclusive allographs provided by Parpola (1982)", not the 1994 book list; the README calls the repo "currently WIP" and licenses it MIT and/or Apache — [README](https://github.com/mayig/indus-valley-script-corpus/blob/main/README.md)
- The README warns that Parpola's numeral-stroke signs duplicate each other: P121/P144, P122/P145, P123/P147, P124/P150; the author uses P144+ for full-height strokes and P121+ for partial-height — [README](https://github.com/mayig/indus-valley-script-corpus/blob/main/README.md)
- (computed) 396 feature files, P000-P396, plus `default_features.json`. P000 is not a sign: "Represents a section of significant damage or lost material" — [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features)
- (computed) Corpus: 96 files in `corpus/m001_m099` + 83 in `corpus/m100_m199` = 179 artefact files, 179 sides, 1,003 grapheme tokens, 182 distinct P IDs in use — [corpus/](https://github.com/mayig/indus-valley-script-corpus/tree/main/corpus)
- (computed) The repo has a single commit, dated 16 Apr 2025 (message "m184"); nothing has been pushed since — [repo](https://github.com/mayig/indus-valley-script-corpus)
- A third-party project that parsed this repo reports the same size: "`data/cisi_real_corpus.csv` (179 inscriptions, 142 signs) — parsed from mayig/indus-valley-script-corpus" — [joyboseroy/indus_decipher](https://github.com/joyboseroy/indus_decipher)
- Spot checks of well-known signs are correct: P324 "Classic jar symbol" -> M342 / W740; P050 "Fish with no other decoration" -> M059 / W220; P013 person -> M001 / W090; P385 diamond -> M267; P145 two full strokes -> M087; P098 -> M176; P217 -> M211; P310 simple U -> M328 — [features/P324.json](https://github.com/mayig/indus-valley-script-corpus/blob/main/features/P324.json), [features/P050.json](https://github.com/mayig/indus-valley-script-corpus/blob/main/features/P050.json)

### Inferences
- The student's stated blocker ("without a P->M mapping the real sequences cannot be rendered") is already solved by data in the same repo; it just is not advertised in the README.
- Because the Mahadevan column has no stated source and comes from one person in one commit, treat it as a working hypothesis, not a scholarly concordance. The 11 spot checks above all pass, which suggests it was done carefully from the printed sign lists.
- The Rust code validates uniqueness of Wells IDs across files but (from the grep) has no equivalent check for Mahadevan IDs, which is consistent with 13 M numbers appearing under two P signs.
- 179 artefacts is roughly 4% of the 4,660 inscribed artefacts in ICIT; the repo is a demo-scale corpus, not a full CISI digitisation.

### Gaps
- Which printed source the author used for P->M (Parpola 1994 has no such table that I could confirm; possibly Fuls & Wells 2023, possibly by eye) is unstated.
- I did not diff the 58 missing M numbers against the printed lists to tell "no Parpola equivalent" from "author has not got to it yet".

## 2. Do the printed sign lists (Mahadevan 1977, Parpola 1994, Wells 2011/2015, Fuls ICIT) publish a concordance, and is any of it online?

### Takeaway
A proper scholarly concordance exists in print: Fuls & Wells, "A Catalog of Indus Signs" (2023), which contains a concordance to Mahadevan's list and one to Parpola's. It is not downloadable as data. The ICIT website is behind HTTP Basic authentication.

### Cited Findings
- Wells built his sign list "without reference to Mahadevan or Parpola's earlier ones for which a concordance was added later as a quick reference for scholars still using those sign lists" — [harappa.com, A Catalog of Indus Signs II](https://www.harappa.com/content/catalog-indus-signs-ii) (page returned 403 to my fetcher; text taken from the search-result extract)
- "A Catalog of Indus Signs" was self-published July 2023, ISBN 979-8-398-42230-6, and includes a concordance with Mahadevan's sign list and a concordance with Parpola's sign list — [ResearchGate record](https://www.researchgate.net/publication/373522673_A_Catalog_of_Indus_Signs) (403 to my fetcher; from search extract). Also listed at [Academia.edu](https://www.academia.edu/103538728/A_Catalog_of_Indus_Signs)
- Sign-list versions: Wells (2015) list updated to 709 signs (v2.8, May 2022) and 713 signs (v2.9, May 2023) — [ResearchGate, Catalog of Indus Signs](https://www.researchgate.net/publication/373522673_A_Catalog_of_Indus_Signs); [ResearchGate, Corpus of Indus Inscriptions](https://www.researchgate.net/publication/373522666_Corpus_of_Indus_Inscriptions)
- ICIT holds 4,660 inscribed artefacts and 5,644 texts with 17,957 legible signs; a concordance of ICIT inscriptions with Mahadevan's IDF80 texts was made with Rama Viswanathan, "making it possible ... to provide a concordance with Mahadevan's sign list" — [ResearchGate, Corpus of Indus Inscriptions](https://www.researchgate.net/publication/373522666_Corpus_of_Indus_Inscriptions) (search extract)
- Comparative sizes quoted in the same source: Mahadevan 419 signs / 459 variants; Parpola (1982) 394 signs / 281 variants; Wells 1999 610 signs; Fuls 2022 709 signs — [ResearchGate, Corpus of Indus Inscriptions](https://www.researchgate.net/publication/373522666_Corpus_of_Indus_Inscriptions) (search extract)
- `https://www.indus.epigraphica.de/` answers `HTTP/1.1 401 Unauthorized`, `WWW-Authenticate: Basic realm="Enter the general ICIT user connection details"` (checked 1 Oct 2026) — [indus.epigraphica.de](https://www.indus.epigraphica.de/)
- Mahadevan's sign list is 417 signs, with two added in a late addendum (419) — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)
- Other published counts: Parpola 386 (1994), Wells 694 (2015), Wells 676 (2006) — [Yadav & Vahia, Indus Script: A Study of its Sign Design](https://www.harappa.com/sites/default/files/pdf/Indus-sign-design.pdf); [arXiv 1812.00049](https://arxiv.org/pdf/1812.00049)
- One open project states flatly: "Wells's corpus does not exist as data — published aggregate statistics only" — [isramos/indus-script-sign-merging](https://github.com/isramos/indus-script-sign-merging)

### Inferences
- Status by source: Fuls & Wells 2023 = concordance exists, print only (buy the book or request the ResearchGate full text). ICIT = concordance exists in the database, login required (credentials are given to researchers on request). Mahadevan 1977 = cannot contain a Parpola concordance for the 1982/1994 lists because it predates them. Parpola 1994 = I found no evidence either way of an M-number column in its sign list.
- The cheapest route to a scholarly-grade table is to transcribe the two concordance tables from the 2023 catalogue (W->M and W->P), then join on W to get P<->M. That would also validate the mayig mapping.

### Gaps
- I could not open the catalogue itself, so the layout, page count and completeness of its concordance tables are unverified.
- Whether Parpola's 1994 sign list (Fig. 5.1) or CISI volumes print Mahadevan equivalents is unconfirmed.
- Whether Wells 2011 ("Epigraphic Approaches") or Wells 2015 contain a concordance appendix is unconfirmed; the harappa.com wording ("added later") suggests the concordance first appears in the 2023 catalogue.
- No Yadav/Vahia or Rao et al. paper was found that publishes a cross-system table; they work in M77 numbering only.

## 3. Do indusscript.in, Unicode documents, Wikipedia or GitHub/Kaggle/Hugging Face projects provide a digital mapping table?

### Takeaway
Apart from mayig's feature files, the only other machine-readable cross-system table found is a 145-entry ICIT/Wells -> Mahadevan "bridge" JSON in `paragorth/Indus-`. More useful for the student: that repo and `joyboseroy/indus_decipher` both ship the full Mahadevan-numbered corpus, which removes the need for P->M. No Unicode document or Wikipedia table gives a concordance.

### Cited Findings
- `paragorth/Indus-` contains `data/derived/bridge_extended.json` (2,070 bytes), described as "the extended sign bridge (143 signs → Mahadevan numbers)". Format is `{"740": [342], "390": [161, 162, 167, 168, 169], "2": [99, 100], ...}`, keys being Wells/ICIT numbers — [bridge_extended.json](https://github.com/paragorth/Indus-/blob/HEAD/data/derived/bridge_extended.json); [repo README](https://github.com/paragorth/Indus-)
- (computed) That bridge has 145 keys, 3 with empty lists. Compared with W->M derived through mayig's feature files: 125 identical, 13 different, 7 Wells numbers absent from mayig — [bridge_extended.json](https://github.com/paragorth/Indus-/blob/HEAD/data/derived/bridge_extended.json)
- `paragorth/Indus-` also ships the Mahadevan corpus: `data/im77/im77_corpus_lines.csv` (3,916 rows, 2,906 distinct text numbers, 14,153 sign tokens), `im77_sign_frequencies.csv` (token count for each of the 417 signs), `im77_field_symbols.csv`. "Pulled 23 Sept 2026 from indusscript.in ... Exported from the site's own database while signed in" — [IM77_README.md](https://github.com/paragorth/Indus-/blob/HEAD/data/im77/IM77_README.md)
- The same note says sign 0 marks an illegible/lost passage (781 occurrences), and that indusscript.in "holds a drawing of every sign at https://indusscript.in/assets/images/<sign>.png (0–417)" — [IM77_README.md](https://github.com/paragorth/Indus-/blob/HEAD/data/im77/IM77_README.md)
- `https://indusscript.in/assets/images/342.png` returned HTTP 200, `content-type: image/png`, without login (checked 1 Oct 2026) — [indusscript.in](https://indusscript.in/assets/images/342.png)
- GitHub reports the paragorth licence as "Other / NOASSERTION"; repo created 24 Sep 2026, last push 1 Oct 2026 — [paragorth/Indus-](https://github.com/paragorth/Indus-)
- `joyboseroy/indus_decipher` ships `data/m77_indusscript_real_corpus.csv` (3,573 lines, 418 signs, "the actual canonical M77/IDF-80 corpus", obtained from indusscript.in via a logged-in browser session), `data/indus_website_real_corpus.csv` (2,543 inscriptions, 592 signs, from the yajnadevam/indus-website MySQL dump) and the mayig-derived CISI file. It has no sign-number concordance; its "CISI concordance crosswalk table" is "a pure identifier crosswalk" of object numbers — [joyboseroy/indus_decipher](https://github.com/joyboseroy/indus_decipher)
- `isramos/indus-script-sign-merging` works with Wells IDs (`W017`) and corpus codes (`G17`) from the vendored mayig data and ships no cross-catalogue file — [isramos/indus-script-sign-merging](https://github.com/isramos/indus-script-sign-merging)
- Unicode: Michael Everson's 1999 proposal was never approved; the script is not encoded — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)
- The only recent Unicode-related document found is Devajyoti Sarkar, "Unicode Encoding Proposal for the Indus Script – Early Draft" (3 Apr 2023, CC-BY-4.0, 13 MB PDF), which proposes a Private Use Area encoding and is tied to the author's own decipherment claim; it is not a UTC/L2 document — [Zenodo 7794001](https://zenodo.org/records/7794001)
- Wikipedia's Indus script article has no cross-reference table between numbering systems — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)
- An arXiv paper notes that ICIT glyph numbering (G### prefix) is distinct from Mahadevan's M-codes (417) and Parpola's P-codes (~398) — [arXiv 2604.17828](https://arxiv.org/abs/2604.17828) (search extract)

### Inferences
- The paragorth bridge agrees with mayig on 125 of 138 comparable entries, including identical multi-value groups. It is very probably derived largely from mayig's files and then hand-edited, so it is not independent confirmation; the 13 disagreements are worth eyeballing as likely error sites in one or the other.
- Simplest path for the project: use `im77_corpus_lines.csv` (column `signs_clean`) as the sequence source. It is already in M numbers 1-417, is 14x larger than mayig's corpus, and needs no mapping. Keep mayig + P->M only if CISI object IDs or the allograph feature vectors are needed.
- Licence caution: both M77 exports were scraped from a login-gated RMRL site and neither repo carries a clear open licence for that data. Fine for a student project; cite Mahadevan 1977 / RMRL, and do not redistribute without checking.
- indusscript.in sign PNGs are a second, canonical source of M-numbered glyphs to cross-check the Kaggle labels.

### Gaps
- I could not text-extract the Sarkar PDF (no PDF tool available), so whether it tabulates M/P/ICIT equivalents is unknown.
- No formal L2 / WG2 proposal by Pandey or anyone else for Indus was found; the search was not exhaustive of the L2 register.
- indusscript.in was not inspected directly for a published sign-concordance page; the data pages need a login.
- Kaggle `storesource/im-417-150` was not inspected. Its folder/label convention (1-417 vs 0-416, zero-padding) must be checked: class 342 should be the jar and class 59 the plain fish. An off-by-one would corrupt every render silently.
- The yajnadevam/indus-website dump (592 signs, ICIT-style numbering) was not inspected for an M-number column (GitHub API rate limit).

## 4. Do hellosindh/indus-script-synthetic and the "AI-Epigraphy" toolkit use Mahadevan numbering or ship a mapping?

### Takeaway
Neither uses Mahadevan numbers and neither ships a mapping. hellosindh uses its own `T###` IDs over 715 signs; AI-Epigraphy's public code has a 45-row CSV with internal `IVC_001`-style IDs.

### Cited Findings
- hellosindh dataset files are `synthetic_indus_5k.csv` and `reference/sign_index.json`; licence CC-BY-4.0; last modified 10 Apr 2026 — [HF dataset](https://huggingface.co/datasets/hellosindh/indus-script-synthetic)
- `sign_index.json` declares `"total_signs": 715`, `"sign_id_format": "T{numeric_id} — e.g. T638, T101, T604"`; each entry has `sign_id`, `glyph` (a private-use font character), `corpus_count`, `corpus_freq_pct`, `role`, start/end rates. No M, P or W field — [sign_index.json](https://huggingface.co/datasets/hellosindh/indus-script-synthetic/blob/main/reference/sign_index.json)
- The card says the models were trained on 3,310 real inscriptions with 641 signs in use, and names the files as `synthetic_indus_5k.jsonl` and `sign_index.json` "All 641 signs", which does not match the actual files (`.csv`, 715 signs). Its grep for Mahadevan/Parpola/Wells/ICIT/Fuls returned nothing — [dataset card](https://huggingface.co/datasets/hellosindh/indus-script-synthetic/blob/main/README.md)
- T-IDs go well above 417 (T638, T741, T701), so they cannot be Mahadevan numbers — [dataset card](https://huggingface.co/datasets/hellosindh/indus-script-synthetic/blob/main/README.md)
- AI-EPIGRAPHY is an IndiaHCI 2025 paper/tool combining n-gram/collocation statistics with ML; code at `atulsharma0071/indiahci2025` — [ACM DL](https://dl.acm.org/doi/10.1145/3768633.3770145)
- That repo contains `indus - symbols.csv` (45 symbols; columns No., Inscription, Reading, Site, Meaning/Interpretation, Period; IDs like `IVC_001`), `hciup.py` and a README; no reference to M77 or ICIT numbering and no mapping file — [atulsharma0071/indiahci2025](https://github.com/atulsharma0071/indiahci2025)

### Inferences
- The most frequent hellosindh sign T638 at about 10% of tokens and usually text-final behaves like the jar (M342 / W740), but T638 is neither number, so the T scheme is some third list. Joining it to M numbers would need its source list identified first. The synthetic sequences cannot be rendered with the Kaggle M-glyphs as things stand.
- AI-Epigraphy is irrelevant to the mapping problem.

### Gaps
- The origin of the T### scheme (715 signs) is not documented on the card; I did not identify it and will not guess.
- The AI-Epigraphy paper text (paywalled ACM) was not read; the paper may describe a larger corpus than the public repo holds.

## 5. How many-to-one / one-to-many is the mapping, and what share maps cleanly?

### Takeaway
In mayig's table about three quarters of Parpola signs map to exactly one Mahadevan sign, and 77% of real corpus tokens convert unambiguously. Roughly 20% of tokens hit a one-to-many case (mostly stroke numerals and tree/variant families) and need a tie-break; about 3% have no Mahadevan equivalent listed.

### Cited Findings
All counts below are (computed) over [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features) and [corpus/](https://github.com/mayig/indus-valley-script-corpus/tree/main/corpus) at commit `ad2f1e2`.
- P side: 396 P IDs. 334 have at least one M; 62 have an empty list (including P000, the damage marker) — [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features)
- M per P: 305 P signs -> 1 M; 24 -> 2 M; 3 -> 3 M; 2 -> 5 M — [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features)
- M side: 359 of 417 Mahadevan numbers are reachable; 58 are never referenced (6, 10, 22, 33, 39, 44, 45, 46, 49, 58, 63, 69, 148, 154, 184, 202, 205, 212, 213, 217, 218, 222, 229, 232, 240, 248, 260, 265, 269, 281, 291, 302, 315, 324, 331, 333, 337, 340, 353, 354, 359, 360, 361, 366, 368, 369, 370, 371, 378, 382, 385, 386, 389, 394, 396, 406, 411, 414) — [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features)
- P per M: 346 M signs appear under one P; 13 appear under two P (M002, M050, M064, M076, M099, M103, M161, M167, M168, M199, M216, M264, M409) — [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features)
- Strict one-to-one pairs (one M for the P and that M under no other P): 286, i.e. 72% of 396 P signs and 69% of 417 M signs — [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features)
- Wells side: 331 P signs have at least one W; 498 distinct W numbers are referenced, so P->W is far more one-to-many than P->M — [features/](https://github.com/mayig/indus-valley-script-corpus/tree/main/features)
- Token level, 1,003 tokens / 182 distinct P in the corpus: 153 P types with exactly one M = 768 tokens (77%); 20 P types with several M = 202 tokens (20%); 9 P types with no M = 33 tokens (3%): P000, P094, P166, P170, P201, P207, P358, P368, P381 — [corpus/](https://github.com/mayig/indus-valley-script-corpus/tree/main/corpus)
- Example of merging differences: Parpola P086 (tree) = 5 Mahadevan signs (161, 162, 167, 168, 169) = 6 Wells signs; Parpola P122 (two short strokes) = M099 + M100 — [features/P086.json](https://github.com/mayig/indus-valley-script-corpus/blob/main/features/P086.json), [features/P122.json](https://github.com/mayig/indus-valley-script-corpus/blob/main/features/P122.json)
- Published sign counts show the general direction of merging: Parpola 386-398 < Mahadevan 417-419 < Wells/Fuls 676-713 — [Yadav & Vahia](https://www.harappa.com/sites/default/files/pdf/Indus-sign-design.pdf); [ResearchGate, Corpus of Indus Inscriptions](https://www.researchgate.net/publication/373522666_Corpus_of_Indus_Inscriptions)

### Inferences
- P000 should map to Mahadevan's 0 (illegible), not count as unmapped. That leaves 8 genuinely unmapped sign types in use; render them as a placeholder glyph.
- Parpola merges more allographs than Mahadevan, so P->M is the lossy direction (one P, several M candidates). M->P is nearly a function (346 of 359 single-valued).
- Practical tie-break for one-to-many: pick the candidate M with the highest count in `im77_sign_frequencies.csv`. Using mayig's per-token feature vector to choose the exact allograph is possible but needs hand-written rules per sign; not worth it for 29 signs unless visual fidelity matters.
- Minimal loader (IDs are zero-padded strings `M059`; Kaggle labels are presumably plain ints 1-417):

```python
import json, glob, csv
freq = {int(r[0]): int(r[1]) for r in list(csv.reader(open("im77_sign_frequencies.csv")))[1:]}  # sign,tokens,doubtful_tokens
p2m = {}
for f in glob.glob("features/P*.json"):
    d = json.load(open(f))
    ms = [int(m[1:]) for m in d["mahadevan_graphemes"]]
    p2m[d["id"]] = max(ms, key=lambda m: freq.get(m, 0)) if ms else None   # None -> placeholder glyph
p2m["P000"] = 0                                                           # damage -> Mahadevan 'illegible'
assert p2m["P324"] == 342 and p2m["P050"] == 59                           # jar, fish
```

### Gaps
- These proportions describe one unverified hand-made table, not the scholarly consensus. The true figure from Fuls & Wells's printed concordance may differ.
- No published paper was found that quantifies M<->P agreement (share of signs mapping 1:1).
- Token-level figures come from 179 Mohenjo-daro seals only and will shift for other sites and object types.
- The loader was not run end to end. The header of `im77_sign_frequencies.csv` is `sign,tokens,doubtful_tokens` (checked), so the column order in the loader is right.
