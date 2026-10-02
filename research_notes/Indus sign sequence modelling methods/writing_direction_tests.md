# Writing direction of the Indus script: archaeological evidence, statistical tests, and avoiding circularity (as of October 2026)

Sourcing conventions used below:
- **[fetched]** = read from the source page itself in this session.
- **[second-hand]** = taken from a search-engine summary or a tertiary page; the primary source was not opened (Harappa.com, PNAS and Springer all returned 403/auth walls; Mahadevan 1977, Parpola 1994, Possehl 1996 and Lal 1966 are not open access).
- **[local]** = computed in this session from the project's own file `data/raw/im77/im77_corpus_lines.csv` (third-party export of Mahadevan 1977, pinned commit, see `data/README.md`). These are not published numbers.

## Archaeological and epigraphic evidence for right-to-left writing

### Takeaway
Direction was settled by physical evidence, not statistics: stroke overlaps on incised pottery (Lal 1966), signs cramped or displaced to a second line at the left edge, and short texts that start at the right edge leaving blank space on the left. All of this refers to the impression (or direct-incised) view; the carved seal itself is the mirror image.

### Cited Findings
- [second-hand] B.B. Lal published in *Antiquity* in 1966 that the direction was right to left, from inscribed potsherds at Kalibangan with overlapping signs: the sign on the extreme left overrode the one to its right, which overrode the next one to the right, so the rightmost sign was incised first. Kalibangan was excavated under Lal between 1960-61 and 1969-70. — [search summary of Lal-related pages incl. Wikipedia "Kalibangan" / "B. B. Lal"](https://en.wikipedia.org/wiki/B._B._Lal)
- [second-hand] A Scribd-hosted 2-page document "The Direction of Harappan Writing" lists the same three arguments: overlapping lines on potsherds, character spacing on seals, and a long seal inscription that wraps around three sides. Author and date not visible. — [Scribd](https://www.scribd.com/document/151494844/The-Direction-of-Harappan-Writing)
- [fetched] "Most scholars agree that the Indus script was generally read from right to left, though some exceptions wherein the script is written left to right or in a boustrophedon mode are also known." — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)
- [fetched] "Although the script is undeciphered, the writing direction has been deduced from external evidence, such as instances of the symbols being compressed on the left side as if the writer is running out of space at the end of the row." Footnotes cited for this section: Mahadevan (1977) pp. 10-14; Possehl (1996) pp. 59-62; Possehl (2002) pp. 132-134. — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)
- [fetched] "In the case of seals, which create a mirror image impression on the clay or ceramic on which the seal is affixed, the impression of the seal is read from right to left." — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)
- [second-hand] Parpola's summary of the external evidence: shorter inscriptions start at the right edge and leave blank space near the left edge; in longer inscriptions the leftmost signs are displaced to a second line for want of space. — [search summary pointing to Parpola 2010, "A Dravidian solution to the Indus script problem"](https://www.harappa.com/sites/default/files/pdf/Parpola-2010-Coimbatore.pdf) (page itself returned 403)
- [fetched] Ashraf & Sinha restate the evidence: direction is inferred because "signs get more crowded at the left end of some inscriptions or spill out of an otherwise linear arrangement". — [Ashraf & Sinha 2018, PLoS ONE 13(1): e0190735](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [second-hand] Direction is "left to right in seals and right to left in the impressions". — [search summary of Yadav & Vahia, "Indus Script: A Study of its Sign Design"](https://www.harappa.com/sites/default/files/pdf/Indus-sign-design.pdf) (PDF returned 403)

### Inferences
- The epigraphic arguments are of three independent kinds: (1) order of execution (stroke overlap, only possible on directly incised media such as pottery graffiti), (2) layout (cramping, spill-over, blank space on the left), (3) consistency of recurring sequences across objects with different layouts. Only (3) uses sign identity; (1) and (2) are independent of any transcription and are therefore the non-circular anchor for a statistical test.
- Lal's argument applies to direct-incised objects; for seals the argument is made on the impression. Any physical-position test must fix one view (impression view) and state it.

### Gaps
- Lal 1966 full reference (commonly given as *Antiquity* 40: 52-55, "The direction of writing in the Harappan script") was not confirmed from a primary page; number of sherds examined is unknown here.
- Parpola 1994 (*Deciphering the Indus Script*, direction section) and Possehl 1996 pp. 59-62 could not be opened; page numbers are from Wikipedia footnotes only.
- Earlier contributors (Marshall, Gadd, Hunter, Smith) and who first made the cramping argument: not found in an accessible source. One search summary credits the cramping observation to Mahadevan, which is likely a simplification; treat as unverified.

## What Mahadevan's concordance records about direction, and how the transcription normalises it

### Takeaway
Mahadevan 1977 (2,906 texts, 3,573 lines, 417 signs) records a direction code per line and prints every text in a single normalised reading order; published secondary figures put left-to-right at roughly 6-7% and the project's export gives 7.3% of directed lines. The export has no boustrophedon or top-to-bottom category.

### Cited Findings
- [fetched] The M77 corpus has "417 unique signs in 3573 lines of 2906 texts". — [Yadav et al. 2010, PLoS ONE 5(3): e9506](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [second-hand] Mahadevan's concordance: 2,906 inscribed objects from 26 sites (the search summary's wording; Yadav et al. call them texts); "6.6% examples are of left-to-right writing"; "16 texts must be read from top to bottom, and 10 texts are written in boustrophedon". The originating page was not identified (candidates returned by the search: Rao 2018 arXiv:1812.00049, Yadav & Vahia sign-design paper). — [search result set incl. Rao 2018](https://arxiv.org/abs/1812.00049)
- [second-hand] Mahadevan in interview: about 90 percent of the writing is right to left. — [Harappa.com, Mahadevan complete interview](https://www.harappa.com/content/iravatham-mahadevan-complete-interview) (403 on direct fetch)
- [second-hand] The concordance includes tables of sign distribution by position, site, object type, field symbol and direction of writing. — [search summary of Mahadevan 1977 listings](https://www.harappa.com/script/maha0.html)
- [fetched] Normalisation is standard practice: Ashraf & Sinha used a "database where the relatively few sequences which are believed to have been written from left to right have been reversed so as to be oriented in the same direction as the majority, following standard procedure used for constructing concordances for Indus Valley Civilization inscriptions". — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Print convention in Yadav et al.: "the texts depicted by pictures are to be read from right to left, whereas the texts represented by just strings of sign numbers are to be read from left to right." — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [local] The export has 3,916 rows (lines/placeholders) for 2,906 distinct `text_no`. `direction` values and row counts: right-to-left 2,974; unrecorded 343; left-to-right 235; single sign 190; no text 155; doubtful 12; symmetrical 7 (codes 1, 0, 2, 3, 9, 5, 4). — [local CSV](file:///Users/farhandhattiwala/AI_indus/-indus-epigraphy-ai-/data/raw/im77/im77_corpus_lines.csv)
- [local] All 343 "unrecorded" rows have `line == 9` and `n_signs == 0`: they are empty placeholders, not a direction class. 3,916 − 343 = 3,573 non-empty lines, which matches the published M77 line count exactly; they hold 14,153 sign occurrences (3.96 per line, 4.87 per text). — [local CSV](file:///Users/farhandhattiwala/AI_indus/-indus-epigraphy-ai-/data/raw/im77/im77_corpus_lines.csv)
- [local] The 155 "no text" rows (code 9) are not empty: 109 have 2 signs, 25 have 1, 21 have 3. The label is probably a mis-decoded code in the export; exclude from direction tests and investigate. — [local CSV](file:///Users/farhandhattiwala/AI_indus/-indus-epigraphy-ai-/data/raw/im77/im77_corpus_lines.csv)
- [local] LTR share: 235 / (2,974 + 235) = 7.3% of directed lines. At text level, 2,584 texts have at least one RTL line, 206 at least one LTR line, 75 have both; 206 / 2,715 = 7.6% of directed texts contain an LTR line. — [local CSV](file:///Users/farhandhattiwala/AI_indus/-indus-epigraphy-ai-/data/raw/im77/im77_corpus_lines.csv)
- [local] LTR lines are concentrated by medium and site (RTL / LTR lines): seal 1,749 / 69; sealing 601 / 61; miniature tablet 362 / 87; copper tablet 139 / 6; pottery graffito 74 / 10; ivory/bone rod 26 / 1; bronze implement 7 / 1; miscellaneous 16 / 0. By site: Mohenjodaro 1,533 / 48; Harappa 1,076 / 148; Lothal 169 / 15; Kalibangan 87 / 10; Chanhudaro 70 / 6; other 26 / 5; West Asian 13 / 3. — [local CSV](file:///Users/farhandhattiwala/AI_indus/-indus-epigraphy-ai-/data/raw/im77/im77_corpus_lines.csv)

### Inferences
- The export's 7.3% (lines) / 7.6% (texts) is close to but not the same as the second-hand "6.6%"; denominators probably differ (all lines vs directed lines). Report them separately.
- The 75 texts mixing RTL and LTR lines are the boustrophedon candidates in this export; this is larger than the second-hand "10 boustrophedon texts", so most are probably different sides or separate lines rather than true boustrophedon. Needs manual check.
- Harappa miniature tablets carry a disproportionate share of LTR lines (87 of 235), so "LTR" is confounded with object type and site.

### Gaps
- Mahadevan's own direction table (exact categories, counts and percentages, pp. 10-14 of the 1977 introduction) was not seen. His criteria for labelling a line LTR are not known here; this matters for circularity (see below).
- Whether direction for seals in M77 is coded in impression view was not confirmed from the primary source (it is the convention reported in secondary sources).

## Published statistical methods for inferring direction from sign sequences

### Takeaway
The one published direction statistic is Ashraf & Sinha's normalised difference of Gini index (or entropy) between the sign distributions at the two ends of sequences; its sign separates LTR from RTL scripts in 24 languages and is positive (RTL-like) for the Indus corpus. Yadav et al. 2010 give the underlying Indus fact: enders are far more concentrated than beginners (23 vs 82 signs for 80% of usage).

### Cited Findings
Ashraf & Sinha 2018 (Md. Izhar Ashraf and Sitabhra Sinha, IMSc Chennai; PLoS ONE 13(1): e0190735, 17 January 2018; arXiv:1509.09121):
- [fetched] Gini index (Eq 1), for a discrete distribution P(x) over N values indexed in non-decreasing order of probability: G = 2 Σ_{i=1..N} i·P(x_i) / (N Σ_{i=1..N} P(x_i)) − (N+1)/N. G = 0 when all values are equally probable; G = 1 when a single value always occurs. (The equation is an image on the page; this is the fetch tool's rendering, and it matches the standard form.) — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Entropy (Eq 2): S = −Σ_{i=1..N} P(x_i) log2 P(x_i), in bits. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Asymmetry measures: "ΔG = 2(GL − GR)/(GL + GR) where GL and GR are the Gini indices for the signs occurring in the left and right terminal positions, respectively"; ΔS = 2(SL − SR)/(SL + SR). — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Sign convention: "All languages that are conventionally read from left to right (or rendered in that format in the databases used here) show a negative value for ΔG, while those read right to left exhibit positive values." For entropy the signs flip: LTR gives positive ΔS, RTL gives ΔS < 0. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Reported values: English 1-grams ΔG = −0.50 (95% bootstrap CI [−0.48, −0.51]); English 2-grams ΔG = −0.25 (CI [−0.247, −0.254]); Basic English root words ΔG = −0.33 (CI [−0.27, −0.47]). Asymmetry weakens for higher-order n-grams. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Corpora: 24 languages (22 from nine families plus two isolates) and the Indus inscriptions; average corpus about ten thousand unique words; English from Mieliestronk (about 58,000 unique words) and Google Books (about 97,000). RTL scripts in the set include Arabic, Hebrew, Persian, Urdu. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Indus result: "the ΔG for sign usage distribution is positive, indicating that the choice of signs is less restricted in the right terminal position than the left. This would suggest ... that the IVC inscriptions are written from right-to-left, which corroborates the consensus view". No numeric Indus ΔG is given in the text; it is shown only in Fig 3. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Null model: surrogate data "generated by taking each sequence in turn that belongs to a database and doing a random permutation of the signs"; "an ensemble of 1000 randomized versions for each of the databases"; expected mean ΔG of surrogates is zero. No p-values or z-scores are reported; significance is non-overlap of the bootstrap interval with the surrogate interval. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Bootstrap: "we have created 10³ resampled datasets ... by random sampling with replacement, containing the same number of sequences as the original dataset"; interval by the "Bias-Corrected and accelerated (BCa) method". — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Data requirement: "the database needs to be larger than a minimal size (∼100 words for English) in order for the significance of the observed asymmetry to be established". For the Indus case: "Availability of a sufficiently large corpus would, however, be necessary for a reliable determination of the direction of writing in these inscriptions". — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Abstract claim: "the beginning of a word is less restrictive in sign usage than the end ... We use the existence of this asymmetry to infer the direction of writing in undeciphered inscriptions that agrees with the archaeological evidence." — [arXiv:1509.09121](https://arxiv.org/abs/1509.09121)

Yadav, Joglekar, Rao, Vahia, Adhikari, Mahadevan 2010 (PLoS ONE 5(3): e9506; arXiv:0901.3017):
- [fetched] EBUDS: 1,548 texts filtered from M77 by removing "all texts containing lost, damaged, illegible, or doubtfully read parts", texts spread across multiple lines, and duplicates ("only a single copy is retained"). — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [fetched] "82 text beginners account for about 80% of the text beginner usage, but only 23 text enders are needed to account for the same percentage of text ender usage." "text enders being more strictly defined than text beginners." — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [fetched] Most frequent beginners: signs 267, 391, 293. Most frequent enders: 342, 176, 211. Sign 342 (the "jar") "alone accounts for 10% of EBUDS". — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [fetched] "There is greater flexibility for signs preceding the text enders than the signs which tend to follow the text beginners." — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [fetched] "the unigram entropy of the EBUDS corpus is 6.68 and the bigram mutual information is 2.24"; "The perplexity of the bigram model is 26.69 which is significantly lower than that of unigram model which equals 68.82." These are forward-only; no backward model is reported. — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [second-hand] Abstract-level summary: the script has "well-defined signs which begin and end texts, directionality and strong correlations in sign order". Earlier paper: Yadav, Vahia, Mahadevan, Joglekar 2008, "A statistical approach for pattern search in Indus writing", *International Journal of Dravidian Linguistics*. — [arXiv:0901.3017](https://arxiv.org/abs/0901.3017)
- [fetched] Wikipedia notes the jar sign is the most frequent sign (Parpola's sign 311; that this is Mahadevan's 342 is the usual equivalence, not stated on the page and not checked here) and that "Just 67 signs account for 80 percent of usage". — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)

Other:
- [second-hand] Sinha et al., "Network analysis of a corpus of undeciphered Indus civilization inscriptions indicates syntactic organization" (*Computer Speech & Language*) and a 2023 paper "Markers and agencies of anisotropy in the Indus sign system" exist and bear on directional/positional structure; neither was readable (paywall/redirect). — [Springer](https://link.springer.com/article/10.1007/s43539-023-00102-3)

### Inferences
- A&S compute L and R on sequences *as displayed in the database*. Their Indus database had the LTR minority reversed, so their Indus result is a test on a normalised corpus: it shows the normalised corpus has ender-concentration on its left, not that each object was physically written RTL. It is exactly the circular design the project must avoid.
- A&S acknowledge Bryan K. Wells, which suggests (does not prove) their Indus data is a Wells/ICIT-type corpus rather than M77.
- The inference from sign of ΔG to direction rests on an empirical regularity (ends more restricted than beginnings) holding in 24 languages at word level; Indus "texts" are not known to be words, so the transfer is an analogy.
- Gini needs a stated support N. If GL and GR are computed over different supports (signs observed at each end), ΔG partly reflects support size; computing both over the union of signs seen at either end is a defensible choice and should be stated. ΔS does not have this problem.

### Gaps
- Numeric Indus ΔG/ΔS, the Indus corpus size and source in A&S: not in the main text; would need Fig 3 / S1 Table.
- Whether A&S use unique sequences only and exclude single-sign sequences is not stated in the fetched text.
- Rao et al. 2009 PNAS "A Markov model of the Indus script" (start/end tokens, positional statistics) could not be fetched (403); no numbers from it are recorded here.
- No published paper was found that runs a forward-vs-backward conditional-entropy or perplexity comparison on the Indus corpus as a direction test.

## Is forward-vs-backward bigram perplexity a valid direction test?

### Takeaway
No, not as an independent test. The joint distribution of adjacent pairs is the same read either way, so an exact bigram model has identical likelihood forward and backward; any gap comes from boundary handling, smoothing or held-out sparsity. What is direction-sensitive is the terminal distributions, which is what ΔG/ΔS measure directly.

### Cited Findings
- [fetched] The only perplexities published for the Indus bigram model are forward (unigram 68.82, bigram 26.69). — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [fetched] A&S define asymmetry purely from the two terminal-position distributions, not from transition probabilities. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)

### Inferences
- Chain rule: H(X,Y) = H(X) + H(Y|X) = H(Y) + H(X|Y). For a stationary sequence H(X) = H(Y), so H(next|current) = H(previous|current). Mutual information I(X;Y) is symmetric by definition.
- For an MLE bigram model with BOS and EOS tokens, scored on its own training data, forward and backward log-likelihoods are identical: both equal Σ c(a,b) log c(a,b) − Σ_s c(s) log c(s) − N log N (N = number of texts, sums over real signs s and all adjacent pairs including boundary pairs), because each sign's count as a left element equals its count as a right element and BOS, EOS both occur once per text.
- Without boundary tokens, H(Y|X) − H(X|Y) = H(Y) − H(X), where X ranges over non-final positions and Y over non-initial positions. Those two marginals differ only by the first and last signs of each text, so the gap is a terminal-distribution statistic again (ΔS in disguise), amplified for short texts (the local export averages 3.96 signs per line).
- With smoothing and held-out data a forward/backward gap can appear, but it reflects the estimator, vocabulary sparsity and where the rare signs sit, not an independent property. If the project reports it, present it as a secondary descriptive number alongside the identity above.
- What is legitimately asymmetric and reportable in a "directional matrix": per-sign positional profile (P(first), P(last), P(medial)); distinct-successor vs distinct-predecessor counts for each sign (Yadav et al.'s "flexibility" observation); number of signs covering 80% of each end.

### Gaps
- No literature source was found that states the forward/backward equivalence specifically for the Indus corpus; the argument above is standard information theory, not a cited Indus result.

## De-normalising to physical positions, null models, bootstrap and robustness checks

### Takeaway
Reverse the stored order for RTL-labelled lines and keep it for LTR-labelled lines to get physical left-to-right order, then compute ΔG/ΔS on physical left vs right ends. The non-circular result is the physical placement in the majority: in the 92% of lines labelled RTL the concentrated ender set sits on the physical left (sign 342 is the left terminal 802 times, the right terminal 0 times; exploratory ΔS = −0.395), which with Ashraf & Sinha's regularity points to right-to-left. The sign flip in the LTR-labelled minority (ΔS = +0.324) is only a consistency check, and is independent evidence only if Mahadevan assigned those labels from layout rather than from sign identity.

### Cited Findings
- [fetched] Published null: within-sequence random permutation, 1000 realisations; published CI: 10³ bootstrap resamples of sequences, BCa interval. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Published filtering precedent: drop damaged/doubtful texts and multi-line texts, keep one copy of duplicates (EBUDS). — [Yadav et al. 2010](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- [local] Exploratory run (lines with ≥2 signs, neither terminal sign `0`; physical order = stored order reversed for RTL lines, as stored for LTR lines; Gini over the union support; percentile bootstrap over lines, 1000 resamples; seed 0). — [local CSV](file:///Users/farhandhattiwala/AI_indus/-indus-epigraphy-ai-/data/raw/im77/im77_corpus_lines.csv)

  | Group | n lines | distinct signs at physical L / R | GL | GR | ΔG | SL (bits) | SR | ΔS | ΔS bootstrap 95% | mirror-half null 95% | within-line shuffle null 95% |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | RTL-labelled | 2,461 | 160 / 266 | 0.923 | 0.759 | +0.195 | 4.34 | 6.48 | −0.395 | [−0.423, −0.369] | [−0.026, 0.024] | [−0.017, 0.018] |
  | LTR-labelled | 210 | 72 / 41 | 0.652 | 0.866 | −0.283 | 5.40 | 3.89 | +0.324 | [0.247, 0.383] | [−0.071, 0.068] | [−0.067, 0.067] |
  | Pooled physical | 2,671 | 188 / 277 | 0.899 | 0.750 | +0.181 | 4.71 | 6.57 | −0.329 | not run | not run | not run |

- [local] Sign 342 (jar): in RTL-labelled lines it is the physical-left terminal 802 times and the physical-right terminal 0 times; in LTR-labelled lines physical-left 0, physical-right 40. Top physical-left signs in RTL lines: 342 (802), 176 (251), 328 (214); top physical-right: 267 (239), 391 (118), 87 (111). In LTR lines top physical-right: 328 (53), 342 (40), 176 (28); top physical-left: 89 (23), 87 (19), 267 (11). — [local CSV](file:///Users/farhandhattiwala/AI_indus/-indus-epigraphy-ai-/data/raw/im77/im77_corpus_lines.csv)

### Inferences
- Why the naive test is circular: in stored (reading) order every line has its ender last by construction of the transcription, so "enders are concentrated at the end" is true whatever the real direction was. A&S's Indus result has the same limitation.
- De-normalisation is an exact undoing of the transcriber's reversal, not an extra assumption, provided the `direction` field is the physical direction in impression view. After it, the data are "sign at physical position k on the object".
- What the physical test can and cannot show:
  - Pooled physical ΔG > 0 / ΔS < 0 says: across objects, the concentrated (ender-like) set sits on the physical left. Combined with the cross-linguistic regularity (ends more restricted than beginnings) this points to RTL for the majority. This step depends on A&S's analogy, not on Mahadevan's labels, but the pooled figure is dominated by the 92% RTL-labelled lines.
  - The sign flip in the LTR-labelled group shows the labels are internally consistent: the same signs (342, 176, 328) that sit on the physical left in most objects sit on the physical right in the minority. That is the "directional matrix" the project asks for: rows = direction label, columns = physical left/right, cells = terminal distributions and ΔG/ΔS.
  - Residual circularity that cannot be removed from this file: if Mahadevan labelled a line LTR *because* it looked like a known sequence reversed (e.g. jar at the right), then the flip in the LTR group is partly built in. Only lines whose direction was fixed by layout evidence (cramping, spill-over, stroke order) would be fully independent, and the export does not record the reason for each label.
- Recommended null models, each answering a different question:
  1. Within-text shuffle (A&S): tests whether any end asymmetry exists beyond sign frequencies. It does not test direction.
  2. Mirror-reversal of a random half of texts: gives the zero-direction baseline for the physical statistic; the observed ΔS should fall far outside it (it does in the exploratory run).
  3. Label permutation: shuffle the RTL/LTR labels across lines (stratified by object type and site) and recompute the between-group difference ΔS_RTL − ΔS_LTR. This tests whether the labels carry information about physical placement.
  4. Label-free check: without using the direction field at all, take each line in an arbitrary fixed physical order and find the orientation assignment that maximises ender concentration (two-cluster or EM on orientation); compare the recovered orientations with Mahadevan's labels. Agreement rate is a direct, non-circular statistic. This is a design suggestion, not a published method.
- Bootstrap: resample texts (`text_no`), not lines, so multi-line and multi-side objects stay together; 1000+ resamples; BCa as in A&S or percentile; report CI for each group and for the difference. The exploratory run above resampled lines, so its intervals are slightly too narrow.
- Filtering: drop single-sign lines, "symmetrical" and "doubtful" lines, the 343 empty placeholders and the mislabelled "no text" rows; drop lines whose terminal sign is `0` or flagged in `doubtful_positions`; run once with duplicates removed (identical sequences from moulded tablets and repeated sealings inflate counts; EBUDS precedent).
- Robustness splits: by object type (seal, sealing, miniature tablet, copper tablet, pottery graffito) and by site (Mohenjodaro, Harappa, Lothal, Kalibangan). The LTR group has only 210 usable lines, 87 of them Harappa miniature tablets, so per-cell LTR estimates will be wide; A&S's ~100-sequence minimum (for English) suggests only the seal, sealing and miniature-tablet LTR cells are near usable size.
- Seals vs sealings is a useful internal control: if direction is coded in impression view for both, seal and sealing groups should give the same sign of ΔS. If seals had been coded in carved view the sign would flip.
- Pottery graffiti (74 RTL / 10 LTR lines) are the class where Lal's stroke-order evidence applies; too few for a strong statistic but worth reporting separately.

### Gaps
- The exploratory numbers are from one unreviewed run on a third-party export; they need the text-level bootstrap, deduplication and the label-permutation null before being reported as results.
- No published study was found that runs the direction statistic on physical positions split by direction label; this design appears to be new for the project, so there is no literature value to compare against.

## Known caveats and criticisms of the direction statistics

### Takeaway
The statistic depends on corpus size, on the choice of Gini vs entropy, and on an untested analogy between words and Indus texts; and a 2026 preprint argues that directional asymmetry, like other Indus statistics, is not specific to language. None of this undermines the direction conclusion itself, which rests on physical evidence.

### Cited Findings
- [fetched] Hebrew shows "a low degree of asymmetry that is difficult to distinguish from effects due to stochastic fluctuations arising from sampling effects in a finite corpus". — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Hawaiian has a very low ΔG but "shows a significant asymmetry when information entropy is used", so the two measures can disagree. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Egyptian hieroglyphs (database oriented left to right): with all terminal signs included "the entropy difference ΔS is marginally negative, inconsistent with the hypothesis", attributed to many rare signs at the right terminal position. So logographic sign inventories with many rare signs can break the rule. — [Ashraf & Sinha 2018, S4 Fig](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] "The reason for the appearance of the directional asymmetry in sign usage distributions for linguistic sequences is yet to be definitively identified"; the cognitive-universal explanation is offered as a possibility. — [Ashraf & Sinha 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0190735)
- [fetched] Raghavendra (arXiv:2608.02999, submitted 4 August 2026), "On the Non-Specificity of Statistical Measures Used in Script Decipherment": a constructed non-phonological sign system (SIGIL, 3,000 texts) "receives the same category as the Indus corpus on every criterion scored" across 54 published methods, including directional-asymmetry measures. Abstract only was read; preprint, not peer reviewed. — [arXiv:2608.02999](https://arxiv.org/abs/2608.02999)
- [fetched] Left-to-right and boustrophedon exceptions exist, so "right to left" is a majority rule, not a law. — [Wikipedia, Indus script](https://en.wikipedia.org/wiki/Indus_script)

### Inferences
- The Indus sign list has many rare signs (417 signs, 67 covering 80% of usage), the same condition that produced the Egyptian anomaly. Report both ΔG and ΔS, and repeat with rare signs (frequency below a threshold) pooled or removed.
- Direction asymmetry shows that the system has a consistent start and end; it does not show that it encodes language. The project should word its claim as "sign placement is consistent with a right-to-left convention" and leave the linguistic question to the other research strand.
- The A&S regularity was established on words; a formulaic administrative text (e.g. content followed by a small closed class of terminal markers such as the jar sign) would give the same asymmetry. The inferred direction is the same either way, but the mechanism is not established.
- Normalised concordances propagate the transcriber's judgement: any LTR line misclassified as RTL (or the reverse) is silently reversed. The label-free orientation check above gives an estimate of how often that could have happened.

### Gaps
- No peer-reviewed critique aimed specifically at Ashraf & Sinha's direction result was found.
- Raghavendra 2026 full text was not read; which directional measures it tests and its numbers are unknown.
- Sproat's critiques of Indus statistics (e.g. *Language* 2014, "On statistical measures and ancient writing systems") surfaced in search but were not read for direction-specific content.
