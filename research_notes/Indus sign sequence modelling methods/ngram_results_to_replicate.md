# Published n-gram / Markov-chain analyses of the Indus script: numbers to replicate (state of literature, October 2026)

Source key (all opened and read in full unless marked otherwise):

- **Y10-PLoS** = Yadav, Joglekar, Rao, Vahia, Adhikari, Mahadevan 2010, PLoS ONE 5(3): e9506 — https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506 (full text XML read; Tables 2 and 3 read from the PLoS table images)
- **Y10-arXiv** = arXiv:0901.3017v1 (20 Jan 2009), the preprint of the same paper — https://arxiv.org/pdf/0901.3017 (table numbering differs from PLoS; some numbers differ, see below)
- **R09-PNAS** = Rao et al. 2009, PNAS 106(33):13685-13690, incl. SI — https://homes.cs.washington.edu/~rao/PNASIndus.pdf (author copy; pnas.org itself returned HTTP 403)
- **R09-Sci** = Rao et al. 2009, Science 324:1165, incl. Supporting Online Material — https://homes.cs.washington.edu/~rao/ScienceIndus.pdf
- **R10-IEEE** = Rao 2010, IEEE Computer 43(4):76-80 — https://homes.cs.washington.edu/~rao/ieeeIndus.pdf
- **R10-CL** = Rao et al. 2010, Computational Linguistics 36(4) reply to Sproat — https://homes.cs.washington.edu/~rao/IndusCompLing.pdf
- **S09** = Sinha, Pan, Yadav, Vahia, Mahadevan 2009, ACL-IJCNLP TextGraphs-4 workshop, pp. 5-13 — https://aclanthology.org/W09-3202.pdf
- **S10** = Sinha, Izhar, Pan, Wells 2010, arXiv:1005.4997 — https://arxiv.org/pdf/1005.4997
- **VF23** = Venkatesh & Farghaly 2023, Journal of Emerging Investigators 6 — https://emerginginvestigators.org/articles/22-256
- **N26** = Nair 2026, arXiv:2604.17828 — https://arxiv.org/pdf/2604.17828
- **T26** = Tiwari 2026, NLP4DH 2026 pp. 314-319 — https://aclanthology.org/2026.nlp4dh-1.28.pdf
- **Rg26** = Raghavendra 2026, arXiv:2608.02999 — https://arxiv.org/pdf/2608.02999

NOT opened (second-hand only): Yadav, Vahia, Mahadevan, Joglekar 2008a "A statistical approach for pattern search in Indus writing", IJDL 37:39-52, and 2008b "Segmentation of Indus texts", IJDL 37:53-72 (harappa.com PDF blocked by Cloudflare, HTTP 403); Venkatesh & Farghaly 2023 ICCCNT "Identifying anomalous Indus texts from West Asia…" (IEEE paywall / academia.edu 403).

Sign numbers below are Mahadevan 1977 (M77) sign numbers. In Yadav et al., number strings are written in reading order (left to right = first sign to last sign), i.e. the pair "267, 99" means 267 is read first.

## Yadav et al. 2010 (PLoS ONE): EBUDS rules, n-gram orders, smoothing, protocol, and every headline number

### Takeaway
Everything in the paper is computed on EBUDS (1,548 single-line, undamaged, de-duplicated texts; 7,000 tokens; 377 signs) with SRILM Witten-Bell-smoothed, Katz-backoff n-grams (n = 1..5) plus `<s>`/`</s>` tokens; the headline targets are perplexity 68.82 / 26.69 / 26.09 / 25.26 / 25.26, unigram entropy 6.68 bits, bigram mutual information 2.24 bits, 69 / 82 / 23 signs for 80% coverage of all / beginner / ender tokens, and 74% restoration "sensitivity". The 74% is NOT top-1 accuracy, the perplexity train/test split is not stated, and the entropy numbers differ between the arXiv and PLoS versions, so replication tolerances must be loose on those items.

### Cited Findings

Replication-target table (all EBUDS):

| Quantity | Published value | Where |
|---|---|---|
| M77 full concordance | 417 unique signs, 3573 lines, 2906 texts | Y10-PLoS, Results "Empirical Analysis" and Methods "Corpus" |
| EBUDS texts | 1548 | Y10-PLoS Methods "Corpus" |
| EBUDS sign occurrences | 7,000 ("1,548 lines of text, with 7,000 sign occurrences") | R09-PNAS Methods; R09-Sci SOM "Datasets"; S09 sec. 4.3 |
| EBUDS distinct signs | 377 (40 of 417 absent) | Y10-PLoS Methods; S09 sec. 3 |
| Max text length | 14 | Y10-PLoS "Probability of Sequences" |
| Modal text lengths | n = 3 and n = 5 | Y10-PLoS Fig. 15 caption |
| Most frequent signs | 342, then 99, 267, 59 (Fig. 2 labels five: 342, 99, 267, 59, 87) | Y10-PLoS "Empirical Analysis"; Y10-arXiv Fig. 3 |
| Share of sign 342 | "alone accounts for 10% of EBUDS"; 715 occurrences | Y10-PLoS Fig. 4 text; S09 sec. 4.3; R09-PNAS Fig. 2A |
| Signs covering 80% of all tokens | 69 | Y10-PLoS Fig. 4, Table 6 row 2 |
| Signs covering 80% of text-beginner tokens | 82 | same |
| Signs covering 80% of text-ender tokens | 23 | same |
| Zipf-Mandelbrot fit, log f_r = a - b log(r + c) | a = 15.39, b = 2.59, c = 44.47 ("95% Confidence interval") | Y10-PLoS Fig. 3 caption, Table 6 row 1 |
| Perplexity, n = 1..5 | 68.82, 26.69, 26.09, 25.26, 25.26 | Y10-PLoS Table 1 (= Y10-arXiv Table 3 = R09-Sci SOM Table S1) |
| Cross-entropy H_n(Q,P), n = 1..5 (bits) | 6.10, 4.74, 4.71, 4.66, 4.66 | Y10-PLoS Table 1 |
| Unigram entropy H (bits) | EBUDS 6.68; uniform-random baseline log2(377) = 8.56 | Y10-PLoS Table 4 text, Table 6 row 6 |
| Bigram mutual information I (bits) | EBUDS 2.24; random 0 | Y10-PLoS Table 4 text, Table 6 row 7 |
| Top text beginners | 267, 391, 293 (no counts published) | Y10-PLoS "Analysis of Bigrams", Fig. 6 |
| Top text enders | 342, 176, 211 (no counts published) | same |
| Signs most often following beginners | 99 and 123 | Y10-PLoS Fig. 8 text |
| Most frequent self-repeat pairs | (153,153) and (245,245) | Y10-PLoS "Analysis of Bigrams" |
| Most frequent pair | 267,99 with frequency 168, LLR 792.40 | Y10-PLoS Table 2 |
| Most frequent triplet | 336,89,211 with frequency 34 | Y10-PLoS Table 3 |
| Restoration sensitivity (bigram, 5-fold) | 74% mean, SD 2 | Y10-PLoS Table 5, Fig. 14 |

EBUDS filtering rules (verbatim):
- "an Extended Basic Unique Data Set (EBUDS) is created by removing from the concordance all texts containing lost, damaged, illegible, or doubtfully read parts [12]. Texts spread across multiple lines are also removed. For texts that occur more than once, only a single copy is retained." Site, stratigraphy and object type are "at present, not taken into account". — [Y10-PLoS, Materials and Methods "Corpus"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Reason for dropping multi-line texts: it is unclear whether each line is a separate text or a continuation, and the cross-line reading order "remains ambiguous". — [Y10-PLoS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- PNAS phrases the rule slightly differently: EBUDS "excludes all texts from Mahadevan's concordance containing ambiguous or missing signs and all texts having multiple lines on a single side of an object. In the case of duplicates of a text, only one copy is kept". — [R09-PNAS Methods](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- S09 phrasing: remove "all sequences that are incomplete, because of the presence of signs that are damaged, lost, illegible or not unambiguously identified"; "only sequences which are written on a single line are retained"; "the original number of 3573 sequences in M77 is reduced to 1548 sequences in EBUDS". — [S09 sec. 3](https://aclanthology.org/W09-3202.pdf)
- Order of operations shown in the text-length figure: raw M77 -> M77 without the "four instances of outliers, texts of length n = 2 and n = 3 which occur in unusually large numbers" (only single copies kept) -> "M77-unique" (unique texts only) -> remove "damaged, illegible and multi-line texts" -> EBUDS. — [Y10-PLoS Fig. 15 caption](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Sanity check for the filter: the 40 signs absent from EBUDS are "one sign (374) appears 9 times, one sign (237) appears 8 times, two signs (282, 390) appear 3 times, three signs (324, 376, 378) appear twice and thirty-three signs appear only once in M77". — [Y10-PLoS Methods "Corpus"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- "The relative frequencies have no significant change in the M77 and EBUDS corpora." — [Y10-PLoS "Empirical Analysis"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)

Model definition and smoothing:
- n-gram model = (n-1)th-order stationary Markov chain over signs; start token `#` (`<s>` in the arXiv version) and end token `$` (`</s>`); P(#) = 1 by convention, so beginner probability of sign a is P(a|#) and ender probability is P($|a) (written "P(a$)"). — [Y10-PLoS "Analysis of Bigrams"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506); [Y10-arXiv sec. 4](https://arxiv.org/pdf/0901.3017)
- "Here, we use the Witten-Bell algorithm to smooth our n-gram models." and "we used the Katz backoff algorithm to estimate bigram, trigram and quadrigram probabilities when appropriate." — [Y10-PLoS "Learning Markov Chains"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- "Markov chain learning was implemented using the SRILM toolkit [28] while the NSP [29] toolkit was used for the log-likelihood ratio test of association." — [Y10-PLoS Methods](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Worked example of smoothing (a direct unit test): before smoothing, P(b | a = 2) is non-zero only for b in {12, 14, 162, 176}; Witten-Bell "restores a finite, but small probability to the unseen sign pairs". — [Y10-PLoS Fig. 16 and text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Orders modelled: "successive orders of Markov chains starting with n = 1 to n = 5 for calculating perplexity"; detailed analysis is bigram, with "representative results for trigrams". — [Y10-PLoS "n-Gram Model for the Indus Script"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)

Information-theoretic definitions:
- H = -sum_a P(a) log2 P(a); I = sum_ab P(ab) log2[P(ab) / (P(a)P(b))] (bits). — [Y10-arXiv eqs. 9-10](https://arxiv.org/pdf/0901.3017)
- Cross-entropy H(Q,P) = -sum_a Q(a) log2 P(a), estimated as -(1/M) sum log2 P over the held-out corpus; perplexity = 2^H(Q,P). — [Y10-arXiv eqs. 11-13](https://arxiv.org/pdf/0901.3017)
- "We measure the cross-entropy against a held-out portion of the EBUDS corpus." — [Y10-PLoS "Significance of Correlations"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Version conflict on entropy: PLoS gives random 8.56 (log2 377), EBUDS H 6.68, I 2.24; the arXiv v1 Table 2 gives "Random 8.7039, EBUDS 6.6811" and "Mutual information 0 / 2.236", its body text says "the entropy of the EBUDS corpus is 6.3 and the mutual information is 2.3", and its Table 9 says "Random: 8.73; EBUDS: 6.68". — [Y10-PLoS Table 4/6](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506); contradicted in detail by [Y10-arXiv Table 2, sec. 4.3, Table 9](https://arxiv.org/pdf/0901.3017)

Table 2 (PLoS) / Table 1 (arXiv): 20 most frequent pairs (frequency in EBUDS) and 20 most significant pairs (log-likelihood ratio, Dunning 1993, null hypothesis: signs a and b independent) — [Y10-PLoS Table 2](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506); identical in [Y10-arXiv Table 1](https://arxiv.org/pdf/0901.3017):

| Rank | Frequent pair | Freq | Significant pair | LLR |
|---|---|---|---|---|
| 1 | 267,99 | 168 | 267,99 | 792.40 |
| 2 | 336,89 | 75 | 336,89 | 522.03 |
| 3 | 342,176 | 59 | 342,1 | 286.46 |
| 4 | 8,342 | 58 | 51,130 | 252.02 |
| 5 | 391,99 | 56 | 342,176 | 210.24 |
| 6 | 347,342 | 56 | 347,342 | 208.68 |
| 7 | 342,1 | 48 | 8,342 | 201.77 |
| 8 | 293,123 | 40 | 293,123 | 196.14 |
| 9 | 87,59 | 39 | 245,245 | 195.67 |
| 10 | 48,342 | 38 | 130,149 | 181.35 |
| 11 | 171,59 | 36 | 171,59 | 169.99 |
| 12 | 249,162 | 34 | 249,162 | 156.07 |
| 13 | 89,211 | 34 | 391,99 | 155.63 |
| 14 | 245,245 | 33 | 222,254 | 147.21 |
| 15 | 59,211 | 31 | 182,293 | 137.34 |
| 16 | 51,130 | 27 | 150,123 | 132.38 |
| 17 | 65,67 | 27 | 89,211 | 130.77 |
| 18 | 99,67 | 26 | 216,254 | 128.64 |
| 19 | 162,342 | 25 | 171,8 | 114.87 |
| 20 | 123,343 | 25 | 87,59 | 111.01 |

Table 3 (PLoS only): 20 most frequent triplets and 20 most significant triplets, LLR under the null P(abc) = P(ab)P(c) — [Y10-PLoS Table 3](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506):

| Rank | Frequent triplet | Freq | Significant triplet | LLR |
|---|---|---|---|---|
| 1 | 336,89,211 | 34 | 267,99,387 | 780.19 |
| 2 | 293,123,343 | 25 | 267,99,65 | 756.18 |
| 3 | 249,162,342 | 24 | 267,99,70 | 752.65 |
| 4 | 249,169,342 | 20 | 267,99,67 | 748.54 |
| 5 | 171,8,342 | 19 | 267,99,87 | 744.07 |
| 6 | 51,130,149 | 19 | 267,99,204 | 737.60 |
| 7 | 99,87,59 | 16 | 267,99,53 | 734.74 |
| 8 | 403,87,342 | 16 | 267,99,48 | 733.81 |
| 9 | 130,149,342 | 16 | 267,99,72 | 730.26 |
| 10 | 267,99,67 | 14 | 267,99,86 | 728.56 |
| 11 | 267,99,87 | 14 | 267,99,336 | 723.86 |
| 12 | 72,336,89 | 14 | 267,99,59 | 723.25 |
| 13 | 267,99,65 | 12 | 336,89,211 | 672.25 |
| 14 | 67,244,342 | 12 | 72,336,89 | 514.78 |
| 15 | 178,389,15 | 11 | 70,336,89 | 500.43 |
| 16 | 53,171,59 | 10 | 65,336,89 | 489.80 |
| 17 | 25,245,245 | 10 | 67,336,89 | 482.13 |
| 18 | 48,8,342 | 9 | 59,336,89 | 480.92 |
| 19 | 112,194,342 | 9 | 99,336,89 | 473.64 |
| 20 | 65,336,89 | 9 | 51,130,149 | 436.36 |

Restoration experiment and cross-validation:
- Restoration scores each candidate sign s_x by the full-string bigram probability, e.g. for a 3-sign text P(S) = P(</s>|s3) P(s3|sx) P(sx|s1) P(s1|<s>); maximise over the 417 signs by enumeration for one gap, and by a Viterbi algorithm "suitably modified to make it applicable to a Markov chain" for several gaps. — [Y10-arXiv eq. 14 / Y10-PLoS eq. 9](https://arxiv.org/pdf/0901.3017)
- Protocol: "The ratio of training to test set size used is 80:20. We divide EBUDS into 5 equal parts i.e. P1, P2, P3, P4 and P5"; each part in turn is the test set; "We drop out signs randomly from the test set and ask the model to fill in the dropped signs. We then count the number of true positives, that is, the number of times the predicted signs match with the signs under 90% area of the cumulative probability curve; otherwise, they are considered false negatives. This is repeated 100 times". Sensitivity = TP / (TP + FN). — [Y10-PLoS "Model Performance Evaluation by Cross-Validation"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Per-fold mean sensitivity (SD): P1 75 (2), P2 74 (2), P3 74 (2), P4 75 (2), P5 72 (2); average 74 (2). — [Y10-arXiv Table 7](https://arxiv.org/pdf/0901.3017) (PLoS Table 5 has the same average; its per-fold cells are images)
- Abstract wording: "The model can restore doubtfully read texts with an accuracy of about 75%"; Table 6 wording: "Sensitivity of the bigram model = 74%" / "Model can predict signs with 74% accuracy". — [Y10-PLoS abstract, Table 6](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Worked restoration examples (usable as spot checks): text 4312 = 342 48 53 70 with 48 deleted, model restores 48 (next candidates 347, 8); text 5237 = 342 135 67 99 267 with 267 deleted, restored 267 (then 391); text 4016 = 342 327 70 67 53 97 391 with 67 deleted, restored 67 (then 99, 65). — [Y10-arXiv Table 5](https://arxiv.org/pdf/0901.3017)
- Most probable texts under the bigram model: length 4 = 342 48 99 267 (occurs as M77 texts 1232 and 2580); length 5 = 342 8 171 99 267 (text 2476); length 6 = 342 8 171 53 99 267 (closest: text 1322). These strings are printed as on the object, i.e. the reading order is the reverse (267 99 ... 342). — [Y10-arXiv Table 8](https://arxiv.org/pdf/0901.3017)

Other findings:
- Frequent text beginners 267, 391, 293 are followed by few signs, whereas enders 342, 176, 211 are preceded by many different signs. — [Y10-PLoS Fig. 7 text](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Reading convention: "the texts depicted by pictures are to be read from right to left, whereas the texts represented by just strings of sign numbers are to be read from left to right"; "text beginners" = signs at the right extreme of pictured texts. — [Y10-PLoS Methods "Corpus"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)

### Inferences
- Mean EBUDS text length = 7000 / 1548 = 4.52 signs; texts equal lines in EBUDS because multi-line texts are removed. If the student's export reproduces 1,548 / 7,000 / 377 and the list of 40 absent signs (374, 237, 282, 390, 324, 376, 378 + 33 hapaxes), the filter is right.
- Internal check on Table 1: H_n = log2(perplexity) (2^6.10 = 68.6, 2^4.74 = 26.7, 2^4.66 = 25.3), so only one row needs to be matched.
- 168 / 211: sign 267 occurs 211 times (PNAS Fig. 2A, see next section) and the pair 267,99 occurs 168 times, so the MLE P(99 | 267) = 0.80, matching the "approximately 0.8" transition PNAS reports. This is a strong single-cell check on the transition matrix.
- The 74% is the rate at which the true sign falls inside the smallest candidate set holding 90% of the model's probability mass; it is not top-1 accuracy. Rg26 reports, for a synthetic corpus of similar size under the same criterion, 84.88% by the 90%-mass rule but only 24.84% top-1, which indicates how far apart the two metrics are. A student reporting top-1 should expect a number far below 74%.
- The perplexity table was produced by SRILM. To match it, the student should check in the SRILM documentation: whether Witten-Bell was used in backoff form (the paper says Katz backoff) rather than interpolated form; SRILM's default count cut-offs for orders 3 and above; and that SRILM perplexity counts the `</s>` prediction but not `<s>`. Library implementations such as NLTK's interpolated Witten-Bell are a different estimator and will give different n >= 3 values.
- The published unigram entropy 6.68 bits is larger than the held-out unigram cross-entropy 6.10 bits, which cannot happen if both are computed over the same event space; the likely explanation is that the perplexity includes the highly predictable `</s>` token (1,548 of 8,548 events) while H is over signs only. The student should compute both variants.
- Because the n = 4 and n = 5 values are identical (25.26) and texts average 4.5 signs, higher orders are effectively backing off; a Witten-Bell bigram matrix is the model actually used for all substantive results.

### Gaps
- The train/test split used for the perplexity table is not stated in either version (the 5-fold 80:20 protocol is described only for restoration). Whether Table 1 is one split or a fold average is unknown.
- No per-sign counts for text beginners or text enders are published anywhere I could open; only the top-3 identities and the 82 / 23 coverage figures.
- The number of signs dropped per test text in the restoration test, and whether "about 100" repeats re-draw one sign per text, are not stated.
- The Zipf-Mandelbrot log base and fitting procedure are not stated; the fit is quoted only with "95% Confidence interval".
- SRILM command-line options are not given.
- Only 7,000 is given as the token total; a full per-sign frequency table for EBUDS is not published (top-10 counts are in PNAS, next section).

## Yadav et al. 2008 (IJDL pattern-search and segmentation papers) and Rao et al. 2009 (Science, PNAS): methods, entropy definitions, numbers, what the Markov model was used for

### Takeaway
The 2008 papers defined EBUDS and did frequency-based pair/triplet/quadruplet and segmentation analysis (88% of texts of length >= 5 segmentable; reported here second-hand). The 2009 Rao papers used the same EBUDS but a different estimator (modified Kneser-Ney bigram, 417 signs + end-of-text token, 418 x 418 matrix): Science compared conditional entropy (in nats) against other systems, PNAS used the Markov chain for sampling, likelihood scoring of West Asian texts, and filling in damaged texts.

### Cited Findings

Yadav et al. 2008 (not opened; as reported by later papers):
- Bigram and trigram frequencies in EBUDS "differ significantly from frequencies expected from a Bernoulli scheme" (2008a). — [Y10-PLoS, "n-Gram Model for the Indus Script"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- "88% of the texts of length 5 and above can be segmented using frequent unigrams, bigrams, trigrams and quadrigrams and complete texts of length 2, 3 and 4. Moreover, frequent bigrams or texts of length 2 alone account for 52% of the segmented corpus." (2008b). — [Y10-PLoS, same section](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- EBUDS was first constructed in 2008a; the relative sign-frequency distribution in EBUDS was shown there to be comparable to M77. — [Y10-PLoS Methods "Corpus"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506); [S09 sec. 3](https://aclanthology.org/W09-3202.pdf)

Rao et al. 2009, PNAS "A Markov model of the Indus script":
- Data: EBUDS, "1,548 lines of text, with 7,000 sign occurrences. We used Mahadevan's list of 417 signs plus an additional EOT sign to denote the end of each text. Signs were fed to the model from right to left in each line of text, ending in the EOT sign." A "418 x 418 matrix was learned". — [R09-PNAS Methods and Fig. 2 caption](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Model: first-order Markov chain with prior (start) probabilities P(s_i) = fraction of texts beginning with s_i, and transition probabilities P(s_i | s_j); no separate start token. Smoothing: "we used modified Kneser-Ney smoothing (23) (based on ref. 24)" (Chen & Goodman; Kneser & Ney 1995). — [R09-PNAS Methods "Learning a Markov Model from Data"](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Ten most frequent signs with frequencies "in the dataset used to train the model": 715, 377, 211, 192, 166, 162, 147, 137, 134, 128 (signs shown as glyphs, ranked 1-10). — [R09-PNAS Fig. 2A](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Signs ranked 3 and 10 in that list "are much more likely to start a text than the others"; the most frequent sign is "highly unlikely to start a text" and highly likely to end one. — [R09-PNAS Results](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Highest transition value in the 10 x 10 block: approximately 0.8, the probability that the sign ranked 3 is followed by the sign ranked 2. — [R09-PNAS Results, Fig. 2C](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Text enders: among signs occurring at least 20 times, the ten most likely enders have end-of-text probabilities "in the range 0.87-0.39". — [R09-PNAS Results, Fig. 3C](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Self-repetition: one frequent sign "occurs as a pair in 33 of the 58 inscriptions in which it occurs"; another occurs in 8 inscriptions, 6 times as a pair. — [R09-PNAS Results, Fig. 3D](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Table 1 lists, for each of the 10 most frequent signs, the highest-probability predecessors and successors with transition values, restricted to signs occurring 20 or more times (e.g. most frequent sign: successor EOT 0.74; predecessors 0.92, 0.89, 0.87, 0.62, 0.55). Signs are glyphs, not numbers. — [R09-PNAS Table 1](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Uses of the model: (i) sampling new texts and comparing to the nearest corpus text by string edit distance; (ii) likelihood P(x|M) = P(x1) P(x2|x1) ... P(xL|xL-1); (iii) filling in missing signs by most probable explanation (Viterbi). — [R09-PNAS "Markov Models for Analyzing the Indus Script"](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- Restoration accuracy is not measured in PNAS itself; it cites the n-gram paper: "a first-order Markov model can correctly predicted deleted signs with ~74% accuracy (10)". — [R09-PNAS "Filling in Incomplete Indus Inscriptions"](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- West Asian texts: likelihoods of the 8 complete West Asian texts are 0; 3.11e-10; 7.09e-8; 6.34e-14; 0; 1.13e-11; 1.22e-12; 2.87e-17 (two zeros because of signs unseen in training); median 6.40e-13 versus median 1.12e-7 for "a randomly selected set of 100 texts originating from within the Indus valley, which were held out and not used for training" and length-matched. — [R09-PNAS Table 2 and text](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- SI example: an Indus text has likelihood 8.0e-6; the same text with the last sign moved to the front 5.2e-10; a West Asian seal 1.2e-12. SI Table S1 gives filled-in versions of damaged M77 texts 1001, 1002, 1017, 1020, 1034, 1059, 1111, 1144, 1165, 1176, 1240, 1265, 1289, 1343, 1408 (glyphs only). — [R09-PNAS SI Fig. S1, Table S1](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)

Rao et al. 2009, Science "Entropic evidence for linguistic structure in the Indus script":
- Indus data: the same EBUDS, "1548 lines of text, with 7000 sign occurrences in total"; tokens "417 signs identified by Mahadevan". — [R09-Sci SOM "Datasets"](https://homes.cs.washington.edu/~rao/ScienceIndus.pdf)
- Definitions (natural log, units nats): H = -sum_i P(i) log P(i) (eq. S1); conditional entropy C = -sum_i P(i) sum_j P(j|i) log P(j|i) (eq. S2). P(j|i) estimated with "modified Kneser-Ney smoothing"; P(i) from token frequency. — [R09-Sci SOM "Calculation of Conditional Entropy"](https://homes.cs.washington.edu/~rao/ScienceIndus.pdf)
- Fig. 1A plots conditional entropy against the number of tokens retained, ordered by frequency (x axis 0-400, y axis 0-6 nats); Fig. 1B plots relative conditional entropy (relative to a uniformly random sequence with the same number of tokens). No numeric values are printed in the text or SOM. — [R09-Sci Fig. 1](https://homes.cs.washington.edu/~rao/ScienceIndus.pdf)
- Comparison corpora: Brown corpus (20,000 sentences, about 1,026,600 words, 5,897,000 characters; tokens = 417 most frequent words, or 128 ASCII characters); Rig Veda book 1 hymns 1-100 (about 65,000 tokens, 388 symbols); Sangam Tamil Ettuthokai (about 876,000 tokens, 244 symbols); Sumerian ETCSL subset (about 10,300 signs, top 417); human chromosome 2 (first 1,000,000 nucleotides); E. coli proteins (374,986 amino acids); a Fortran program (55,625 tokens, top 417). — [R09-Sci SOM "Datasets"](https://homes.cs.washington.edu/~rao/ScienceIndus.pdf)
- Controls: Type 1 = 10,000 lines of 20 signs, each sign equally likely to follow any other; Type 2 = 10,000 lines of 20 signs where "each sign has a unique successor sign"; both with the same number of signs as the Indus corpus. — [R09-Sci SOM "Datasets"](https://homes.cs.washington.edu/~rao/ScienceIndus.pdf)
- The SOM repeats the Zipf-Mandelbrot fit (a = 15.39, b = 2.59, c = 44.47) and the perplexity table (68.82, 26.69, 26.09, 25.26, 25.26), with perplexity defined as 2^H_n, H_n = -(1/M) sum log2 P(x_i^n). — [R09-Sci SOM, Table S1](https://homes.cs.washington.edu/~rao/ScienceIndus.pdf)

Block entropy (Rao 2010):
- Block entropy for block size N is the Shannon entropy of the distribution over N-sign blocks; computed for N = 1..6 with the NSB estimator (Nemenman, Shafee, Bialek 2002); normalised by taking the logarithm to base L, the alphabet size ("417 for Indus, 4 for DNA"); error bars are 1 SD and "negligibly small except for block size 6". Results shown only as a plot (Fig. 3; y axis 0-6). — [R10-IEEE Fig. 3 and text](https://homes.cs.washington.edu/~rao/ieeeIndus.pdf)
- The corpus for the block-entropy result is again "about 1,550 lines of text and 7,000 sign occurrences". — [R10-CL](https://homes.cs.washington.edu/~rao/IndusCompLing.pdf)

### Inferences
- Mapping the PNAS top-10 glyph list to M77 numbers (my inference, not stated in PNAS): rank 1 = 342 (715), rank 2 = 99 (377), rank 3 = 267 (211), rank 4 = 59 (192), rank 5 = 87 (166); ranks 6-10 (162, 147, 137, 134, 128 occurrences) are, as a set, {176, 67, 211, 162, 391}, because S09 Fig. 3 lists the ten highest-frequency EBUDS signs as 267, 99, 342, 391, 162, 211, 67, 176, 87, 59. Rank 10 = 391 is consistent with PNAS saying signs ranked 3 and 10 are the likely text starters (267 and 391 are Yadav's top two beginners). The order within ranks 6-9 should be confirmed from the student's own counts.
- The two papers differ in three replication-relevant ways although they share a corpus: smoothing (Witten-Bell + Katz backoff vs modified Kneser-Ney), boundary handling (`<s>` and `</s>` vs start-prior + EOT), and log base (bits vs nats). The milestone brief asks for Witten-Bell, so Yadav 2010 is the replication target and Rao 2009 is the cross-check.
- Rao's conditional entropy has no published numeric value for EBUDS; the only way to compare is visually against Fig. 1A, or against N26's independent 3.232 bits on a different corpus (next section).
- The West Asia likelihood test is a held-out-likelihood comparison with length matching; the two zero likelihoods show the PNAS model gave zero probability to signs unseen in training, i.e. the vocabulary was not smoothed at the unigram level.

### Gaps
- Yadav et al. 2008a/b were not opened (HTTP 403). Their frequency thresholds for "frequent" signs/pairs/triplets/quadruplets, their own pair/triplet tables, and any positional (beginner/ender) tables are unreported here; Rg26 notes that the segmentation paper "does not reproducibly specify its singleton-frequency cutoff".
- No numeric conditional-entropy or block-entropy values for the Indus corpus are printed in Science, its SOM, IEEE Computer or the Computational Linguistics reply; they exist only as plotted curves.
- NSB parameter settings are deferred to "(Rao 2010a)" and were not found in the IEEE text.
- UW technical report UW-CSE-08-11-02 (the PNAS precursor) was not opened; it may contain extra numbers.

## Later replications and extensions with comparable metrics (Sinha et al. network analysis, 2020s papers)

### Takeaway
Sinha et al. 2009 is the only later study on EBUDS itself and gives directly checkable transition-graph numbers (2,178 distinct observed pairs, 284 pairs with z > 10). Every 2020s study with numbers uses a different corpus or sign list (ICIT/Wells, 584-593 signs), so their values (conditional entropy 3.23 bits, Hit@10 63%) are context, not replication targets; no post-2010 paper was found that re-derives the Yadav 2010 perplexities from the M77 data.

### Cited Findings

Sinha et al. 2009 (EBUDS):
- Directed sign network: 377 nodes; edge i -> j if j immediately follows i; "only 1.5% (=2178) of the 377x377 possible directed pairs are seen to occur"; "connectance ... only around 0.01". — [S09 sec. 4.1](https://aclanthology.org/W09-3202.pdf)
- "the most frequent sign appearing as many as 715 times, which is 10% of the total of 7000 occurrences of the 377 signs in EBUDS". — [S09 sec. 4.3](https://aclanthology.org/W09-3202.pdf)
- Ten highest-frequency signs: 267, 99, 342, 391, 162, 211, 67, 176, 87, 59 (Fig. 3; both 342 -> 162 and 162 -> 342 occur). — [S09 Fig. 3](https://aclanthology.org/W09-3202.pdf)
- Innermost k-core has order 8 (in-degree, out-degree and undirected); the in-degree core has 26 signs; "391-99 and 336-89 are the commonest pairs" within it. — [S09 Figs. 4-5](https://aclanthology.org/W09-3202.pdf)
- Modularity optimisation gives 8 communities of 6 to 87 signs; no significant correlation with artifact type, field symbol or site. — [S09 sec. 4.2](https://aclanthology.org/W09-3202.pdf)
- Pair significance by z-score against shuffled corpora (one million realisations; shuffling over the whole corpus or within each seal): "284 sign pairs with z-score larger than 10, while 46 pairs have z-score more than 20"; pair 267-99 has empirical probability about 0.011 versus a null centred near 0.001 (Fig. 7). — [S09 sec. 4.3, Figs. 7-8](https://aclanthology.org/W09-3202.pdf)
- Frequency ranks: sign 51 is the 35th most common, 149 the 50th, 130 the 59th; 13 of the 16 most frequent signs appear in the significant-pair network of 22 signs. — [S09 sec. 4.3](https://aclanthology.org/W09-3202.pdf)
- Rank-frequency: two power-law regimes with MLE exponents of about 0.52 and 2.38 for the cumulative distribution (Fig. 6), "Neither equal 1". — [S09 Fig. 6](https://aclanthology.org/W09-3202.pdf)
- Segmentation trees built by greedily merging the highest-z pair; for EBUDS texts of length >= 10 the average tree height is "around 5". — [S09 sec. 4.4](https://aclanthology.org/W09-3202.pdf)

Sinha et al. 2010 (NOT EBUDS; Wells sign list "W09IMSc", "WUCS" data set):
- 593 signs, 8,095 sign occurrences; most frequent sign "740" occurs 831 times; 21 signs occur only as solo inscriptions; 369 distinct signs can start a sequence of length >= 2, of which 128 occur only initially; 196 distinct signs terminate a sequence, of which 43 occur only terminally; 120 signs both begin and end sequences; connectivity 0.0077; cumulative frequency exponent 0.66 (Zipf exponent about 1.5); 377 pairs with z > 3 and 31 with z > 8 (10^4 randomisations). — [S10](https://arxiv.org/pdf/1005.4997)

Venkatesh & Farghaly 2023 (NOT EBUDS; ICIT corpus, Wells/Fuls sign codes):
- ICIT "about 4500+ texts", cleaned to "2223 unique texts"; 80% train / 20% test; n-gram orders 2 to 7; models compared: MLE, Kneser-Ney interpolated, Lidstone, Stupid Backoff, Witten-Bell interpolated; one sign removed per test text, 100 random repeats. — [VF23](https://emerginginvestigators.org/articles/22-256)
- Best configuration (Lidstone with their "Sign Fill-in Full" algorithm): exact match (Hit@1) about 35%, Hit@5 about 57%, Hit@10 about 63%, sign-category match about 49%; random guess gives about 1.4% at Hit@10 and positional probabilities alone about 7%. — [VF23](https://emerginginvestigators.org/articles/22-256)
- Position: ICIT signs 820, 861, 817, 920 dominate text-initial position; sign 740 (the most frequent) occurs mostly text-finally. — [VF23](https://emerginginvestigators.org/articles/22-256)

Nair 2026 (NOT EBUDS; ICIT via the Yajnadevam digitisation):
- 2,511 inscriptions, 595 (24%) exact duplicates, leaving 1,916; 11,110 tokens; 584 sign types; mean length 4.42, median 4.0. — [N26](https://arxiv.org/pdf/2604.17828)
- Zipf slope -1.492 (OLS, R^2 = 0.956); conditional entropy 3.232 bits versus within-inscription shuffle null 4.613 +/- 0.015 bits (1,000 permutations; observed below all); hapax fraction 33.2%; positional Cramer's V 0.149; 7 terminal-class and 3 initial-class signs after Bonferroni correction. — [N26 Table 3, sec. 5.2, 5.6](https://arxiv.org/pdf/2604.17828)

Raghavendra 2026 (re-uses published EBUDS values; builds a synthetic non-language "Sigil"):
- Uses EBUDS anchors 1,548 texts, about 7,000 tokens, 377/417 types, mean length about 4.5. — [Rg26 Table 3](https://arxiv.org/pdf/2608.02999)
- A re-implementation of the Yadav protocol on the synthetic corpus (3,000 texts, 13,717 tokens, 419 types): five-fold Witten-Bell perplexity on distinct strings with fixed vocabulary and recursive backoff = 64.223, 25.797, 28.592, 30.177, 30.387 for orders 1-5 (fold SD 0.86-1.77); restoration with one position removed from each validation text 100 times (275,300 trials) = 84.88% by the 90%-mass criterion, 24.84% top-1; Zipf-Mandelbrot b = 2.574, c = 28.46; 39 vs 10 signs cover 80% of initial vs final tokens. — [Rg26 sec. 6.2, App. B, Figs. 8-9](https://arxiv.org/pdf/2608.02999)
- Reported "reproduced Indus outcome" values: adjacent/total repetition 23/40 = 0.575; terminal Gini asymmetry dG = 0.1788 (p = 0.001); terminal entropy asymmetry dS = -0.4144 (p = 0.001); transition-graph reciprocity 0.148 observed vs 0.338 shuffle mean; connectance 0.77% observed vs 1.1% null. — [Rg26 Table 2 and sec. 6](https://arxiv.org/pdf/2608.02999)

Tiwari 2026 (corpus unspecified; low reliability):
- Claims "a corpus of 6,579 inscriptions", about 400 signs, 80/20 split; conditional entropies (bits) H1 = 6.58, H2 = 3.60, H3 = 1.42 in the original direction and 6.58, 4.11, 1.61 reversed; positional rigidity R_i = 1 - H_i/H_1 of 0.082, 0.110, 0.050, 0.140, 0.220, 0.740 for positions 0-5; forward/backward KL divergence 16.45 bits (bigram), 21.43 bits (trigram); BiLSTM (embedding 128, hidden 256, 10 epochs) test perplexity 2.60; 4-state HMM selected by BIC. — [T26 Tables 1-4](https://aclanthology.org/2026.nlp4dh-1.28.pdf)

A public toolkit, joyboseroy/indus_decipher, ships an "m77_indusscript_real_corpus.csv" (3,573 lines, 418 signs) and reports top-90%-mass restoration of "90 to 98%" with candidate sets of "60 to 85% of vocabulary"; no paper is linked. — [GitHub README](https://github.com/joyboseroy/indus_decipher)

Pointer only (out of scope here): the linguistic vs non-linguistic argument over these statistics runs through Sproat 2010/2014, Rao et al. 2010 (R10-CL), N26 and Rg26; writing-direction methodology is covered in the Y10-arXiv sec. 2.2 remark on directionality and in T26/Rg26 asymmetry tests.

### Inferences
- The S09 numbers are the best extra unit tests for the student's transition-count matrix on EBUDS: exactly 2,178 non-zero sign-to-sign cells (2,178 / 377^2 = 1.53%), sign 342 total 715.
- Connectance conflict: S09 gives 1.5% for EBUDS, Rg26 quotes 0.77% as the Indus value; the Rg26 figure matches the S10 Wells-list network (0.0077), so it is a different corpus and sign list, not a contradiction about EBUDS.
- T26's numbers are not usable as targets: 6,579 inscriptions exceeds any published corpus derived from the cited sources, a trigram conditional entropy of 1.42 bits on about 400 signs indicates unsmoothed plug-in estimates on sparse data, and a neural test perplexity of 2.60 is far below the 25-27 of smoothed n-grams. Its unigram entropy 6.58 bits is nonetheless close to Yadav's 6.68.
- The toolkit's 90-98% with candidate sets covering 60-85% of the vocabulary, and Rg26's 85% vs 25%, both show the 90%-mass criterion is lenient when the predictive distribution is flat; the student should report the mean candidate-set size alongside any replication of the 74%.
- N26's conditional entropy (3.23 bits = 2.24 nats) and T26's H2 (3.60 bits) are plug-in or differently smoothed estimates on different sign lists; they bracket where a Witten-Bell estimate on EBUDS might fall but do not validate it.

### Gaps
- Venkatesh & Farghaly 2023 ICCCNT (West Asian texts, per-text perplexity under a best-fit n-gram model) was not opened; no numbers are reported here.
- "Kumar et al. 2021, Neural sequence models for the Indus script (arXiv preprint)" is cited only in N26's reference list; I could not locate it and it should not be treated as a verified source.
- Other works named in Rg26's census (Ramani & Kishor 2024 dependency clustering; Khanna & Merriam 2025 sign-function analyses; Recchia & Louwerse 2016) were not opened.
- No published independent re-computation of the Y10 perplexity table, entropy, or 82/23 asymmetry from the M77 data was found as of October 2026; Rg26 and N26 re-use or approximate the published values.

## Which numbers are EBUDS versus full concordance, and published sign-frequency (Zipf-Mandelbrot) fit parameters

### Takeaway
Essentially every published statistic (frequencies, pair/triplet counts, beginner/ender coverage, entropy, perplexity, restoration, network measures, Zipf-Mandelbrot fit) is computed on EBUDS; the full M77 concordance appears only as the corpus description, the source of damaged texts to restore, the West Asian test texts, and the counts of the 40 dropped signs. The only Zipf-Mandelbrot parameters published for M77-coded data are a = 15.39, b = 2.59, c = 44.47 (EBUDS).

### Cited Findings
- Full-M77 numbers: 417 signs, 3573 lines, 2906 texts; M77 frequencies of the 40 signs missing from EBUDS (374 x9, 237 x8, 282 and 390 x3, 324, 376, 378 x2, 33 signs x1); text-length distribution curves for M77, M77 without the 4 most frequent texts, M77-unique and EBUDS (Fig. 15, plotted only). — [Y10-PLoS Methods "Corpus", Fig. 15](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- M77 texts used as test material, scored by EBUDS-trained models: doubtfully-read-sign restorations for M77 texts 8302, 5317, 1193, 1407, 2179, 3396, 8101, 2802 (Y10) and damaged texts 1001-1408 (PNAS SI Table S1); the 8 West Asian texts in PNAS Table 2 are "complete and unambiguous West Asian texts from ref. 13". — [Y10-arXiv Table 6](https://arxiv.org/pdf/0901.3017); [R09-PNAS Table 2, SI](https://homes.cs.washington.edu/~rao/PNASIndus.pdf)
- EBUDS numbers: Figs. 2-8 and Tables 1-5 of Y10-PLoS are all captioned "EBUDS"; PNAS and Science Methods state the model/entropy data set is EBUDS; S09 "report[s] the results of applying network analysis techniques to the sign sequences in EBUDS". — [Y10-PLoS](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506); [R09-PNAS](https://homes.cs.washington.edu/~rao/PNASIndus.pdf); [R09-Sci SOM](https://homes.cs.washington.edu/~rao/ScienceIndus.pdf); [S09 sec. 4](https://aclanthology.org/W09-3202.pdf)
- Zipf-Mandelbrot: "log f_r = a - b log(r + c)"; "Our fitted values are a = 15.39, b = 2.59 and c = 44.47. For English (the Brown Corpus), a = 12.43, b = 1.15 and c = 100"; arXiv Table 9 rounds to a = 15.4, b = 2.6, c = 44.5. — [Y10-arXiv Fig. 4, Table 9](https://arxiv.org/pdf/0901.3017); [Y10-PLoS Fig. 3](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- The 69-sign / 80% observation "is consistent with previous analysis by Mahadevan for M77 corpus". — [Y10-PLoS "Empirical Analysis"](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0009506)
- Other published frequency fits (different corpora and functional forms): S09 EBUDS two-regime power law, exponents about 0.52 and 2.38; S10 Wells-list cumulative exponent 0.66 (Zipf exponent about 1.5); N26 ICIT OLS Zipf slope -1.492 (R^2 = 0.956). — [S09 Fig. 6](https://aclanthology.org/W09-3202.pdf); [S10 Fig. 10](https://arxiv.org/pdf/1005.4997); [N26 Table 3](https://arxiv.org/pdf/2604.17828)
- Lexical-statistics goodness-of-fit tests reported for the Indus frequency spectrum reject all three LNRE models: Zipf-Mandelbrot p = 1.23e-23, finite Zipf-Mandelbrot p = 1.28e-9, GIGP p = 6.51e-4. — [Rg26 Table 2](https://arxiv.org/pdf/2608.02999)

### Inferences
- With c = 44.47 and b = 2.59 the published fit is far from a classical Zipf law (b = 1, c = 0); the curve is nearly flat over the first few dozen ranks and steep in the tail. Fig. 4 of the arXiv version shows the two most frequent signs (342, 99) lying visibly above the fitted curve, so the student should expect a poor fit at ranks 1-2 and should fit in log-log space over all 377 ranks. The value a = 15.39 with a log-frequency axis running 0-6.5 implies natural logarithms (ln 715 = 6.57), which is my reading of the plot rather than a stated fact.
- For the milestone, statistics computed on the full 2,906-text export (13,372 tokens, 417 signs) have no published counterpart to match against beyond Mahadevan's own 1977 tables; they should be presented as the student's extension, with EBUDS as the like-for-like replication.

### Gaps
- No Zipf-Mandelbrot parameters are published for the full M77 concordance, and no confidence intervals are printed for the EBUDS fit despite the "95% Confidence interval" label.
- Mahadevan's 1977 frequency and positional tables (the source for full-concordance beginner/ender counts) were not consulted; they are a printed book and were outside the sources opened.
- The four "outlier" texts of length 2-3 that are heavily duplicated in M77 are not identified by text number in any opened source.
