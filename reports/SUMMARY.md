# Summary of results

Built by `reports/build_summary.py` from the results files; rerun it after any milestone script.

## Milestone 1: from a seal photograph to sign numbers

**Question.** Can a seal photograph be turned into an ordered list of Mahadevan sign numbers?

**Method.** A single-class detector (YOLOv8n or RT-DETR) boxes each sign and a frozen DINOv2 nearest-neighbour classifier names it. Trained on synthetic seals made from the IM-417 drawings; tested on held-out synthetic seals and, where labelled, on real photographs.

| Measure | Value |
|---|---|
| Synthetic test: detection recall | 99.9% |
| Synthetic test: sign named correctly | 77.2% |
| Synthetic test: whole sequence exact | 43.0% |
| Synthetic corpus-text seals: sign error rate, top-1 then re-ranked by the n-gram | 0.170 then 0.134 |
| Real test photos (12): detector recall | 58.8% |
| Real test photos: sign named correctly on true boxes | 24.1% |
| Real test photos: whole-pipeline sign error rate | 1.02 |
| Real test photos: best zero-shot segmenter recall (FastSAM) | 41.2% |

**What it shows.** The pipeline works on synthetic seals and the detector finds most signs on real photographs. **What it does not.** It does not yet name real signs reliably, so it cannot read a real inscription.

Reports: [synthetic](milestone_1/milestone_1_report.md), [real photographs](milestone_1/real/real_report.md), [re-ranking](milestone_1/rescoring.md), [zero-shot](milestone_1_zero_shot/zero_shot_report.md).

## Milestone 2: positional bias and Markov-chain modelling

**Question.** Is the order of Indus signs structured, and which way is it written?

**Method.** N-gram models, pair statistics and direction tests on the 1,548 EBUDS texts of Mahadevan's 1977 corpus, with shuffled corpora as nulls. Published figures from Yadav et al. 2010 are reproduced first.

| Measure | Value |
|---|---|
| Corpus reproduced (texts / tokens / signs) | 1,548 / 7,000 / 377 |
| Information shared by neighbouring signs, real vs shuffled (bits) | 3.12 vs 2.07, p = 0.001 |
| Best n-gram perplexity (Kneser-Ney trigram) | 26.73 |
| Hidden sign restored, first guess (Kneser-Ney trigram) | 42.2% |
| Direction statistic, delta S (negative = right to left) | -0.418 |

**What it shows.** Sign order is structured and non-random, and the layout is consistent with right-to-left reading. **What it does not.** None of it shows that the signs record a language.

Report: [milestone_2_report.md](milestone_2/milestone_2_report.md).

## Milestone 3: predicting hidden signs

**Question.** Can a model fill a missing sign from the rest of its inscription, with honest confidence?

**Method.** Small neural models (transformer or BiLSTM) against the best n-gram on identical held-out trials, 5-fold, with calibration and conformal candidate sets.

| Model | First guess right | Right in top 5 |
|---|---|---|
| Kneser-Ney trigram | 42.2% | 64.5% |
| Neural model | 32.2% | 53.0% |
| Mixture of the two | 42.4% | 64.8% |

**Reading.** Neural model is reliably worse than the Kneser-Ney trigram (-10.0 points, 95% interval -11.0 to -9.0); Mixture of neural model and Kneser-Ney trigram is statistically tied with the Kneser-Ney trigram (+0.2 points, 95% interval -0.1 to +0.4); Residual model: Kneser-Ney trigram + learned correction is statistically tied with the Kneser-Ney trigram (-0.1 points, 95% interval -0.3 to +0.2). The neural model cannot replace the n-gram on about 1,200 short training texts per fold, and what it adds on top is small: the neighbouring signs carry almost everything a model can use.

Report: [milestone_3_report.md](milestone_3/milestone_3_report.md).

## Milestone 4: testing claimed decipherments

**Question.** Does a published sign-to-sound table read the inscriptions better than chance?

**Method.** A pre-registered permutation test: the claimed table against 10,000 shuffles of its own values, scored by how Sanskrit-like the order of its output is, with positive and negative controls.

| Measure | Value |
|---|---|
| Known table detected (positive control) | 100.0% |
| Random tables flagged (negative control, 5% expected) | 2.5% |
| Yajnadevam key: percentile among frequency-matched shuffles | 94.6 |
| Yajnadevam key: Holm-corrected p | 0.65 |

**What it shows.** The one fully published key is not distinguishable from shuffled versions of itself. **What it does not.** It cannot show whether the script records a language.

Report: [milestone_4_report.md](milestone_4/milestone_4_report.md).

## Exploratory extensions

Not pre-registered. Each line is the result file's own reading.

- **Restoring damaged inscriptions** ([report](milestone_3/restoration_extensions_report.md)): two hidden signs both right 19.3% of the time with a joint search; a damaged stretch's length right 53.6%; a miscopied sign located 61.7% of the time (chance 22.1%).
- **Do the stroke signs behave like numerals?** ([report](milestone_2/numerals_report.md)): Counted things: stroke signs are followed by a narrower set of signs (entropy of the next sign, bits): does not hold (observed 3.792 against 2.798 for matched pseudo sets, q 1.000); a numeral-like result would be lower. Counted things: different stroke signs are followed by the same signs (mean Jensen-Shannon divergence): does not hold (observed 0.764 against 0.778 for matched pseudo sets, q 0.415); a numeral-like result would be lower. Interchangeability: text pairs differing only in which stroke sign stands in one position: holds (observed 116.000 against 34.792 for matched pseudo sets, q 0.005); a numeral-like result would be higher. Position: share of stroke-sign tokens at the end of a text: does not differ from matched pseudo sets (observed 0.058 against 0.192, q 0.250); this test has no single numeral-like direction. Runs: adjacent pairs of stroke signs (additive systems show them, positional ones do not): does not differ from matched pseudo sets (observed 159.000 against 95.867, q 0.132); this test has no single numeral-like direction. Frequency by value: rank correlation -0.60 (q 0.005); numerals would fall with value, so a negative correlation is the numeral-like result.
- **Does the text predict the object?** ([report](milestone_2/predict_object_report.md)): site: 31% balanced accuracy against 20% chance and 24% at most for permuted labels (p 0.032); text length alone gives 22%, and adding pairs to single signs changes it by +0.1 points. object type: 50% balanced accuracy against 20% chance and 26% at most for permuted labels (p 0.032); text length alone gives 42%, and adding pairs to single signs changes it by -2.1 points. field symbol: 46% balanced accuracy against 33% chance and 38% at most for permuted labels (p 0.032); text length alone gives 40%, and adding pairs to single signs changes it by +1.5 points.
- **Do signs that look alike behave alike?** ([report](milestone_2/visual_vs_context_report.md)): There is no reliable association between how alike two signs look (DINOv2) and how alike they are used (Spearman +0.053, Mantel p = 0.1811); raw pixels give +0.001 (p = 0.4970). The most similar-looking pairs are more alike in use than random pairs (p = 0.0001). The candidate pairs are where to look for sign-list variants; each needs checking by eye against carved examples.
- **What a blind search can reach** ([report](milestone_4/overfitting_report.md)): 
- **Dictionary match rate** ([report](milestone_4/lexicon_report.md)): yajnadevam-2024-11-30: beats fully shuffled tables on whole text match but not tables shuffled within frequency bins (percentile 62 on whole text match, percentile 92 on coverage), so the excess is explained by how often each sign occurs, not by which sound it carries. yajnadevam-2026-07-15: beats fully shuffled tables on whole text match but not tables shuffled within frequency bins (percentile 58 on whole text match, percentile 93 on coverage), so the excess is explained by how often each sign occurs, not by which sound it carries. Against the Old Tamil lexicon the frequency-binned test is significant for no key and score.
- **Linear B as Greek: a real decipherment as control** ([report](milestone_4/linear_b_report.md)): The true Linear B table scores at percentile 99.0 of frequency-matched shuffles (p 0.0101) and percentile 67.6 of fully shuffled tables (p 0.3244): the test detects it under binned 10 only. With 10% of values replaced the detection rate is 90% (binned null), and with 30% replaced 35%. Scored against the wrong language the table is still significant, which is an open problem for the method. Taken together this is at best a partial pass: the true table is detected only against frequency-matched shuffles, and a table that is right for Greek also scores significantly for Sanskrit, so a significant result from this test is not evidence that a table is right for the language it was scored against.
- **More controls on real scripts** ([report](milestone_4/real_controls_report.md)): Ugaritic as Hebrew (true): detected against frequency-matched shuffles (p 0.0001); at Indus size it is detected in 100% of subsamples. Linear A as Greek (wrong): significant, which is a false alarm and an open problem for the method (p 0.0189). With Linear B (detected), the method finds 2 of 2 true tables and raises 1 false alarm on 1 wrong table, against the frequency-matched null.
- **The same claim against four languages** ([report](milestone_4/languages_report.md)): Against frequency-matched shuffles the current table sits at percentile sanskrit 94, tamil 87, english 76, sumerian 22. None is significant after Holm correction. The search ceiling lies above the claimed table by sanskrit +0.58, tamil +2.44, english +0.88, sumerian +2.48 bits; the largest difference between languages is 1.90 bits against a typical spread over searches of 0.06, so the ceiling differs between languages by more than its spread.
- **How large would the corpus have to be?** ([report](milestone_4/corpus_size_report.md)): The true table does not beat every blind search in 80% of replicates at any size run, up to 20 times the Indus size (111,681 sign tokens). The permutation test against shuffled tables detects the true table at every size (power 100%, 100%, 100%, 100%, 100%, 100%), so what fails is the comparison with searched tables: a plain blind search finds tables that score above the true one on held-out texts at every size, and the tables it finds read only 3-5% of tokens the way the true table does. With this score, the true table is not what a search maximises, however large the corpus, so scoring below the search ceiling is not evidence against a table and scoring above shuffled tables is not evidence for one.

## What this project found

Indus sign order is structured: neighbouring signs share 3.12 bits of information against 2.07 when shuffled, and the layout is consistent with right-to-left reading.
A missing sign is guessed right 42.2% of the time by a trigram; neural models do not do better on a corpus this small.
The one fully published decipherment key scores at percentile 95 of shuffled versions of itself (Holm p 0.65): not distinguishable from chance.
A blind search passes the claimed key within about 50 swaps, so on this corpus a good-reading table is easy to find by search alone.
The test finds true decipherments of Linear B and Ugaritic but also flags Linear A read as Greek, so passing it is not evidence for a language.
On real seal photographs the detector finds 58.8% of signs, but naming them remains the bottleneck.
