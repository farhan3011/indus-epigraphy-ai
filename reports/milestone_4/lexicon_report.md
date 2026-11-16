# Dictionary match rate (exploratory, not pre-registered)

A claimed table is applied to the Wells/ICIT inscriptions and its output is reduced to the consonant skeleton used in Milestone 4. Two scores are computed: the share of outputs (length 3 or more) that are exactly a word of the lexicon, and the share of output letters covered by a greedy longest-match segmentation into lexicon words. Each is compared with the same score for shuffled versions of the table, either shuffled over all signs or only within 10 frequency bins.

Lexicon: skeletons of words of 3 to 12 letters seen at least 2 times in the DCS training split: 8,264 entries (Old Tamil control lexicon: 7,030). Short skeletons match by chance, which is why only the comparison with shuffled tables means anything. Vowels are ignored, so one skeleton stands for many words.

## Controls

Positive control: Sanskrit enciphered with the claimed key, 20 replicates per row, 500 draws each. Share of replicates with p < 0.05 when the key is correct and when a share of its values is replaced.

| Values replaced | Score | Value-shuffle null | Binned null |
|---|---|---|---|
| 0.0% | whole text match | 95.0% | 100.0% |
| 0.0% | coverage | 100.0% | 100.0% |
| 10.0% | whole text match | 80.0% | 95.0% |
| 10.0% | coverage | 70.0% | 100.0% |
| 20.0% | whole text match | 70.0% | 70.0% |
| 20.0% | coverage | 30.0% | 100.0% |
| 30.0% | whole text match | 45.0% | 55.0% |
| 30.0% | coverage | 0.0% | 85.0% |

Negative control: 100 random shuffles of the key, each tested against its own shuffles (199 draws). Share with p < 0.05 (about 5% expected): whole text match 9.0%, coverage 4.0%.

## Claimed keys

| Key | Lexicon | Score | Observed | Value-shuffle: mean (sd), percentile, p, Holm p | Binned: mean (sd), percentile, p, Holm p |
|---|---|---|---|---|---|
| yajnadevam-2024-11-30 (1,334 texts) | sanskrit | whole text match | 0.2563 | 0.1130 (0.0307), 100.0, 0.0005, 0.004 | 0.2454 (0.0403), 61.7, 0.3838, 1.000 |
| yajnadevam-2024-11-30 (1,334 texts) | sanskrit | coverage | 0.6133 | 0.6521 (0.0555), 23.2, 0.7681, 1.000 | 0.5633 (0.0390), 92.0, 0.0800, 0.429 |
| yajnadevam-2024-11-30 (1,334 texts) | tamil | whole text match | 0.1403 | 0.0722 (0.0315), 97.3, 0.0275, - | 0.1352 (0.0425), 50.1, 0.4988, - |
| yajnadevam-2024-11-30 (1,334 texts) | tamil | coverage | 0.3419 | 0.3538 (0.0967), 46.1, 0.5397, - | 0.3084 (0.0614), 65.6, 0.3443, - |
| yajnadevam-2026-07-15 (1,318 texts) | sanskrit | whole text match | 0.2578 | 0.1127 (0.0320), 100.0, 0.0005, 0.004 | 0.2514 (0.0394), 57.7, 0.4233, 1.000 |
| yajnadevam-2026-07-15 (1,318 texts) | sanskrit | coverage | 0.6138 | 0.6500 (0.0565), 24.6, 0.7541, 1.000 | 0.5649 (0.0375), 92.9, 0.0715, 0.429 |
| yajnadevam-2026-07-15 (1,318 texts) | tamil | whole text match | 0.1401 | 0.0673 (0.0298), 98.3, 0.0175, - | 0.1312 (0.0421), 53.7, 0.4633, - |
| yajnadevam-2026-07-15 (1,318 texts) | tamil | coverage | 0.3357 | 0.3472 (0.0979), 47.1, 0.5292, - | 0.2995 (0.0599), 67.8, 0.3223, - |

Holm correction over 8 tests (2 keys, 2 scores, 2 nulls, Sanskrit lexicon). The Old Tamil rows are a control for what the score measures: a table that matches the Tamil lexicon as well as the Sanskrit one is not showing anything Sanskrit.

## Reading

yajnadevam-2024-11-30: beats fully shuffled tables on whole text match but not tables shuffled within frequency bins (percentile 62 on whole text match, percentile 92 on coverage), so the excess is explained by how often each sign occurs, not by which sound it carries. yajnadevam-2026-07-15: beats fully shuffled tables on whole text match but not tables shuffled within frequency bins (percentile 58 on whole text match, percentile 93 on coverage), so the excess is explained by how often each sign occurs, not by which sound it carries. Against the Old Tamil lexicon the frequency-binned test is significant for no key and score.

## Limits

Skeletons ignore vowels, so many unrelated words share one skeleton. The lexicon comes from five post-Vedic texts only. Greedy segmentation can miss better covers. The test can reject a specific table under this score; it cannot show whether the script records a language. Not pre-registered and outside the corrected family of Milestone 4 tests.

Settings: 2000 draws per null, seed 20261116, runtime 4.0 min, quick run: False.
