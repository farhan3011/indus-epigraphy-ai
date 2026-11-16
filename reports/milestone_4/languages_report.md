# The same claim against four languages (exploratory, not pre-registered)

Is Sanskrit special for the claimed table? Four languages are reduced to the same 16-letter consonant skeleton and cut to the same length (336,466 letters each, so each trigram is equally well estimated): Sanskrit (DCS), Old Tamil (Sangam verse), English (Pride and Prejudice) and Sumerian (CDLI sign readings of Sumerian texts). For each, the claimed table is compared with shuffled versions of itself, and a blind search is run on half the texts and scored on the other half. Only the primary score is used: log-probability under the language's character trigram minus under its letter frequencies, in bits per phoneme.

Scores in bits are not comparable across languages in absolute terms; what compares is the percentile against each language's own shuffles and the gap to each language's own search ceiling.

## Permutation test

2,000 draws per null. Holm correction over 8 tests (4 languages, 2 keys) on the frequency-binned null.

| Key | Language | Observed | Value-shuffle percentile (p) | Binned percentile (p) | Binned Holm p |
|---|---|---|---|---|---|
| yajnadevam-2024-11-30 | sanskrit | -0.294 | 60.3 (0.3973) | 94.0 (0.0605) | 0.484 |
| yajnadevam-2024-11-30 | tamil | -2.429 | 77.8 (0.2224) | 86.9 (0.1319) | 0.789 |
| yajnadevam-2024-11-30 | english | -0.629 | 46.7 (0.5332) | 73.2 (0.2684) | 0.952 |
| yajnadevam-2024-11-30 | sumerian | -2.219 | 15.6 (0.8446) | 21.6 (0.7841) | 1.000 |
| yajnadevam-2026-07-15 | sanskrit | -0.299 | 68.3 (0.3173) | 93.8 (0.0625) | 0.484 |
| yajnadevam-2026-07-15 | tamil | -2.461 | 80.5 (0.1954) | 86.9 (0.1314) | 0.789 |
| yajnadevam-2026-07-15 | english | -0.633 | 49.4 (0.5062) | 76.2 (0.2379) | 0.952 |
| yajnadevam-2026-07-15 | sumerian | -2.228 | 15.7 (0.8431) | 22.4 (0.7761) | 1.000 |

## Blind search

10 searches of 2,000 swaps per language, key `yajnadevam-2026-07-15`; scores are on the half of the texts the search did not see.

| Language | Search ceiling (held-out) | Claimed table (held-out) | Gap, search minus table |
|---|---|---|---|
| sanskrit | +0.284 ± 0.036 | -0.297 | +0.581 ± 0.040 |
| tamil | -0.003 ± 0.022 | -2.443 | +2.440 ± 0.061 |
| english | +0.242 ± 0.037 | -0.641 | +0.882 ± 0.038 |
| sumerian | +0.242 ± 0.063 | -2.234 | +2.476 ± 0.088 |

## Best searched table of each language, scored under every language

Rows: language the table was searched on. Columns: language of the trigram that scores it (whole corpus, bits per phoneme).

| Searched on | sanskrit | tamil | english | sumerian |
|---|---|---|---|---|
| sanskrit | +0.389 | -1.901 | -0.498 | -1.005 |
| tamil | -0.223 | +0.126 | -0.346 | -0.914 |
| english | -0.177 | -2.702 | +0.337 | -1.014 |
| sumerian | -0.112 | -3.622 | -0.659 | +0.470 |

## Reading

Against frequency-matched shuffles the current table sits at percentile sanskrit 94, tamil 87, english 76, sumerian 22. None is significant after Holm correction. The search ceiling lies above the claimed table by sanskrit +0.58, tamil +2.44, english +0.88, sumerian +2.48 bits; the largest difference between languages is 1.90 bits against a typical spread over searches of 0.06, so the ceiling differs between languages by more than its spread.

## Limits

Consonant skeletons erase much of what separates languages. The claimed table's values are Sanskrit consonants, so against another language part of its gap to the ceiling may come from a poor fit of its values to that language rather than from the inscriptions; this was not separated. Four languages, one search method, one key. The English and Sumerian letter maps are mine and are given in the script. Settings: seed 20261116, runtime 1.2 min, quick run: False.
