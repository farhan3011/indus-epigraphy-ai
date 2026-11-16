# Positive control on a deciphered script: Linear B as Greek (exploratory, not pre-registered)

The Milestone 4 positive control enciphers Sanskrit with the claim's own table, so the true table is one this project made. Linear B was deciphered independently (Ventris and Chadwick, 1952) and the Mycenaean reading is accepted, so it is a table nobody here chose. The same test is run on it, unchanged: a table is scored for how much its output looks like the target language, against 10,000 shuffled versions of itself. If the test is any good, the true table should stand out.

## Data

- Linear B words: `linear_b-greek.cog` from NeuroDecipher (github.com/j-luo93/NeuroDecipher, commit 480bad24; no licence stated; only column 1 is read, not the Greek cognates, which hold the answers). 919 rows, 918 distinct words of two or more syllabograms kept, 3,450 sign tokens, 69 distinct signs. One text is one word.
- Greek target: the Iliad and Odyssey from PerseusDL canonical-greekLit (commit dacb0490, CC BY-SA 4.0), 190,335 words, one word per target line.
- Control language for the scores: Sanskrit (DCS training split), as in the Indus test.
- Fetched at run time, never committed; this report holds aggregates only. The better source, DAMOS tablet lines (CC BY-NC-SA 4.0), has no bulk export.

## How Linear B and Greek are put in one alphabet

The harness scores strings over 16 consonant letters. A syllabogram's value is the consonant series of its transliteration (vowels dropped, so `pte` is `pt`, `a` is empty); Greek is reduced the same way (accents, breathings and vowels dropped; ξ is ks, ψ is ps).

| Series | Harness letter | Greek letters |
|---|---|---|
| d | d | δ |
| j | y | - |
| k | k | κ γ χ |
| m | m | μ |
| n | n | ν |
| p | p | π β φ |
| q | c | - |
| r | r | ρ λ |
| s | s | σ ς |
| t | t | τ θ |
| w | v | - |
| z | j | ζ |

Losses that work against the true table: the j, q and w series have no Greek counterpart (labiovelars and digamma were lost by Homer's time), Linear B writes almost no syllable-final consonants, Homeric Greek is centuries later than the tablets, and vowels are discarded.

## Results

**1. The true table against shuffles** (10,000 draws; the primary score is order beyond letter frequency, in bits per phoneme). Observed -2.516.

| Null | Null mean (sd) | Percentile | p |
|---|---|---|---|
| value shuffle | -2.685 (0.385) | 67.6 | 0.3244 |
| binned 10 | -2.772 (0.114) | 99.0 | 0.0101 |

Other scorers, value-shuffle p: sequence 0.3244, margin tamil 0.5812, naive bayes 0.4375.

**2. How wrong can a table be and still pass.** A share of the true table's values replaced by other values; 20 corrupted tables per row, 1,000 draws. Share with p < 0.05 on the primary score:

| Values replaced | Value-shuffle null | Binned null |
|---|---|---|
| 10% | 0% | 90% |
| 20% | 15% | 50% |
| 30% | 10% | 35% |
| 50% | 5% | 30% |

**3. Size.** This corpus has 3,450 sign tokens in 918 texts. The Indus test scored 5,584 tokens in 1,318 texts. This corpus is the smaller one, so a pass here is the stronger result, and a fail could reflect size alone.

**4. Wrong language.** The same table scored with Sanskrit as the target and Greek as the control: observed -0.205, percentile 100.0 against the binned null, p 0.0001.

## Reading

The true Linear B table scores at percentile 99.0 of frequency-matched shuffles (p 0.0101) and percentile 67.6 of fully shuffled tables (p 0.3244): the test detects it under binned 10 only. With 10% of values replaced the detection rate is 90% (binned null), and with 30% replaced 35%. Scored against the wrong language the table is still significant, which is an open problem for the method. Taken together this is at best a partial pass: the true table is detected only against frequency-matched shuffles, and a table that is right for Greek also scores significantly for Sanskrit, so a significant result from this test is not evidence that a table is right for the language it was scored against.

## Limits

A pass shows the method detects a real decipherment despite the losses above; a fail does not show the method is useless, only that it cannot see this table on a word list of this size under this reduction. Words are not running text, the target is later Greek, and consonant-only scoring discards what a syllabary records. Not pre-registered and outside the corrected family of Milestone 4 tests.

Settings: seed 20261116, runtime 1.3 min, quick run: False.
