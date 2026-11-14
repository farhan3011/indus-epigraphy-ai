# Milestone 3 extension: more than one missing sign, unknown-length damage, miscopies

Milestone 3 filled one hidden sign and filled several gaps independently of each other. This extension adds a joint search over all gaps, damaged stretches whose length is not known, a check for a sign that does not fit its text, and retrieval of parallel texts. Same 1,548 EBUDS texts and the same 5 folds as Milestones 2 and 3; every number below is on held-out texts. The joint search runs on the Kneser-Ney trigram, the best single model of Milestone 3.

## Two missing signs

Two signs are hidden in each held-out text: an adjacent pair (texts of 3+ signs) and a separated pair (4+ signs).

| Method | Pair | Trials | Both signs right | Signs right | True pair in the 5 best |
|---|---|---|---|---|---|
| Kneser-Ney trigram, joint search | adjacent | 1,290 | 14.3% | 26.1% | 27.8% |
| Kneser-Ney trigram, greedy left to right | adjacent | 1,290 | 8.1% | 18.2% | 8.1% |
| Neural model, each gap on its own (Milestone 3) | adjacent | 1,290 | 2.4% | 17.8% | 7.5% |
| Kneser-Ney trigram, joint search | separated | 1,006 | 25.5% | 48.4% | 44.4% |
| Kneser-Ney trigram, greedy left to right | separated | 1,006 | 13.2% | 35.8% | 13.2% |
| Neural model, each gap on its own (Milestone 3) | separated | 1,006 | 14.1% | 37.6% | 28.1% |
| Kneser-Ney trigram, joint search | all | 2,296 | 19.3% | 35.9% | 35.1% |
| Kneser-Ney trigram, greedy left to right | all | 2,296 | 10.3% | 25.9% | 10.3% |
| Neural model, each gap on its own (Milestone 3) | all | 2,296 | 7.5% | 26.4% | 16.6% |

Joint search minus the neural model, both signs right: +11.7 points (95% interval +10.0 to +13.5, resampling whole test texts).

Joint search minus greedy search, both signs right: +8.9 points (95% interval +7.4 to +10.4, resampling whole test texts).

The neural model was trained with one sign hidden at a time, so two hidden signs are outside what it was trained on; this is the comparison with how Milestone 3 handled several gaps, not a verdict on neural models.

## A damaged stretch of unknown length

A run of 0 to 3 signs is removed from each held-out text of 3+ signs (0 means nothing is missing) and the model is told only where the damage is. It scores every fill of every length as a whole text, end included, so lengths compete directly.

- Length right: 53.6% of 1,290 trials (always guessing the commonest length: 32.6%).
- Mean probability given to the true length: 46.3%.
- Whole stretch restored exactly: 38.8%; among the 5 best fills: 55.7%.

The length guess is the length with the most total probability; the restored stretch is the single most probable fill. They can disagree: one complete text can be the likeliest single answer while most of the probability is spread over many longer fills.

| True length | Trials | Length right | Stretch restored exactly | Guessed 0 / 1 / 2 / 3 |
|---|---|---|---|---|
| 0 | 414 | 77.5% | 89.9% | 321 / 75 / 17 / 1 |
| 1 | 420 | 60.7% | 25.2% | 92 / 255 / 67 / 6 |
| 2 | 281 | 36.3% | 7.8% | 55 / 114 / 102 / 10 |
| 3 | 175 | 7.4% | 0.6% | 40 / 67 / 55 / 13 |

## One miscopied sign

One sign of each held-out text of 3+ signs is replaced by a different sign drawn by corpus frequency. Each sign is then scored by its probability given all the others; the least expected sign is flagged.

- The flagged sign is the replaced one in 61.7% of 1,290 texts (chance: 22.1%).
- Telling a corrupted text from its clean original by its least expected sign: AUC 0.747 (0.5 is chance).
- At a threshold that flags 5% of clean texts, 16.2% of corrupted texts are flagged.
- Once the position is known, the original sign is the model's first suggestion in 43.2% and among the first five in 64.9%. Found and repaired at the first guess: 31.0%.

## How far the neural model looks

For each of 6,553 positions, each context sign was replaced in turn by the unknown-sign token and the drop in the probability of the model's top guess recorded. The most influential sign is next to the gap in 89.7% of positions, two signs away in 6.8% and further in 3.5%. Descriptive only: this is the final model on texts it was trained on.

## Reading the numbers

- These are restoration scores on texts whose answer is known. They say how often the model agrees with the corpus, not that a restored sign on a really damaged seal is right.
- The corruption test replaces a sign with a frequency-drawn one. A real copyist's slip is more likely a similar-looking sign, which would be harder to catch; the detection rate is an upper bound for that case.
- A rare but genuine sign looks the same to the model as a miscopy. The flag is a prompt to look at the object again, not a correction.
- Probabilities from the joint search are shares of the candidates the search kept, so they run slightly high.

## Use it

```bash
python sequence_analysis/restore.py fill "267 ? ? 342"   # reading order; ? = one missing sign
python sequence_analysis/restore.py fill "267 * 342"     # * = a damaged stretch of 0-3 signs
python sequence_analysis/restore.py check "267 99 342"   # how expected each sign is
```

Settings: beam 200, 4,000 partial texts kept after a gap, Kneser-Ney trigram (discount 0.75). Run time 7 minutes on a laptop CPU.
