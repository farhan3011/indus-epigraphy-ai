# Milestone 3 report: predicting a hidden sign

Small neural models are trained to fill one hidden sign from the rest of its inscription and compared with the Milestone 2 n-grams on the same task. All models use the same 5 folds of the 1,548 EBUDS texts and score the same 7,000 trials: every position of every held-out text, ranked over the same 377 candidate signs. **The bar is the best n-gram, the Kneser-Ney trigram**, not the bigram.

The n-grams reproduce Milestone 2 (top-1 bigram 38.0%, Witten-Bell trigram 40.2%, Kneser-Ney trigram 42.2%; Milestone 2: 38.0%, 40.2%, 42.2%), so the folds, candidates and trials are the same. Tied candidates are broken by sign number, so the ranking is the same on every machine.

## Results

| Model | Top-1 | Top-5 | Bits per hidden sign | Conformal set (LAC): coverage, mean size | Conformal set (APS): coverage, mean size | Calibration error (ECE) |
|---|---|---|---|---|---|---|
| Most frequent signs (no context) | 10.2% | 23.5% | 6.77 | 90.4%, 134 signs | 90.0%, 131 signs | 0.003 |
| Bigram (Milestone 2) | 38.0% | 62.4% | 4.40 | 90.4%, 71 signs | 91.1%, 107 signs | 0.023 |
| Bigram, calibrated | 38.0% | 62.4% | 4.39 | 90.3%, 71 signs | 91.1%, 106 signs | 0.027 |
| Witten-Bell trigram (Milestone 2) | 40.2% | 63.0% | 4.40 | 89.9%, 68 signs | 90.6%, 132 signs | 0.081 |
| Witten-Bell trigram, calibrated | 40.2% | 63.0% | 4.33 | 89.8%, 67 signs | 90.4%, 125 signs | 0.024 |
| Kneser-Ney trigram (best n-gram, Milestone 2) | 42.2% | 64.5% | 4.20 | 90.4%, 65 signs | 89.8%, 178 signs | 0.078 |
| Kneser-Ney trigram, calibrated | 42.2% | 64.5% | 4.15 | 90.4%, 63 signs | 89.8%, 169 signs | 0.019 |
| Neural model, raw | 32.2% ± 0.5 | 53.0% ± 1.1 | 5.14 | 90.0%, 93 signs | 89.2%, 103 signs | 0.065 |
| Neural model, calibrated | 32.2% ± 0.5 | 53.0% ± 1.1 | 5.03 | 90.1%, 96 signs | 89.8%, 101 signs | 0.025 |
| Mixture of neural model and Kneser-Ney trigram, fitted weight, calibrated | 42.4% ± 0.1 | 64.8% ± 0.1 | 4.11 | 90.2%, 59 signs | 89.4%, 125 signs | 0.019 |
| Residual model: Kneser-Ney trigram + learned correction, calibrated | 42.1% ± 0.1 | 64.6% ± 0.1 | 4.14 | 90.3%, 62 signs | 90.0%, 167 signs | 0.019 |

Neural rows: mean ± sd over 3 training seeds. Conformal sets aim to contain the true sign 90.0% of the time; a smaller set at the same coverage is the more useful model.

Top-1 difference from the Kneser-Ney trigram, in points, with a 95% interval from resampling whole test texts (per-trial hits averaged over seeds):

| Model | Difference | 95% interval |
|---|---|---|
| Neural model, calibrated | -10.0 | [-11.0, -9.0] |
| Mixture of neural model and Kneser-Ney trigram, fitted weight, calibrated | +0.2 | [-0.1, +0.4] |
| Residual model: Kneser-Ney trigram + learned correction, calibrated | -0.1 | [-0.3, +0.2] |

Over 5 random fold partitions the n-grams vary little: Kneser-Ney trigram top-1 42.0% ± 0.3, Witten-Bell trigram 40.3% ± 0.3, bigram 38.2% ± 0.2. The neural rows use the first partition only.

## Reading the numbers

- Neural model is reliably worse than the Kneser-Ney trigram (-10.0 points, 95% interval -11.0 to -9.0); Mixture of neural model and Kneser-Ney trigram is statistically tied with the Kneser-Ney trigram (+0.2 points, 95% interval -0.1 to +0.4); Residual model: Kneser-Ney trigram + learned correction is statistically tied with the Kneser-Ney trigram (-0.1 points, 95% interval -0.3 to +0.2). The neural model cannot replace the n-gram on about 1,200 short training texts per fold, and what it adds on top is small: the neighbouring signs carry almost everything a model can use.
- Calibration (the proposal's "validation confidence ratings"): ECE is the average gap between how confident the top guess is and how often it is right; 0 is perfect. `reliability.png` shows it per confidence band.
- Conformal sets replace the earlier "90% probability mass" set, which did not contain the true sign 90% of the time and whose size could not be compared across models. They are calibrated on the inner split of each training fold with models fitted on the rest of it and applied to models retrained on the whole fold, so coverage is approximate; the measured coverage above is the check. The bigram's 90%-mass set is kept for the comparison with Yadav et al. 2010 (74%): 86.1%, mean 44 signs.

## Damaged inscriptions

The final neural model, trained on all 1,548 texts, was run on the 337 distinct single-line corpus texts with exactly one illegible sign. They have no answer key, so this shows what the model offers an epigrapher, not how accurate it is. Its top guess carries a median confidence of 19.9%; 63 texts get a top guess above 50% and 13 above 80%. The most frequent top guesses are 342, 211, 65, 67, 267. Per-text suggestions are in `data/derived/` on the machine that ran this (not published: they contain corpus texts).

## Use it

```bash
python sequence_analysis/masked_model.py predict "267 99 ? 342"   # reading order, ? marks each gap
```

Several gaps are masked together and each is predicted on its own; there is no joint search over gap combinations.

## Settings

- Neural model: one configuration per fold is chosen from 8 fixed in advance (transformer or bidirectional LSTM; width 32 or 64; 1 or 2 layers), by calibrated log loss on a 10% inner split of the training fold. Chosen: transformer width 64, 2 layers (2 folds); lstm width 64, 1 layer (2 folds); lstm width 64, 2 layers (1 folds). Inputs are the signs in reading order between start and end tokens, with position counted from both ends; a context sign unseen in training becomes an unknown token. Every position of every training text is a masked example in every epoch. Early stopping on the inner split (patience 10, median 9 epochs) sets the epoch count; the model is then retrained on the whole training fold for that many epochs, so it sees the same texts as the n-grams.
- Mixture: weight on the neural model fitted on the inner split (median 0.15), then a temperature after mixing.
- Residual model: a width-32, one-layer transformer whose output layer starts at zero and is added to the trigram's log-probabilities, so it starts exactly at the Kneser-Ney trigram and can only learn corrections. Its training priors come from trigrams fitted without the text being predicted (5-fold cross-fitting inside the training fold).
- Temperatures (median): bigram 1.1, Kneser-Ney trigram 1.15, neural 1.15. AdamW, learning rate 0.002, weight decay 0.01, dropout 0.2. Trained on CPU; wall-clock time for the whole run 22 minutes.
