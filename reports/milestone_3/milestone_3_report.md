# Milestone 3 report: predicting a hidden sign

A small masked-sign transformer (297,497 parameters) is trained to fill one hidden sign from the rest of its inscription, and compared with the Milestone 2 bigram on the same task. Both use the same 5 folds of the 1,548 EBUDS texts and score the same 7,000 trials: every position of every held-out text, ranked over the same 377 candidate signs. Early stopping and the temperature that calibrates confidence are chosen on a 10% inner split of each training fold; the test fold is only scored. Network rows are the mean ± sd over 3 training seeds.

The bigram reproduces Milestone 2 exactly (top-1 38.0% here, 38.0% there), so the folds, candidates and trials are the same.

## Results

| Model | Top-1 | Top-5 | Bits per hidden sign | True sign inside the 90% set (its mean size) | Calibration error (ECE) |
|---|---|---|---|---|---|
| Most frequent signs (no context) | 10.2% | 23.5% | 6.77 | 90.4% (134 signs) | 0.003 |
| Bigram, Milestone 2 | 38.0% | 62.4% | 4.40 | 86.1% (44 signs) | 0.023 |
| Bigram, temperature-calibrated | 38.0% | 62.4% | 4.39 | 88.9% (61 signs) | 0.027 |
| Masked transformer | 30.0% ± 1.1 | 51.3% ± 1.1 | 5.44 | 76.2% (35 signs) | 0.100 |
| Masked transformer, temperature-calibrated | 30.0% ± 1.1 | 51.3% ± 1.1 | 5.23 | 87.4% (82 signs) | 0.038 |
| Average of calibrated bigram and transformer | 38.4% ± 0.1 | 62.0% ± 0.0 | 4.47 | 91.3% (82 signs) | 0.095 |

Top-1 difference from the bigram, with a 95% interval from resampling whole test texts (seed 0): transformer -9.2 points [-10.4, -8.1]; average +0.3 points [-0.3, +0.9].

For context, Milestone 2 also ran a trigram backoff (top-1 40.1%) and Kneser-Ney bigram (38.7%). Yadav et al. 2010 report that the true sign falls inside the 90% candidate set 74% of the time; that is the column it compares with, not top-1.

## Reading the numbers

- The transformer's top-1 is reliably lower than the bigram's. With about 1,100 training texts per fold it does not even match the neighbour statistics a bigram counts directly, and averaging the two only ties the bigram, so the wider context the network can read adds nothing measurable on this corpus.
- Calibration (the proposal's "validation confidence ratings"): ECE is the average gap between how confident the top guess is and how often it is right; 0 is perfect. `reliability.png` shows the same thing per confidence band.
- The 90% set is the smallest group of signs whose probabilities add up to 90%. It is the useful output for an epigrapher: a short list that should contain the right sign nine times in ten.

## Damaged inscriptions

The final model, trained on all 1,548 texts, was run on the 337 distinct single-line corpus texts with exactly one illegible sign. They have no answer key, so this shows what the model offers an epigrapher, not how accurate it is. Its top guess carries a median confidence of 17.7%; 49 texts get a top guess above 50% and 6 above 80%. The 90% candidate set holds a median of 94 signs. The most frequent top guesses are 342, 65, 267, 211, 87. Per-text suggestions are in `data/derived/` on the machine that ran this (not published: they contain corpus texts).

## Use it

```bash
python sequence_analysis/masked_model.py predict "267 99 ? 342"   # reading order, ? marks each gap
```

Several gaps are masked together and each is predicted on its own; there is no joint search over gap combinations.

## Settings

Transformer: 2 layers, width 96, 4 heads, dropout 0.2, AdamW (learning rate 0.002, weight decay 0.01), one random masked sign per text per epoch, early stopping after 40 epochs without improvement (median 70 epochs). Inputs are the signs in reading order between start and end tokens; a context sign unseen in a training fold becomes an unknown token. Temperatures chosen: bigram median 1.1, transformer median 1.3. Trained on CPU. The configuration was fixed in advance; no search over settings was run.
