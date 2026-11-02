# Milestone 1: re-ranking with a sign-sequence model

Synthetic test seals regenerated with corpus texts disjoint across train, val and test (`synthetic_generator.split_of`); the detector and classifier bank are the ones from the main report.

YOLOv8n output, each sign's top 5 re-ranked by beam search with the best Milestone 2 n-gram (Kneser-Ney trigram) trained on the corpus texts of the training split only; synthetic train, val and test seals use disjoint corpus texts. Weight chosen on the val seals that carry corpus texts: 1 (0 = no re-ranking).

| Test seals | Seals | Sign error rate, top-1 | Sign error rate, re-ranked | Exact match, top-1 | Exact match, re-ranked |
|---|---|---|---|---|---|
| All | 400 | 0.224 [0.201, 0.249] | 0.257 [0.231, 0.285] | 0.495 [0.448, 0.542] | 0.460 [0.410, 0.507] |
| Real corpus texts | 188 | 0.170 [0.134, 0.208] | 0.134 [0.102, 0.166] | 0.585 [0.516, 0.660] | 0.665 [0.596, 0.734] |
| Random sign strings | 212 | 0.270 [0.236, 0.305] | 0.362 [0.325, 0.396] | 0.415 [0.349, 0.481] | 0.278 [0.222, 0.344] |

Half the synthetic seals carry random sign strings, which a sequence model cannot help and may hurt; real inscriptions behave like the corpus-text rows.

**Sign size.** Penalising candidates whose drawn height does not fit the box (weight 128, chosen on val seals) changes the sign error rate on all test seals from 0.224 [0.201, 0.249] to 0.202 [0.180, 0.225].
