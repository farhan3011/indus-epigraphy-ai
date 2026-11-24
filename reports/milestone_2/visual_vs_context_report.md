# Do signs that look alike behave alike? (exploratory)

Signs with at least 20 tokens in the 1,548 EBUDS texts: 76 signs, 2,850 pairs. Visual similarity is the cosine of DINOv2-small embeddings of the clean IM-417 drawings (one per sign); a raw-pixel cosine is the baseline. Context similarity is 1 minus the mean Jensen-Shannon divergence of three distributions: the sign before (text start included), the sign after (text end included) and the relative position in the text (5 bins).

| Measure | Spearman correlation with context similarity | Mantel p (sign labels permuted) |
|---|---|---|
| DINOv2 visual similarity | +0.053 | 0.1811 |
| Pixel similarity | +0.001 | 0.4970 |

The 50 most visually similar pairs have a mean context similarity of 0.560, against 0.444 ± 0.017 for random sets of 50 pairs (p = 0.0001).

Sensitivity to the token threshold:

| Minimum tokens | Signs | DINOv2 Spearman | p | Pixel Spearman | p |
|---|---|---|---|---|---|
| 10 | 114 | +0.023 | 0.3088 | +0.091 | 0.0538 |
| 50 | 38 | +0.102 | 0.0691 | +0.043 | 0.2895 |

## Candidate variant pairs (20 pairs in the top 5% on both)

| Signs (Mahadevan) | Visual | Context |
|---|---|---|
| 104 and 106 | 0.954 | 0.788 |
| 70 and 72 | 0.939 | 0.819 |
| 67 and 70 | 0.945 | 0.772 |
| 294 and 296 | 0.946 | 0.763 |
| 102 and 104 | 0.947 | 0.755 |
| 65 and 70 | 0.938 | 0.756 |
| 65 and 67 | 0.926 | 0.793 |
| 387 and 389 | 0.921 | 0.801 |
| 59 and 72 | 0.954 | 0.701 |
| 65 and 72 | 0.909 | 0.771 |
| 67 and 72 | 0.907 | 0.790 |
| 12 and 15 | 0.942 | 0.678 |
| 48 and 53 | 0.910 | 0.735 |
| 70 and 387 | 0.913 | 0.715 |
| 102 and 106 | 0.897 | 0.761 |
| 59 and 70 | 0.963 | 0.654 |
| 267 and 284 | 0.937 | 0.659 |
| 53 and 67 | 0.905 | 0.722 |
| 87 and 102 | 0.930 | 0.659 |
| 53 and 389 | 0.891 | 0.747 |

## Reading

- There is no reliable association between how alike two signs look (DINOv2) and how alike they are used (Spearman +0.053, Mantel p = 0.1811); raw pixels give +0.001 (p = 0.4970).
- The most similar-looking pairs are more alike in use than random pairs (p = 0.0001).
- The candidate pairs are where to look for sign-list variants; each needs checking by eye against carved examples.

## Limits

- Compound signs contain other signs, so they look alike for a reason that has nothing to do with variation.
- The drawings are normalised sign-list forms, not carved forms.
- A correlation near zero would mean visual form does not predict use, which is also a result. A positive one does not show that two signs are variants: signs of a class (for example the stroke signs) can look and behave alike and still be distinct.

Settings: 10,000 label permutations, 10,000 random pair sets, seed 20261124; run time 0.2 minutes.
