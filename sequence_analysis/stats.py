"""Frequency, positional and association statistics for sign sequences, and the direction test.

    python sequence_analysis/stats.py     # self-check
"""
import math
from collections import Counter

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2


def entropy(counts, miller_madow=False):
    """Shannon entropy in bits of a count vector. The plug-in estimate is biased low on small
    samples; Miller-Madow adds (observed types - 1) / (2 n ln 2)."""
    c = np.asarray([v for v in counts if v > 0], float)
    n = c.sum()
    h = float(-(c / n * np.log2(c / n)).sum())
    return h + (len(c) - 1) / (2 * n * math.log(2)) if miller_madow else h


def coverage(counts, frac=0.8):
    """How many of the most frequent types are needed to cover `frac` of the tokens."""
    c = np.sort(np.asarray(list(counts), float))[::-1]
    return int(np.searchsorted(np.cumsum(c), frac * c.sum() - 1e-9) + 1)


def mutual_information(texts):
    """Mutual information in bits between adjacent signs (boundaries not included)."""
    pairs = Counter(p for t in texts for p in zip(t, t[1:]))
    first, second = Counter(), Counter()
    for (a, b), c in pairs.items():
        first[a] += c
        second[b] += c
    return entropy(first.values()) + entropy(second.values()) - entropy(pairs.values())


def g2(k11, k12, k21, k22):
    """Log-likelihood ratio statistic (Dunning 1993) for a 2x2 table; ~chi-square with 1 d.o.f."""
    n = k11 + k12 + k21 + k22
    total = 0.0
    for o, row, col in ((k11, k11 + k12, k11 + k21), (k12, k11 + k12, k12 + k22),
                        (k21, k21 + k22, k11 + k21), (k22, k21 + k22, k12 + k22)):
        if o:
            total += o * math.log(o * n / (row * col))
    return 2 * total


def benjamini_hochberg(p):
    """q-values controlling the false discovery rate."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


MIN_PAIR_COUNT = 5  # below this the chi-square approximation to G2 is unreliable


def pair_table(texts):
    """Every observed adjacent pair (reading order): (first, second, count, expected, G2, p, q).

    q (Benjamini-Hochberg) is computed over pairs seen at least MIN_PAIR_COUNT times; rarer pairs
    get q = nan. Testing every observed pair is not calibrated: most have expected counts below 1,
    and a corpus with all signs permuted still yields about a hundred 'significant' pairs."""
    pairs = Counter(p for t in texts for p in zip(t, t[1:]))
    first, second = Counter(), Counter()
    for (a, b), c in pairs.items():
        first[a] += c
        second[b] += c
    n = sum(pairs.values())
    rows = []
    for (a, b), k in pairs.items():
        stat = g2(k, first[a] - k, second[b] - k, n - first[a] - second[b] + k)
        rows.append((a, b, k, first[a] * second[b] / n, stat, chi2.sf(stat, 1)))
    tested = [i for i, r in enumerate(rows) if r[2] >= MIN_PAIR_COUNT]
    q = dict(zip(tested, benjamini_hochberg([rows[i][5] for i in tested]))) if tested else {}
    rows = [(*r, q.get(i, float("nan"))) for i, r in enumerate(rows)]
    return sorted(rows, key=lambda r: -r[2]), n


def significant_pairs(table):
    """(attracting, avoiding) counts among tested pairs with q < 0.05."""
    sig = [r for r in table if r[6] < 0.05]
    return sum(r[2] > r[3] for r in sig), sum(r[2] < r[3] for r in sig)


def zipf_mandelbrot(counts):
    """Least-squares fit of ln f = a - b ln(r + c) to the rank-frequency curve. Returns (a, b, c)."""
    f = np.sort(np.asarray(list(counts), float))[::-1]
    r = np.arange(1, len(f) + 1)
    (a, b, c), _ = curve_fit(lambda r, a, b, c: a - b * np.log(r + c), r, np.log(f), p0=(10, 1.5, 10),
                             bounds=([0, 0, 0], [50, 10, 500]))
    return float(a), float(b), float(c)


# --- writing direction (Ashraf & Sinha 2018) ---------------------------------------------------

def gini(p):
    """G = 2 sum(i p_i) / (N sum(p)) - (N + 1) / N, with p in non-decreasing order."""
    p = np.sort(np.asarray(p, float))
    n = len(p)
    return float(2 * (np.arange(1, n + 1) * p).sum() / (n * p.sum()) - (n + 1) / n)


SIGNS = np.arange(418)  # 0 (illegible) plus Mahadevan signs 1-417


def asymmetry(left, right, alphabet=SIGNS):  # alphabet=None: signs seen at either end (Ashraf & Sinha's usage)
    """Directional asymmetry of the signs at the two physical ends of each line.

    left, right: integer arrays, one terminal sign per line. Both ends are scored over one fixed
    alphabet (the whole sign list by default), so values are comparable across groups, resamples
    and noise levels; an alphabet taken from the data would shrink or grow with the sample.
    Returns (delta_G, delta_S): left-to-right scripts give delta_G < 0 and delta_S > 0, right-to-left
    scripts the opposite, because the reading-final position is the more predictable one."""
    if alphabet is None:
        alphabet = np.unique(np.concatenate([left, right]))
    cl = np.bincount(np.searchsorted(alphabet, left), minlength=len(alphabet))
    cr = np.bincount(np.searchsorted(alphabet, right), minlength=len(alphabet))
    gl, gr, sl, sr = gini(cl), gini(cr), entropy(cl), entropy(cr)
    return 2 * (gl - gr) / (gl + gr), 2 * (sl - sr) / (sl + sr)


def terminals(lines):
    """(left signs, right signs) for lines given in physical left-to-right order."""
    return np.array([l[0] for l in lines]), np.array([l[-1] for l in lines])


def direction_test(lines, n_resample=1000, seed=0):
    """delta_G / delta_S with a bootstrap standard error over lines and two nulls:
    shuffle (signs permuted within each line) and mirror (each line flipped with probability 1/2).

    A standard error is reported, not a percentile interval: the Gini index of a sparse count vector
    is biased, a resample repeats that bias, and the percentile interval then misses the estimate.
    Inference rests on the nulls."""
    rng = np.random.default_rng(seed)
    left, right = terminals(lines)
    n = len(lines)
    observed = asymmetry(left, right)
    boot = np.array([asymmetry(left[i], right[i]) for i in rng.integers(0, n, (n_resample, n))])
    shuffle, mirror = [], []
    for _ in range(n_resample):
        ends = [rng.choice(len(l), 2, replace=False) for l in lines]
        shuffle.append(asymmetry(np.array([l[i] for l, (i, _) in zip(lines, ends)]),
                                 np.array([l[j] for l, (_, j) in zip(lines, ends)])))
        flip = rng.random(n) < 0.5
        mirror.append(asymmetry(np.where(flip, right, left), np.where(flip, left, right)))
    shuffle, mirror = np.array(shuffle), np.array(mirror)
    out = {"lines": n}
    for k, name in enumerate(("delta_G", "delta_S")):
        out[name] = {"value": float(observed[k]), "bootstrap_se": float(boot[:, k].std(ddof=1)),
                     "shuffle_null_95": np.percentile(shuffle[:, k], [2.5, 97.5]).tolist(),
                     "mirror_null_95": np.percentile(mirror[:, k], [2.5, 97.5]).tolist(),
                     "p_shuffle": float(((np.abs(shuffle[:, k]) >= abs(observed[k])).sum() + 1) / (n_resample + 1))}
    return out


def selfcheck():
    assert abs(entropy([1, 1, 1, 1]) - 2) < 1e-12 and entropy([5]) == 0
    assert entropy([3, 1], miller_madow=True) > entropy([3, 1])
    assert coverage([50, 30, 10, 10]) == 2 and coverage([1, 1, 1, 1, 1]) == 4
    assert g2(10, 10, 10, 10) < 1e-9 and g2(20, 0, 0, 20) > 50
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-5)
    assert abs(gini([1, 1, 1, 1])) < 1e-12 and abs(gini([0, 0, 0, 1]) - 0.75) < 1e-12
    assert abs(mutual_information([(1, 2)] * 5 + [(3, 4)] * 5) - 1) < 1e-12
    # a toy right-to-left script: every text ends (physical left) in sign 9, starts anywhere
    rng = np.random.default_rng(1)
    rtl = [(9, *rng.integers(1, 9, 3)) for _ in range(300)]
    dg, ds = asymmetry(*terminals(rtl))
    assert dg > 0 > ds
    dg2, ds2 = asymmetry(*terminals([l[::-1] for l in rtl]))
    assert abs(dg + dg2) < 1e-12 and abs(ds + ds2) < 1e-12  # mirroring flips the sign exactly
    res = direction_test(rtl, n_resample=100)
    lo, hi = res["delta_G"]["shuffle_null_95"]
    assert lo < 0 < hi < res["delta_G"]["value"]
    assert 0 < res["delta_G"]["bootstrap_se"] < 0.1 and res["delta_G"]["p_shuffle"] < 0.02


if __name__ == "__main__":
    selfcheck()
    print("ok")
