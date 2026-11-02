"""Hand-written n-gram models for a small closed vocabulary. NLTK is deliberately not used: its
Witten-Bell bottoms out in the MLE unigram, so any sign missing from a training fold gets
probability 0 and perplexity is infinite.

Events predicted: the signs plus end-of-text. <s> is a context only, never predicted.

    python sequence_analysis/ngram.py     # self-check against a hand-worked example
"""
import math
import random
from collections import Counter, defaultdict

import numpy as np

BOS, EOS = "<s>", "</s>"


class WittenBell:
    """Witten-Bell smoothing (Chen & Goodman 1999, eq. 6-8), zeroth order uniform over the events.

    backoff=False: interpolated. backoff=True: Katz-style backoff with the Witten-Bell discount,
    the form SRILM uses and Yadav et al. 2010 report."""

    def __init__(self, texts, vocab, order=2, backoff=False, min_count=1):
        """min_count: drop n-grams of order 3 and up seen fewer times (SRILM's default is 2)."""
        self.order, self.backoff, self.n_events = order, backoff, len(vocab) + 1
        self.counts = defaultdict(Counter)  # history tuple -> Counter of next events
        for t in texts:
            s = [BOS, *t, EOS]
            for i in range(1, len(s)):
                for k in range(min(order, i + 1)):
                    self.counts[tuple(s[i - k:i])][s[i]] += 1
        if min_count > 1:
            for h in [h for h in self.counts if len(h) >= 2]:
                kept = Counter({w: c for w, c in self.counts[h].items() if c >= min_count})
                if kept:
                    self.counts[h] = kept
                else:
                    del self.counts[h]
        self.total = {h: sum(c.values()) for h, c in self.counts.items()}
        self._p, self._alpha = {}, {}

    def prob(self, w, history=()):
        h = tuple(history)
        return self._prob(w, h[max(0, len(h) - self.order + 1):] if self.order > 1 else ())

    def _prob(self, w, h):
        key = (w, h)
        if key in self._p:
            return self._p[key]
        c = self.counts.get(h)
        if not h:
            p = (c[w] + len(c) / self.n_events) / (self.total[h] + len(c))
        elif not c:  # unseen context
            p = self._prob(w, h[1:])
        elif not self.backoff:
            p = (c[w] + len(c) * self._prob(w, h[1:])) / (self.total[h] + len(c))
        elif len(c) == self.n_events:  # every event already seen after h: nothing to back off to
            p = c[w] / self.total[h]
        elif c[w]:
            p = c[w] / (self.total[h] + len(c))
        else:
            p = self.alpha(h) * self._prob(w, h[1:])
        self._p[key] = p
        return p

    def alpha(self, h):
        """Backoff weight: the escape mass, renormalised over events unseen after h."""
        if h not in self._alpha:
            c = self.counts[h]
            unseen_lower = 1 - sum(self._prob(v, h[1:]) for v in c)
            self._alpha[h] = len(c) / (self.total[h] + len(c)) / unseen_lower
        return self._alpha[h]


class AddK:
    """Bigram add-k baseline."""
    order = 2

    def __init__(self, texts, vocab, k=1.0):
        self.k, self.n_events, self.counts = k, len(vocab) + 1, defaultdict(Counter)
        for t in texts:
            s = [BOS, *t, EOS]
            for a, b in zip(s, s[1:]):
                self.counts[a][b] += 1

    def prob(self, w, history=()):
        c = self.counts.get(history[-1], Counter())
        return (c[w] + self.k) / (sum(c.values()) + self.k * self.n_events)


class KneserNey:
    """Interpolated Kneser-Ney with a fixed discount, any order. Lower orders use continuation counts (how many
    distinct signs precede an n-gram); the unigram continuation is Witten-Bell smoothed toward uniform so unseen
    signs keep non-zero probability. order=2 is the bigram used throughout Milestone 2."""

    def __init__(self, texts, vocab, discount=0.75, order=2):
        self.d, self.order = discount, order
        self.counts = defaultdict(Counter)  # full available history (up to order-1 events) -> next event
        left = defaultdict(set)  # (history, event) -> distinct preceding events, for continuation counts
        for t in texts:
            s = [BOS, *t, EOS]
            for i in range(1, len(s)):
                for k in range(1, order):
                    if i - k >= 0:
                        h = tuple(s[i - k:i])
                        if k == order - 1 or i - k == 0:  # highest order available at this position: raw counts
                            self.counts[h][s[i]] += 1
                        if i - k - 1 >= 0:
                            left[(h, s[i])].add(s[i - k - 1])
                if i >= 1:
                    left[((), s[i])].add(s[i - 1])
        self.cont = defaultdict(Counter)
        for (h, w), us in left.items():
            self.cont[h][w] = len(us)
        uni, n_events = self.cont[()], len(vocab) + 1
        n, t = sum(uni.values()), len(uni)
        self.base = lambda w: (uni[w] + t / n_events) / (n + t)
        self.total = {h: sum(c.values()) for h, c in self.counts.items()}
        self.ctotal = {h: sum(c.values()) for h, c in self.cont.items()}

    def prob(self, w, history=()):
        h = tuple(history)[-(self.order - 1):] if self.order > 1 else ()
        return self._prob(w, h, True)

    def _prob(self, w, h, top):
        if not h:
            return self.base(w)
        c, total = (self.counts.get(h), self.total.get(h)) if top else (self.cont.get(h), self.ctotal.get(h))
        if not c:
            return self._prob(w, h[1:], False)
        return max(c[w] - self.d, 0) / total + self.d * len(c) / total * self._prob(w, h[1:], False)


class Positional:
    """No sign-to-sign dependence: P(event | position in the text), Witten-Bell smoothed toward the
    unigram. If a bigram model cannot beat this, 'Markov dependence' is only positional preference."""
    order = 10 ** 6  # ask for the whole history; only its length is used

    def __init__(self, texts, vocab, max_pos=6):
        self.max_pos, self.uni = max_pos, WittenBell(texts, vocab, 1)
        self.counts = defaultdict(Counter)
        for t in texts:
            for i, w in enumerate([*t, EOS]):
                self.counts[min(i, max_pos)][w] += 1

    def prob(self, w, history=()):
        c = self.counts.get(min(len(history) - 1, self.max_pos))  # history starts with <s>
        if not c:
            return self.uni.prob(w)
        return (c[w] + len(c) * self.uni.prob(w)) / (sum(c.values()) + len(c))


class TwoEnded:
    """No sign-to-sign dependence, but position counted from both ends: P(sign | distance from the
    start, distance from the end), Witten-Bell smoothed toward the unigram over signs. It knows each
    text's length, so it predicts signs only, never end-of-text."""

    def __init__(self, texts, vocab, max_pos=6):
        self.max_pos, self.uni = max_pos, WittenBell(texts, vocab, 1)
        self.counts = defaultdict(Counter)
        for t in texts:
            for i, w in enumerate(t):
                self.counts[self.slot(i, len(t))][w] += 1

    def slot(self, i, length):
        return min(i, self.max_pos), min(length - 1 - i, self.max_pos)

    def prob(self, w, i, length):
        u = self.uni.prob(w) / (1 - self.uni.prob(EOS))
        c = self.counts.get(self.slot(i, length))
        if not c:
            return u
        return (c[w] + len(c) * u) / (sum(c.values()) + len(c))


class Uniform:
    order = 1

    def __init__(self, texts, vocab):
        self.p = 1 / (len(vocab) + 1)

    def prob(self, w, history=()):
        return self.p


def log2_likelihood(model, texts, count_eos=True, known=None):
    """(sum of log2 p, number of predicted events) over texts.

    known: if given (a set of signs), reproduce SRILM's default handling of out-of-vocabulary
    symbols: a sign outside the set is not scored at all, and it cuts the context of what follows.
    That drops the hardest events, so it flatters perplexity; it is here only to explain the gap
    to published numbers."""
    ll = n = 0
    for t in texts:
        s = [BOS, *t, EOS]
        for i in range(1, len(s) if count_eos else len(s) - 1):
            if known is not None and s[i] != EOS and s[i] not in known:
                continue
            h = s[max(0, i - model.order + 1):i]
            if known is not None:
                for j in range(len(h) - 1, -1, -1):
                    if h[j] != BOS and h[j] not in known:
                        h = h[j + 1:]
                        break
            ll += math.log2(model.prob(s[i], tuple(h)))
            n += 1
    return ll, n


def folds(texts, k=5, seed=0):
    """k (train, test) splits over texts. Pass de-duplicated texts, or copies leak across folds."""
    idx = list(range(len(texts)))
    random.Random(seed).shuffle(idx)
    for f in range(k):
        test = set(idx[f::k])
        yield [texts[i] for i in idx if i not in test], [texts[i] for i in sorted(test)]


def cross_validate(make_model, texts, vocab, k=5, seed=0):
    """Per-fold cross-entropy in bits per event, with and without the end-of-text prediction."""
    with_eos, without = [], []
    for train, test in folds(texts, k, seed):
        m = make_model(train, vocab)
        for out, flag in ((with_eos, True), (without, False)):
            ll, n = log2_likelihood(m, test, flag)
            out.append(-ll / n)
    return with_eos, without


def cross_validate_skipping_unseen(make_model, texts, k=5, seed=0):
    """Per-fold cross-entropy with the vocabulary taken from each training fold and unseen test
    signs skipped (see log2_likelihood). Returns (bits per fold, share of events skipped)."""
    bits, scored, total = [], 0, 0
    for train, test in folds(texts, k, seed):
        known = {s for t in train for s in t}
        ll, n = log2_likelihood(make_model(train, sorted(known)), test, known=known)
        bits.append(-ll / n)
        scored, total = scored + n, total + sum(len(t) + 1 for t in test)
    return bits, 1 - scored / total


def candidate_scores(model, vocab, texts, pairs):
    """Score of every candidate w at each (text index, position) gap: the product of the model's
    probabilities for every event whose history or identity contains the gap (for a bigram,
    P(w | prev) * P(next | w)). Normalising over vocab gives P(w | both sides). Products, not log sums,
    so rankings match the original loop exactly (ties included)."""
    out = np.ones((len(pairs), len(vocab)))
    for row, (k, j) in enumerate(pairs):
        s = [BOS, *texts[k], EOS]
        j += 1
        for e in range(j, min(j + model.order, len(s))):  # events whose history or identity contain position j
            lo = max(0, e - model.order + 1)
            for ci, w in enumerate(vocab):
                h = tuple(w if i == j else s[i] for i in range(lo, e))
                out[row, ci] *= model.prob(w if e == j else s[e], h)
    return out


def transition_matrix(model, vocab):
    """P[context, event]: contexts are <s> then the signs, events are the signs then </s>. Rows sum to 1."""
    return np.array([[model.prob(w, (h,)) for w in [*vocab, EOS]] for h in [BOS, *vocab]])


def selfcheck():
    train, vocab = ["abca", "abd", "cab", "e"], list("abcdefg")
    m = WittenBell(train, vocab, order=2)
    assert abs(m.prob("a") - 4.75 / 21) < 1e-12 and abs(m.prob("f") - 0.75 / 21) < 1e-12
    assert abs(m.prob("b", ("a",)) - 0.5595) < 1e-4 and abs(m.prob("c", ("a",)) - 0.0437) < 1e-4
    assert abs(m.prob("f", ("a",)) - 0.0119) < 1e-4 and m.prob("a", ("f",)) == m.prob("a")
    ll, n = log2_likelihood(m, ["abc"])
    assert abs(2 ** (-ll / n) - 4.04) < 0.01
    ll, n = log2_likelihood(m, ["abc"], count_eos=False)
    assert abs(2 ** (-ll / n) - 2.72) < 0.01
    for model in (m, WittenBell(train, vocab, 3), WittenBell(train, vocab, 2, backoff=True),
                  WittenBell(train, vocab, 3, backoff=True), AddK(train, vocab), KneserNey(train, vocab),
                  KneserNey(train, vocab, order=3), Uniform(train, vocab), Positional(train, vocab)):
        for h in [(BOS,), ("a",), ("f",), ("a", "b"), (BOS, "c")]:
            total = sum(model.prob(w, h) for w in [*vocab, EOS])
            assert abs(total - 1) < 1e-9, (type(model).__name__, h, total)
    # A maximum-likelihood bigram model with boundary tokens scores a corpus identically read forwards
    # or backwards, so comparing forward and backward perplexity cannot reveal writing direction.
    def mle_ll(texts):
        big, ctx = Counter(), Counter()
        for t in texts:
            s = [BOS, *t, EOS]
            big.update(zip(s, s[1:]))
            ctx.update(s[:-1])
        return sum(c * math.log2(c / ctx[a]) for (a, b), c in big.items())
    assert abs(mle_ll(train) - mle_ll([t[::-1] for t in train])) < 1e-9
    assert sum(len(te) for _, te in folds(list(range(23)), 5)) == 23
    # a short history must keep its start token at every order
    m4 = WittenBell(train, vocab, 4)
    assert m4.prob("b", (BOS, "a")) == m4._prob("b", (BOS, "a")) != m4._prob("b", ("a",))
    # backoff when every event was seen after a context still sums to 1
    full = WittenBell(["ab", "aa", "a"], list("ab"), 2, backoff=True)
    assert abs(sum(full.prob(w, ("a",)) for w in ["a", "b", EOS]) - 1) < 1e-12
    # skipping unseen signs: 'z' is not scored and cuts the context of the sign after it
    ll, n = log2_likelihood(m, ["azb"], known=set("abcde"))
    assert n == 3 and abs(ll - math.log2(m.prob("a", (BOS,)) * m.prob("b") * m.prob(EOS, ("b",)))) < 1e-12
    # count cutoff: the trigram (a, b) -> c is seen once and dropped; (a, b) -> EOS is seen twice and kept
    cut = WittenBell(["abc", "ab", "ab"], list("abc"), 3, min_count=2)
    assert cut.counts[("a", "b")] == Counter({EOS: 2}) and abs(sum(cut.prob(w, ("a", "b")) for w in [*"abc", EOS]) - 1) < 1e-12
    two = TwoEnded(train, vocab)
    for i, length in [(0, 1), (1, 3), (2, 3), (0, 9)]:
        assert abs(sum(two.prob(w, i, length) for w in vocab) - 1) < 1e-9


if __name__ == "__main__":
    selfcheck()
    print("ok")
