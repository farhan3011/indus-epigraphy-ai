"""Milestone 3 extension: restoring more than one sign at a time, damaged stretches of unknown length,
parallel texts, and flagging a sign that looks miscopied. All of it runs on the Kneser-Ney trigram, the best
single model of Milestone 3, so nothing here needs PyTorch except the comparison with the neural model.

    python sequence_analysis/restore.py                       # evaluation + report; about 10 minutes on a laptop CPU
    python sequence_analysis/restore.py --quick               # one fold, no neural comparison
    python sequence_analysis/restore.py fill "267 ? ? 342"    # reading order; ? = one missing sign, * = a damaged stretch of 0-3 signs
    python sequence_analysis/restore.py check "267 99 342"    # how expected each sign is, given the rest

Writes reports/milestone_3/ (restoration_extensions.json, restoration_extensions_report.md): aggregates only.
"""
import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

import ngram
from corpus import ROOT, ebuds, load_lines
from ngram import BOS, EOS, KneserNey

OUT = ROOT / "reports/milestone_3"
STAR = "*"      # a damaged stretch whose length is not known
MAX_RUN = 3     # ... of at most this many signs


class Scorer:
    """An n-gram model with its next-event distributions cached per history, so a search costs array arithmetic."""

    def __init__(self, model, vocab):
        self.model, self.vocab, self.n1 = model, list(vocab), model.order - 1
        self.idx = {s: i for i, s in enumerate(self.vocab)}
        self.cache = {}

    def history(self, seq):
        return tuple([BOS, *seq][-self.n1:])

    def logdist(self, h):
        """log P(event | history) for every vocabulary sign, then end-of-text."""
        if h not in self.cache:
            self.cache[h] = np.log([self.model.prob(w, h) for w in (*self.vocab, EOS)])
        return self.cache[h]

    def logp(self, w, h):
        col = len(self.vocab) if w == EOS else self.idx.get(w)
        return self.logdist(h)[col] if col is not None else math.log(self.model.prob(w, h))


def search(sc, text, beam=200, wide=4000):
    """Joint search over every gap (None) of a text given in reading order. Scores whole texts, end-of-text
    included, so fills of different lengths are comparable. Returns [(log-probability, filled text)], best first.
    A gap is expanded to all signs and the `wide` best partial texts are kept; pruning to `beam` waits for the next
    legible sign, so every candidate has met its right-hand neighbour before it can be dropped.
    ponytail: beam search, not exact; runs of 3+ adjacent gaps are pruned on left context only. Exact search over
    a trigram needs |signs|^2 states per adjacent gap pair."""
    V = len(sc.vocab)
    hyps = [(0.0, ())]
    for item in [*text, EOS]:
        if item is None:
            scores = np.array([lp for lp, _ in hyps])[:, None] + np.stack([sc.logdist(sc.history(seq))[:V] for _, seq in hyps])
            flat = scores.ravel()
            keep = min(wide, flat.size)
            top = np.argpartition(-flat, keep - 1)[:keep]
            hyps = [(float(flat[t]), hyps[t // V][1] + (sc.vocab[t % V],)) for t in top]
        else:
            hyps = [(lp + sc.logp(item, sc.history(seq)), seq if item == EOS else seq + (item,)) for lp, seq in hyps]
            hyps = sorted(hyps, key=lambda x: (-x[0], x[1]))[:beam]
    return hyps


def restore(sc, text, k=5, beam=200):
    """text: signs in reading order, None for one missing sign, STAR (at most one) for a damaged stretch of 0..MAX_RUN signs.
    Returns {"fills": [(filled text, probability, positions that were filled)], best first, at most k;
             "gaps": {gap position: [(sign, probability)]}, the per-gap marginals (only without a STAR);
             "lengths": {stretch length: probability} (only with a STAR)}.
    Probabilities are shares of the candidates the search kept, so they are slightly high."""
    text = list(text)
    assert text.count(STAR) <= 1, "one damaged stretch of unknown length at most"
    if STAR in text:
        i = text.index(STAR)
        variants = [(L, text[:i] + [None] * L + text[i + 1:]) for L in range(MAX_RUN + 1)]
    else:
        variants = [(None, text)]
    pool = [(lp, seq, L, tuple(j for j, s in enumerate(v) if s is None)) for L, v in variants for lp, seq in search(sc, v, beam)]
    lps = np.array([p[0] for p in pool])
    probs = np.exp(lps - lps.max())
    probs /= probs.sum()
    order = sorted(range(len(pool)), key=lambda j: (-probs[j], pool[j][1]))
    out = {"fills": [(pool[j][1], float(probs[j]), pool[j][3]) for j in order[:k]]}
    if STAR in text:
        out["lengths"] = {L: float(sum(p for p, row in zip(probs, pool) if row[2] == L)) for L, _ in variants}
    else:
        gaps = defaultdict(Counter)
        for p, (_, seq, _, filled) in zip(probs, pool):
            for g in filled:
                gaps[g][seq[g]] += p
        out["gaps"] = {g: [(s, float(p)) for s, p in sorted(c.items(), key=lambda x: (-x[1], x[0]))[:k]] for g, c in sorted(gaps.items())}
    return out


def parallels(texts, text, gap, k=5):
    """Corpus texts that match the signs around a gap. For every position j of every corpus text, count how many
    signs in a row agree immediately left of the gap and immediately right of it (the start and end of a text
    count as signs, other gaps never match). Returns (hits, fillers): hits = [(matched left, matched right, corpus
    text, j)] best first, one per corpus text; fillers = Counter of what stands at j when both sides match."""
    q = [BOS, *text, EOS]
    g = gap + 1
    hits, fillers = [], Counter()
    for t in texts:
        s = [BOS, *t, EOS]
        best = None
        for j in range(1, len(s) - 1):
            left = right = 0
            while g - 1 - left >= 0 and j - 1 - left >= 0 and q[g - 1 - left] is not None and q[g - 1 - left] == s[j - 1 - left]:
                left += 1
            while g + 1 + right < len(q) and j + 1 + right < len(s) and q[g + 1 + right] is not None and q[g + 1 + right] == s[j + 1 + right]:
                right += 1
            if left and right:
                fillers[s[j]] += 1
            if left + right and (best is None or (left + right, min(left, right)) > (best[0] + best[1], min(best[:2]))):
                best = (left, right, t, j - 1)
        if best:
            hits.append(best)
    hits.sort(key=lambda h: (-(h[0] + h[1]), -min(h[:2]), h[2]))
    return hits[:k], fillers


def surprise(model, vocab, text):
    """For each sign of a complete text: its probability given every other sign, its rank among all candidates,
    and the sign the model would have put there. A sign outside the vocabulary gets probability 0."""
    scores = ngram.candidate_scores(model, vocab, [tuple(text)], [(0, j) for j in range(len(text))])
    scores = scores / scores.sum(1, keepdims=True)
    idx = {s: i for i, s in enumerate(vocab)}
    out = []
    for j, s in enumerate(text):
        p = float(scores[j, idx[s]]) if s in idx else 0.0
        out.append({"sign": s, "p": p, "rank": int((scores[j] > p).sum()) + 1, "expected": vocab[int(np.argmax(scores[j]))],
                    "expected_p": float(scores[j].max())})
    return out


# ---------- evaluation ----------

def auc(pos, neg):
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    order = np.concatenate([pos, neg]).argsort(kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv[np.isin(allv, pos) & np.isin(allv, neg)]):  # average ranks over ties shared by both groups
        ranks[allv == v] = ranks[allv == v].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def two_gap_trials(test, rng):
    """One adjacent pair per text of 3+ signs and one separated pair per text of 4+ signs: (text index, i, j, kind)."""
    out = []
    for k, t in enumerate(test):
        if len(t) >= 3:
            i = rng.randrange(len(t) - 1)
            out.append((k, i, i + 1, "adjacent"))
        if len(t) >= 4:
            i = rng.randrange(len(t) - 2)
            out.append((k, i, rng.randrange(i + 2, len(t)), "separated"))
    return out


def neural_pair(model, tokens, temperature, text, i, j):
    """The Milestone 3 way: both gaps masked together, each filled on its own. Returns (top sign per gap, the 5 best
    pairs under the product of the two distributions)."""
    import masked_model
    got = masked_model.fill(model, tokens, temperature, [None if p in (i, j) else s for p, s in enumerate(text)], k=5)
    pairs = sorted(((pa * pb, (a, b)) for a, pa in got[i] for b, pb in got[j]), reverse=True)[:5]
    return (got[i][0][0], got[j][0][0]), [p for _, p in pairs]


def evaluate(texts, vocab, neural=True, n_folds=5, log=print):
    rng = random.Random(0)
    two = defaultdict(list)       # method -> per-trial dicts
    lengths, anomalies, groups = [], [], []
    for f, (train, test) in enumerate(ngram.folds(texts)):
        if f >= n_folds:
            break
        log(f"fold {f + 1}: fitting")
        model = KneserNey(train, vocab, order=3)
        sc = Scorer(model, vocab)
        net = None
        if neural:
            import masked_model
            cfg = json.loads((OUT / "results.json").read_text())["settings"]["final_config"]
            net = masked_model.train_final(train, vocab, cfg, seed=f)
        log(f"fold {f + 1}: two missing signs")
        for k, i, j, kind in two_gap_trials(test, rng):
            t = test[k]
            gapped = [None if p in (i, j) else s for p, s in enumerate(t)]
            truth = (t[i], t[j])
            groups.append((f, k))
            for name, kw in (("joint", {}), ("greedy", {"beam": 1, "wide": 1})):
                hyps = search(sc, gapped, **kw)[:5]
                pairs = [(seq[i], seq[j]) for _, seq in hyps]
                two[name].append({"kind": kind, "both": pairs[0] == truth, "signs": (pairs[0][0] == truth[0]) + (pairs[0][1] == truth[1]),
                                  "top5": truth in pairs})
            if net:
                top, pairs = neural_pair(*net, t, i, j)
                two["neural"].append({"kind": kind, "both": top == truth, "signs": (top[0] == truth[0]) + (top[1] == truth[1]), "top5": truth in pairs})
        log(f"fold {f + 1}: damaged stretch of unknown length")
        for t in test:
            if len(t) < 3:
                continue
            L = rng.randrange(min(MAX_RUN, len(t) - 2) + 1)
            i = rng.randrange(len(t) - L + 1)
            r = restore(sc, [*t[:i], STAR, *t[i + L:]], k=5)
            guess = max(r["lengths"], key=r["lengths"].get)
            lengths.append({"true": L, "guess": guess, "p_true": r["lengths"][L], "exact": r["fills"][0][0] == t,
                            "top5": any(fill == t for fill, _, _ in r["fills"])})
        log(f"fold {f + 1}: one miscopied sign")
        freq = Counter(s for t in train for s in t)
        signs, weights = zip(*sorted(freq.items()))
        for t in test:
            if len(t) < 3:
                continue
            i = rng.randrange(len(t))
            wrong = t[i]
            while wrong == t[i]:
                wrong = rng.choices(signs, weights)[0]
            bad = [*t[:i], wrong, *t[i + 1:]]
            clean, dirty = surprise(model, vocab, t), surprise(model, vocab, bad)
            found = min(range(len(bad)), key=lambda j: dirty[j]["p"])
            fix = ngram.candidate_scores(model, vocab, [tuple(bad)], [(0, i)])[0]
            top5 = [vocab[c] for c in np.argsort(-fix, kind="stable")[:5]]
            anomalies.append({"n": len(t), "found": found == i, "min_clean": min(c["p"] for c in clean), "min_dirty": min(d["p"] for d in dirty),
                              "repair1": top5[0] == t[i], "repair5": t[i] in top5})

    def rate(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else None

    res = {"two_gaps": {}, "settings": {"texts": len(texts), "folds": n_folds, "neural": neural, "beam": 200, "wide": 4000, "max_run": MAX_RUN}}
    for kind in ("adjacent", "separated", "all"):
        res["two_gaps"][kind] = {"trials": sum(r["kind"] == kind or kind == "all" for r in two["joint"])}
        for name, rows in two.items():
            sel = [r for r in rows if kind in ("all", r["kind"])]
            res["two_gaps"][kind][name] = {"both_right": rate(sel, "both"), "signs_right": rate(sel, "signs") / 2, "pair_in_top5": rate(sel, "top5")}
    if neural:
        import masked_model
        g = np.array([f * 100000 + k for f, k in groups])
        for other in ("neural", "greedy"):
            a, b = (np.array([r["both"] for r in two[n]], float) for n in ("joint", other))
            res["two_gaps"][f"joint_minus_{other}_both_right"] = masked_model.paired_bootstrap(a, b, g)
    by_len = {L: [r for r in lengths if r["true"] == L] for L in range(MAX_RUN + 1)}
    res["unknown_length"] = {
        "trials": len(lengths), "length_right": rate(lengths, "guess") if False else float(np.mean([r["true"] == r["guess"] for r in lengths])),
        "always_most_common_length": max(len(v) for v in by_len.values()) / len(lengths),
        "mean_probability_of_true_length": rate(lengths, "p_true"), "exact_fill": rate(lengths, "exact"), "fill_in_top5": rate(lengths, "top5"),
        "by_true_length": {str(L): {"trials": len(v), "length_right": float(np.mean([r["guess"] == L for r in v])), "exact_fill": rate(v, "exact"),
                                    "guessed": {str(G): sum(r["guess"] == G for r in v) for G in range(MAX_RUN + 1)}} for L, v in by_len.items() if v}}
    res["miscopy"] = {
        "trials": len(anomalies), "position_found": rate(anomalies, "found"), "position_by_chance": float(np.mean([1 / r["n"] for r in anomalies])),
        "text_auc": auc([-r["min_dirty"] for r in anomalies], [-r["min_clean"] for r in anomalies]),
        "repair_top1_given_position": rate(anomalies, "repair1"), "repair_top5_given_position": rate(anomalies, "repair5"),
        "found_and_repaired_top1": float(np.mean([r["found"] and r["repair1"] for r in anomalies]))}
    # a flag threshold with a 5% false-alarm rate on clean texts, and what it catches
    cut = float(np.quantile([r["min_clean"] for r in anomalies], 0.05))
    res["miscopy"] |= {"threshold_at_5pct_false_alarms": cut, "caught_at_that_threshold": float(np.mean([r["min_dirty"] <= cut for r in anomalies]))}
    return res


def reach(texts, vocab):
    """Descriptive, on the final neural model and the texts it was trained on: which context sign matters most for
    its top guess at each position (see masked_model.influence), by distance from the gap."""
    import masked_model
    model, tokens, temperature = masked_model.load()
    dist = Counter()
    for t in texts:
        if len(t) < 3:
            continue
        for g in range(len(t)):
            _, bits = masked_model.influence(model, tokens, temperature, [None if p == g else s for p, s in enumerate(t)], g)
            dist[min(abs(max(bits, key=bits.get) - g), 3)] += 1
    n = sum(dist.values())
    return {"positions": n, "most_influential_sign_is_adjacent": dist[1] / n, "two_away": dist[2] / n, "three_or_more_away": dist[3] / n}


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def write_report(res):
    tg, ul, mc, s = res["two_gaps"], res["unknown_length"], res["miscopy"], res["settings"]
    names = [("joint", "Kneser-Ney trigram, joint search"), ("greedy", "Kneser-Ney trigram, greedy left to right"),
             ("neural", "Neural model, each gap on its own (Milestone 3)")]
    L = ["# Milestone 3 extension: more than one missing sign, unknown-length damage, miscopies", "",
         "Milestone 3 filled one hidden sign and filled several gaps independently of each other. This extension adds a joint search "
         "over all gaps, damaged stretches whose length is not known, a check for a sign that does not fit its text, and retrieval of "
         f"parallel texts. Same {s['texts']:,} EBUDS texts and the same 5 folds as Milestones 2 and 3; every number below is on held-out texts. "
         "The joint search runs on the Kneser-Ney trigram, the best single model of Milestone 3.", "",
         "## Two missing signs", "",
         "Two signs are hidden in each held-out text: an adjacent pair (texts of 3+ signs) and a separated pair (4+ signs).", "",
         "| Method | Pair | Trials | Both signs right | Signs right | True pair in the 5 best |", "|---|---|---|---|---|---|"]
    for kind in ("adjacent", "separated", "all"):
        for key, label in names:
            if key in tg[kind]:
                r = tg[kind][key]
                L.append(f"| {label} | {kind} | {tg[kind]['trials']:,} | {pct(r['both_right'])} | {pct(r['signs_right'])} | {pct(r['pair_in_top5'])} |")
    for other, label in (("neural", "the neural model"), ("greedy", "greedy search")):
        d = tg.get(f"joint_minus_{other}_both_right")
        if d:
            L += ["", f"Joint search minus {label}, both signs right: {100 * d[0]:+.1f} points (95% interval {100 * d[1]:+.1f} to {100 * d[2]:+.1f}, "
                      "resampling whole test texts)."]
    L += ["", "The neural model was trained with one sign hidden at a time, so two hidden signs are outside what it was trained on; "
              "this is the comparison with how Milestone 3 handled several gaps, not a verdict on neural models."]
    L += ["", "## A damaged stretch of unknown length", "",
          f"A run of 0 to {s['max_run']} signs is removed from each held-out text of 3+ signs (0 means nothing is missing) and the model is told "
          "only where the damage is. It scores every fill of every length as a whole text, end included, so lengths compete directly.", "",
          f"- Length right: {pct(ul['length_right'])} of {ul['trials']:,} trials (always guessing the commonest length: {pct(ul['always_most_common_length'])}).",
          f"- Mean probability given to the true length: {pct(ul['mean_probability_of_true_length'])}.",
          f"- Whole stretch restored exactly: {pct(ul['exact_fill'])}; among the 5 best fills: {pct(ul['fill_in_top5'])}.", "",
          "The length guess is the length with the most total probability; the restored stretch is the single most probable fill. "
          "They can disagree: one complete text can be the likeliest single answer while most of the probability is spread over "
          "many longer fills.", "",
          "| True length | Trials | Length right | Stretch restored exactly | Guessed 0 / 1 / 2 / 3 |", "|---|---|---|---|---|"]
    for n, r in ul["by_true_length"].items():
        L.append(f"| {n} | {r['trials']:,} | {pct(r['length_right'])} | {pct(r['exact_fill'])} | {' / '.join(str(r['guessed'][str(g)]) for g in range(s['max_run'] + 1))} |")
    L += ["", "## One miscopied sign", "",
          "One sign of each held-out text of 3+ signs is replaced by a different sign drawn by corpus frequency. Each sign is then scored by "
          "its probability given all the others; the least expected sign is flagged.", "",
          f"- The flagged sign is the replaced one in {pct(mc['position_found'])} of {mc['trials']:,} texts (chance: {pct(mc['position_by_chance'])}).",
          f"- Telling a corrupted text from its clean original by its least expected sign: AUC {mc['text_auc']:.3f} (0.5 is chance).",
          f"- At a threshold that flags 5% of clean texts, {pct(mc['caught_at_that_threshold'])} of corrupted texts are flagged.",
          f"- Once the position is known, the original sign is the model's first suggestion in {pct(mc['repair_top1_given_position'])} and among "
          f"the first five in {pct(mc['repair_top5_given_position'])}. Found and repaired at the first guess: {pct(mc['found_and_repaired_top1'])}.", ""]
    if "reach" in res:
        r = res["reach"]
        L += ["## How far the neural model looks", "",
              f"For each of {r['positions']:,} positions, each context sign was replaced in turn by the unknown-sign token and the drop in the "
              f"probability of the model's top guess recorded. The most influential sign is next to the gap in {pct(r['most_influential_sign_is_adjacent'])} "
              f"of positions, two signs away in {pct(r['two_away'])} and further in {pct(r['three_or_more_away'])}. Descriptive only: this is the "
              "final model on texts it was trained on.", ""]
    L += ["## Reading the numbers", "",
          "- These are restoration scores on texts whose answer is known. They say how often the model agrees with the corpus, not that a "
          "restored sign on a really damaged seal is right.",
          "- The corruption test replaces a sign with a frequency-drawn one. A real copyist's slip is more likely a similar-looking sign, "
          "which would be harder to catch; the detection rate is an upper bound for that case.",
          "- A rare but genuine sign looks the same to the model as a miscopy. The flag is a prompt to look at the object again, not a correction.",
          "- Probabilities from the joint search are shares of the candidates the search kept, so they run slightly high.", "",
          "## Use it", "", "```bash",
          'python sequence_analysis/restore.py fill "267 ? ? 342"   # reading order; ? = one missing sign',
          'python sequence_analysis/restore.py fill "267 * 342"     # * = a damaged stretch of 0-3 signs',
          'python sequence_analysis/restore.py check "267 99 342"   # how expected each sign is', "```", "",
          f"Settings: beam {s['beam']}, {s['wide']:,} partial texts kept after a gap, Kneser-Ney trigram (discount 0.75). "
          f"Run time {s.get('runtime_min', 0):.0f} minutes on a laptop CPU.", ""]
    (OUT / "restoration_extensions_report.md").write_text("\n".join(L))


def selfcheck():
    texts = [(1, 2, 3, 4)] * 6 + [(1, 5, 3, 4)] * 3 + [(2, 3, 4)] * 2 + [(1, 2, 4)]
    vocab = [1, 2, 3, 4, 5]
    model = KneserNey(texts, vocab, order=3)
    sc = Scorer(model, vocab)
    # the search scores a complete text exactly like the model does
    want = sum(math.log(model.prob(w, h)) for w, h in ((1, (BOS,)), (2, (BOS, 1)), (3, (1, 2)), (4, (2, 3)), (EOS, (3, 4))))
    assert abs(search(sc, [1, 2, 3, 4])[0][0] - want) < 1e-9
    # one gap: the joint search ranks and weighs candidates like Milestone 2's two-sided scorer
    one = restore(sc, [1, None, 3, 4])
    ref = ngram.candidate_scores(model, vocab, [(1, 2, 3, 4)], [(0, 1)])[0]
    ref = ref / ref.sum()
    assert [s for s, _ in one["gaps"][1]] == [vocab[i] for i in np.argsort(-ref, kind="stable")]
    assert all(abs(p - ref[vocab.index(s)]) < 1e-9 for s, p in one["gaps"][1])
    # two gaps: the commonest text wins, and a greedy search is never better than the joint one
    two = restore(sc, [1, None, None, 4])
    assert two["fills"][0][0] == (1, 2, 3, 4) and two["fills"][0][2] == (1, 2) and abs(sum(p for _, p in two["gaps"][1]) - 1) < 1e-9
    assert search(sc, [None, None, 4])[0][0] >= search(sc, [None, None, 4], beam=1, wide=1)[0][0]
    # unknown length: (1 2 * 4) is most often (1 2 3 4); probabilities over lengths sum to one
    star = restore(sc, [1, 2, STAR, 4])
    assert max(star["lengths"], key=star["lengths"].get) == 1 and abs(sum(star["lengths"].values()) - 1) < 1e-9
    assert star["fills"][0][0] == (1, 2, 3, 4) and star["fills"][0][2] == (2,)
    # parallels: both neighbours of the gap in (1 ? 3 4) match (1 2 3 4) with the start and end counted
    hits, fillers = parallels(sorted(set(texts)), [1, None, 3, 4], 1)
    assert hits[0][:2] == (2, 3) and hits[0][2] in ((1, 2, 3, 4), (1, 5, 3, 4)) and fillers == Counter({2: 1, 5: 1})
    # a sign that never follows its neighbours is the least expected one
    sur = surprise(model, vocab, (1, 2, 5, 4))
    assert min(range(4), key=lambda j: sur[j]["p"]) == 2 and sur[2]["expected"] == 3 and sur[0]["rank"] == 1
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1, 2], [1, 2]) == 0.5


def parse(s, vocab):
    out = []
    for tok in s.split():
        if tok == "?":
            out.append(None)
        elif tok == STAR:
            out.append(STAR)
        elif tok.isdigit() and int(tok) in vocab:
            out.append(int(tok))
        else:
            sys.exit(f"not a sign number of the corpus: {tok}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="evaluate", choices=["evaluate", "fill", "check"])
    ap.add_argument("text", nargs="?", help="signs in reading order; for fill, ? = one missing sign and * = a damaged stretch")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    texts, _ = ebuds(load_lines())
    vocab = sorted({s for t in texts for s in t})
    if args.command != "evaluate":
        model = KneserNey(texts, vocab, order=3)
        text = parse(args.text, set(vocab))
        if args.command == "fill":
            r = restore(Scorer(model, vocab), text)
            for fill, p, filled in r["fills"]:
                print(f"{p:6.1%}  " + " ".join(f"[{s}]" if j in filled else str(s) for j, s in enumerate(fill)))
            if "lengths" in r:
                print("signs missing: " + ", ".join(f"{n}: {p:.0%}" for n, p in r["lengths"].items()))
        else:
            for row in surprise(model, vocab, text):
                print(f"{row['sign']:>4}  probability {row['p']:6.1%}  rank {row['rank']:>3}  expected {row['expected']} ({row['expected_p']:.0%})")
        sys.exit()
    res = evaluate(texts, vocab, neural=not args.quick, n_folds=1 if args.quick else 5,
                   log=lambda m: print(f"[{(time.time() - start) / 60:.1f} min] {m}", flush=True))
    if not args.quick:
        res["reach"] = reach(texts, vocab)
    res["settings"] |= {"runtime_min": (time.time() - start) / 60, "quick": args.quick}
    if args.quick:
        OUT = ROOT / "reports/quick"  # gitignored
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "restoration_extensions.json").write_text(json.dumps(res, indent=1))
    write_report(res)
    print(json.dumps({k: v for k, v in res.items() if k != "settings"}, indent=1))
    print(f"report -> {OUT / 'restoration_extensions_report.md'}  ({(time.time() - start) / 60:.1f} min)")
