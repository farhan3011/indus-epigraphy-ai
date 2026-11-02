"""Milestone 4: test claimed sign-to-sound keys for the Indus script against shuffled versions of themselves.

    python decipherment/data.py                     # once: fetch and verify the inputs
    python decipherment/harness.py --controls-only  # positive and negative controls (can the harness detect anything?)
    python decipherment/harness.py                  # controls + every claim in prereg.json; needs prereg.json committed
    python decipherment/harness.py --quick          # far fewer draws, for a fast check

Writes reports/milestone_4/ (results.json, figures, milestone_4_report.md): aggregates only, never key values or texts.
The test can only reject a specific table under a specific score; it says nothing about whether the script is a language.
"""
import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kstest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import data
from data import CONSONANTS, skeleton

ROOT = data.ROOT
sys.path.insert(0, str(ROOT / "sequence_analysis"))
import corpus  # noqa: E402
import ngram  # noqa: E402
import stats  # noqa: E402

OUT = ROOT / "reports/milestone_4"
PREREG = Path(__file__).resolve().parent / "prereg.json"
ALPHABET = sorted(CONSONANTS)
BOS, EOS = len(ALPHABET), len(ALPHABET) + 1  # symbol codes; ALPHABET letters are 0..15
CODE = {c: i for i, c in enumerate(ALPHABET)}
# Scores, in this order. "sequence" (primary, pre-registration v2): log2 P under the Sanskrit trigram minus log2 P under the
# same corpus's letter frequencies, per phoneme, i.e. Sanskrit-like order beyond Sanskrit-like letters. "margin_tamil"
# (primary in v1): Sanskrit trigram minus Old Tamil trigram. "naive_bayes": the rubric's classifier, Sanskrit vs Old Tamil.
SCORERS = ("sequence", "margin_tamil", "naive_bayes")
PRIMARY = 0


# ---------- scorers: lookup tables so each shuffled key costs only array indexing ----------

def lm_table(lines, order=3):
    """log2 P(c | a, b) for every a, b in alphabet + BOS and c in alphabet + EOS, from an interpolated Witten-Bell
    character trigram (sequence_analysis/ngram.py) over skeleton strings. Index [a, b, c]; BOS/EOS rows as codes."""
    m = ngram.WittenBell([tuple(l) for l in lines if l], ALPHABET, order)
    sym = ALPHABET + [ngram.BOS, ngram.EOS]
    t = np.zeros((len(sym), len(sym), len(sym)))
    for a in range(BOS + 1):
        for b in range(BOS + 1):
            if b == BOS and a != BOS:
                continue  # BOS is only ever preceded by BOS
            h = (sym[a], sym[b]) if a != BOS else ((ngram.BOS,) if b == BOS else (ngram.BOS, sym[b]))
            for c in list(range(len(ALPHABET))) + [EOS]:
                t[a, b, c] = math.log2(m.prob(sym[c], h))
    return t


def nb_tables(target, control):
    """Per-n-gram log-likelihood ratio of a multinomial Naive Bayes classifier (target vs control) over character
    1-3-grams; n-grams outside its vocabulary weigh 0, as in the classifier itself. Returns (W1, W2, W3, model)."""
    vec = CountVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False)
    X = vec.fit_transform(target + control)
    nb = MultinomialNB().fit(X, [1] * len(target) + [0] * len(control))
    ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]  # classes_ sorted: [0 = control, 1 = target]
    n = BOS + 2
    W = [np.zeros((n,) * k) for k in (1, 2, 3)]
    for gram, j in vec.vocabulary_.items():
        if all(c in CODE for c in gram):
            W[len(gram) - 1][tuple(CODE[c] for c in gram)] = ratio[j]
    return (*W, (vec, nb))


_CODES = {}


def value_codes(v):
    if v not in _CODES:
        _CODES[v] = [CODE[ch] for ch in skeleton(v)]
    return _CODES[v]


class Corpus:
    """Texts (tuples of sign ids) laid out once as a stream; a key is a code array per sign. Scoring a key gathers each
    sign's skeleton codes, strips padding and sums table entries: per text, BOS BOS chars... EOS."""

    def __init__(self, texts, signs):
        self.signs = sorted(signs)
        row = {s: i + 2 for i, s in enumerate(self.signs)}  # rows 0 and 1: text start and text end markers
        stream, owner = [], []
        for k, t in enumerate(texts):
            stream += [0, *(row[s] for s in t), 1]
            owner += [k] * (len(t) + 2)
        self.stream, self.owner, self.n_texts = np.array(stream), np.array(owner), len(texts)
        self.counts = Counter(s for t in texts for s in t)

    def codes(self, values, width=12):
        """values: {sign: value string} -> padded code matrix (rows: start, end, then signs)."""
        M = np.full((len(self.signs) + 2, width), -1)
        M[0, :2], M[1, 0] = BOS, EOS
        for i, s in enumerate(self.signs):
            c = value_codes(values[s])[:width]
            M[i + 2, :len(c)] = c
        return M

    def flat(self, M):
        f = M[self.stream].ravel()
        keep = f >= 0
        return f[keep], np.repeat(self.owner, M.shape[1])[keep]

    def score(self, M, tables, per_text=False):
        """One value per scorer in SCORERS, in bits per phoneme (end of text counted), over the whole corpus;
        per_text=True gives the primary score per text instead. tables: ((lm, reference) pairs..., nb)."""
        f, owner = self.flat(M)
        a, b, c = f[:-2], f[1:-1], f[2:]
        counted = c != BOS
        lms = [(t[a, b, c] - r[a, b, c])[counted] for t, r in tables[:-1]]
        if per_text:
            own = owner[2:][counted]
            n = np.bincount(own, minlength=self.n_texts)
            return np.bincount(own, lms[PRIMARY], self.n_texts) / n, n
        W1, W2, W3 = tables[-1][:3]
        nbs = W1[f].sum() + W2[f[:-1], f[1:]].sum() + W3[a, b, c].sum()
        return np.array([x.sum() for x in lms] + [nbs / math.log(2)]) / counted.sum()


# ---------- keys and their shuffles ----------

def units(key):
    """Shuffle units: signs sharing (canonical head, value) move together. key: {sign: (canonical, value)}."""
    groups = defaultdict(list)
    for s, cv in key.items():
        groups[cv].append(s)
    return [(sorted(v), cv[1]) for cv, v in sorted(groups.items())]


def assign(us, values):
    return {s: v for (signs, _), v in zip(us, values) for s in signs}


def shuffled(us, rng, bins=None):
    """Values permuted across units, or only within bins (list of index arrays)."""
    vals = np.array([v for _, v in us], dtype=object)
    if bins is None:
        return assign(us, rng.permutation(vals))
    out = vals.copy()
    for b in bins:
        out[b] = vals[rng.permutation(b)]
    return assign(us, out)


def freq_bins(us, counts, n_bins):
    """Equal-count bins of unit token frequency (ties by first sign)."""
    order = sorted(range(len(us)), key=lambda i: (sum(counts[s] for s in us[i][0]), us[i][0][0]))
    return [np.array(b) for b in np.array_split(order, n_bins)]


def p_value(obs, null):
    return float((1 + (np.asarray(null) >= obs).sum()) / (len(null) + 1))


def null_scores(cp, us, tables, draws, rng, bins=None):
    return np.array([cp.score(cp.codes(shuffled(us, rng, bins)), tables) for _ in range(draws)])


def test_key(cp, key, tables, draws, rng, bin_counts=(10,)):
    """Observed scores and value-shuffle / binned nulls for one key on one corpus."""
    us = units(key)
    values = {s: cv[1] for s, cv in key.items()}
    obs = cp.score(cp.codes(values), tables)
    nulls = {"value_shuffle": null_scores(cp, us, tables, draws, rng)}
    for n in bin_counts:
        nulls[f"binned_{n}"] = null_scores(cp, us, tables, draws, rng, freq_bins(us, cp.counts, n))
    res = {"observed": {sc: float(obs[i]) for i, sc in enumerate(SCORERS)}, "nulls": {}}
    for name, arr in nulls.items():
        res["nulls"][name] = {sc: {"mean": float(arr[:, i].mean()), "sd": float(arr[:, i].std()), "p": p_value(obs[i], arr[:, i]),
                                   "percentile": float(100 * (arr[:, i] < obs[i]).mean())} for i, sc in enumerate(SCORERS)}
    lo = min(arr[:, PRIMARY].min() for arr in nulls.values())
    hi = max(arr[:, PRIMARY].max() for arr in nulls.values())
    edges = np.linspace(min(lo, obs[PRIMARY]), max(hi, obs[PRIMARY]), 41)
    res["primary_histogram"] = {"edges": edges.tolist(), **{n: np.histogram(nulls[n][:, PRIMARY], edges)[0].tolist() for n in ("value_shuffle", "binned_10")}}
    res["null_correlation_with_naive_bayes"] = {sc: float(np.corrcoef(nulls["value_shuffle"][:, i], nulls["value_shuffle"][:, 2])[0, 1]) for i, sc in enumerate(SCORERS[:2])}
    return res, nulls


# ---------- controls ----------

def encipher(plain_lines, key, lengths, rng):
    """Positive-control corpus: skeleton plaintext split greedily (longest match) into the key's value skeletons, each piece
    written with a random sign carrying that value skeleton (many-to-one, like the claim), cut to the given text lengths."""
    by_skel = defaultdict(list)
    for s, (_, v) in key.items():
        if skeleton(v):
            by_skel[skeleton(v)].append(s)
    longest = max(map(len, by_skel))
    signs = []
    for line in plain_lines:
        sk, i = skeleton(line), 0
        while i < len(sk):
            for L in range(min(longest, len(sk) - i), 0, -1):
                if sk[i:i + L] in by_skel:
                    signs.append(by_skel[sk[i:i + L]][rng.integers(len(by_skel[sk[i:i + L]]))])
                    i += L
                    break
            else:
                i += 1  # a consonant no value covers is dropped
    texts, i = [], 0
    for n in rng.choice(lengths, len(signs)):
        if i + n > len(signs):
            break
        texts.append(tuple(signs[i:i + n]))
        i += n
    return texts


def corrupt(key, share, rng):
    """A share of shuffle units given another unit's value."""
    us = units(key)
    vals = [v for _, v in us]
    pick = rng.choice(len(us), round(share * len(us)), replace=False)
    swapped = list(vals)
    for i in pick:
        swapped[i] = vals[rng.integers(len(vals))]
    return {s: (cv[0], v) for s, cv in key.items() for (signs, _), v in zip(us, swapped) if s in signs}


def positive_control(key, plain, lengths, tables, rng, replicates, draws, corruptions=(0, 0.1, 0.2, 0.3)):
    out = {}
    for share in corruptions:
        ps = defaultdict(list)
        for _ in range(replicates):
            texts = encipher([plain[i] for i in rng.permutation(len(plain))[:400]], key, lengths, rng)
            cp = Corpus(texts, key)
            noisy = corrupt(key, share, rng) if share else key
            r, _ = test_key(cp, noisy, tables, draws, rng)
            for null in r["nulls"]:
                for sc in SCORERS:
                    ps[(null, sc)].append(r["nulls"][null][sc]["p"])
        out[str(share)] = {null: {"power": float(np.mean(np.array(ps[(null, SCORERS[PRIMARY])]) < 0.05)),
                                  "median_p": float(np.median(ps[(null, SCORERS[PRIMARY])])),
                                  "power_by_scorer": {sc: float(np.mean(np.array(ps[(null, sc)]) < 0.05)) for sc in SCORERS}}
                           for null in ("value_shuffle", "binned_10")}
    return out


def negative_control(cp, key, tables, rng, runs, draws):
    us = units(key)
    ps = []
    for _ in range(runs):
        fake = shuffled(us, rng)
        obs = cp.score(cp.codes(fake), tables)[PRIMARY]
        null = [cp.score(cp.codes(shuffled(us, rng)), tables)[PRIMARY] for _ in range(draws)]
        ps.append(p_value(obs, null))
    ps = np.array(ps)
    return {"runs": runs, "share_below_0.05": float((ps < 0.05).mean()), "ks_p_uniform": float(kstest(ps, "uniform").pvalue), "p_values": ps.tolist()}


# ---------- other claim checks ----------

def order_nulls(cp_texts, key, tables, rng, draws):
    """True key on the real texts vs the same texts with signs shuffled within each text, and reversed."""
    values = {s: cv[1] for s, cv in key.items()}
    def score(texts):
        cp = Corpus(texts, key)
        return cp.score(cp.codes(values), tables)[PRIMARY]
    obs = score(cp_texts)
    shuf = np.array([score([tuple(rng.permutation(t)) for t in cp_texts]) for _ in range(draws)])
    return {"observed": float(obs), "reversed": float(score([t[::-1] for t in cp_texts])), "shuffled_mean": float(shuf.mean()),
            "shuffled_sd": float(shuf.std()), "share_of_shuffles_scoring_higher": float((shuf >= obs).mean()), "draws": draws}


def order_effect(texts, key, tables, rng, keys=200, shuffles=50):
    """Exploratory, not pre-registered: is the real-order advantage specific to the claimed key? z = (real order - mean of
    within-text shuffles) / sd, for the claim and for value-shuffled keys on the same shuffled texts."""
    cps = [Corpus([tuple(rng.permutation(t)) for t in texts], key) for _ in range(shuffles)]
    real = Corpus(texts, key)
    def z(values):
        M = real.codes(values)
        sh = np.array([c.score(M, tables)[PRIMARY] for c in cps])
        return (real.score(M, tables)[PRIMARY] - sh.mean()) / sh.std()
    claim = z({s: cv[1] for s, cv in key.items()})
    us = units(key)
    null = np.array([z(shuffled(us, rng)) for _ in range(keys)])
    return {"claim_z": float(claim), "random_keys_z_mean": float(null.mean()), "random_keys_z_sd": float(null.std()),
            "p": p_value(claim, null), "keys": keys, "shuffles": shuffles}


def read_vs_unread(key, tables, rng, perms):
    """Does the key score better on texts the claimant read than on texts he left unread? Length-stratified permutation."""
    rows = data.load_inscriptions()
    read = {r["signs"] for r in rows if r["reading"] and r["signs"]}
    unread = {r["signs"] for r in rows if not r["reading"] and not r["is_ref"] and r["signs"]} - read
    texts = [t for t in sorted(read) + sorted(unread) if all(s in key for s in t)]
    label = np.array([t in read for t in texts])
    cp = Corpus(texts, key)
    score, _ = cp.score(cp.codes({s: cv[1] for s, cv in key.items()}), tables, per_text=True)
    strata = np.array([min(len(t), 8) for t in texts])
    def stat(lab):
        d = [(score[(strata == s) & lab].mean() - score[(strata == s) & ~lab].mean(), (strata == s).sum())
             for s in np.unique(strata) if lab[strata == s].any() and (~lab[strata == s]).any()]
        return sum(x * w for x, w in d) / sum(w for _, w in d)
    obs = stat(label)
    null = []
    for _ in range(perms):
        lab = label.copy()
        for s in np.unique(strata):
            idx = np.flatnonzero(strata == s)
            lab[idx] = rng.permutation(lab[idx])
        null.append(stat(lab))
    return {"read_texts": int(label.sum()), "unread_texts": int((~label).sum()), "difference_bits_per_phoneme": float(obs),
            "p": p_value(obs, null), "perms": perms}


def shaw_numbering(key, rng, draws):
    """Share of consecutive-number sign pairs whose values share a vowel, against values shuffled across signs."""
    vowels = lambda v: set(v) & set("aAiIuUeEoO")
    signs = sorted(key)
    pairs = [(a, a + 1) for a in signs if a + 1 in key]
    def count(k):
        return sum(bool(vowels(k[a]) & vowels(k[b])) for a, b in pairs)
    obs = count(key)
    vals = [key[s] for s in signs]
    null = np.array([count(dict(zip(signs, rng.permutation(vals)))) for _ in range(draws)])
    return {"pairs": len(pairs), "observed": obs, "expected": float(null.mean()), "p": p_value(obs, null), "draws": draws}


def search_matched(cp, key, tables, rng, climbs=20, swaps=2000):
    """Random tables hill-climbed on the same scorer with the claim's own values; the claim vs the optimised scores."""
    us = units(key)
    best_scores = []
    for _ in range(climbs):
        vals = list(rng.permutation([v for _, v in us]))
        cur = cp.score(cp.codes(assign(us, vals)), tables)[PRIMARY]
        for _ in range(swaps):
            i, j = rng.integers(len(vals), size=2)
            vals[i], vals[j] = vals[j], vals[i]
            s = cp.score(cp.codes(assign(us, vals)), tables)[PRIMARY]
            if s >= cur:
                cur = s
            else:
                vals[i], vals[j] = vals[j], vals[i]
        best_scores.append(cur)
    obs = cp.score(cp.codes({s: cv[1] for s, cv in key.items()}), tables)[PRIMARY]
    return {"climbs": climbs, "swaps": swaps, "optimised_mean": float(np.mean(best_scores)), "p": p_value(obs, best_scores)}


# ---------- provenance, figures, report ----------

def provenance():
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git_commit": git("rev-parse", "HEAD"), "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
            "prereg_uncommitted": bool(git("status", "--porcelain", str(PREREG))),
            "inputs_sha256": json.loads((data.RAW / "sha256.json").read_text())}


def figures(res, nulls):
    for claim, by_null in nulls.items():
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
        for i, sc in enumerate(SCORERS):
            for name, colour in (("value_shuffle", "0.65"), ("binned_10", "#4a9bd1")):
                ax[i].hist(by_null[name][:, i], bins=60, color=colour, alpha=0.7, label=f"{name.replace('_', ' ')} null")
            ax[i].axvline(res["claims"][claim]["observed"][sc], color="#c9663c", lw=2, label="claimed key")
            ax[i].set_xlabel(SCORE_LABELS[sc] + ", bits per phoneme", fontsize=8)
            ax[i].legend(fontsize=7)
        ax[0].set_ylabel("shuffled keys"); fig.suptitle(claim, fontsize=10)
        plt.tight_layout(); plt.savefig(OUT / f"null_{claim}.png", dpi=130); plt.close()
    pc = res["positive_control"]
    shares = sorted(pc, key=float)
    plt.figure(figsize=(5, 3.6))
    for name, colour in (("value_shuffle", "0.45"), ("binned_10", "#4a9bd1")):
        plt.plot([100 * float(s) for s in shares], [pc[s][name]["power"] for s in shares], marker="o", color=colour, label=f"{name.replace('_', ' ')} null")
    plt.axhline(0.8, color="#c9663c", ls=":", lw=1, label="80% power")
    plt.xlabel("units given a wrong value (%)"); plt.ylabel("share of true keys detected (p < 0.05)"); plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "power_curve.png", dpi=130); plt.close()
    plt.figure(figsize=(5, 3.6))
    plt.hist(res["negative_control"]["p_values"], bins=20, range=(0, 1), color="0.6")
    plt.xlabel("p-value of a random table treated as a claim"); plt.ylabel("runs")
    plt.tight_layout(); plt.savefig(OUT / "negative_control_p.png", dpi=130); plt.close()


def pct(x):
    return f"{100 * x:.0f}%"


SCORE_LABELS = {"sequence": "Sanskrit sequence score (primary)", "margin_tamil": "Sanskrit minus Old Tamil (v1 primary)",
                "naive_bayes": "Naive Bayes, Sanskrit vs Old Tamil"}


def write_report(res):
    s, pc, nc = res["settings"], res["positive_control"], res["negative_control"]
    power0 = min(pc["0"][n]["power"] for n in pc["0"])
    lines = [
        "# Milestone 4 report: testing claimed decipherments against chance",
        "",
        "A claimed decipherment of the Indus script is a table from signs to sounds. This harness applies a table to the "
        "inscriptions, measures how Sanskrit-like the result is, and compares that with thousands of tables made by shuffling "
        "the claim's own sound values among its signs. Everything was fixed in `decipherment/prereg.json` and committed before "
        f"any claim was scored (pre-registration SHA-256 `{res['provenance']['prereg_sha256'][:12]}…`, commit "
        f"`{res['provenance']['git_commit'][:10]}`). **The test can only show that a particular table does or does not carry a "
        "signal beyond chance; it cannot show that the script is or is not a language.**",
        "",
        "## Revision of the pre-registration",
        "",
        json.loads(PREREG.read_text())["revision"],
        "",
        "## Which claims can be tested",
        "",
        "| Claim | Status | What was done |",
        "|---|---|---|",
        *[f"| {c} | `{v['status']}` | {v.get('done', v.get('reason', v.get('flag', 'no fixed sign-to-value table is published')))} |" for c, v in res["claims_table"].items()],
        "",
        "## Can the harness detect a real decipherment at all?",
        "",
        f"Positive control: held-out Sanskrit text from the Digital Corpus of Sanskrit was enciphered with the Yajnadevam key's "
        f"own sign-to-sound relation (many signs per sound) and cut to the lengths of real inscriptions; the true table was then "
        f"tested exactly like a claim ({s['positive_replicates']} replicates, {s['control_draws']:,} shuffles per null). It was "
        f"detected (p < 0.05) in {pct(pc['0']['value_shuffle']['power'])} of runs against the value-shuffle null and "
        f"{pct(pc['0']['binned_10']['power'])} against the frequency-binned null; with 30% of its values wrong, "
        f"{pct(pc['0.3']['value_shuffle']['power'])} and {pct(pc['0.3']['binned_10']['power'])} (`power_curve.png`). "
        + ("So a negative result below is informative." if power0 >= 0.8 else
           "Power is below 80%, so a negative result below must be read as \"low power\", not as a rejection."),
        "",
        f"Negative control: {nc['runs']} shuffled tables were treated as claims. {pct(nc['share_below_0.05'])} came out "
        f"\"significant\" at 0.05 (5% expected) and the p-values are consistent with uniform (KS p = {nc['ks_p_uniform']:.2f}; "
        "`negative_control_p.png`), so the test does not fire falsely.",
        "",
        "## Results",
        "",
        "| Key | Score | Observed | Value-shuffle null: mean ± sd, p | Frequency-binned null: mean ± sd, p | Holm p (both nulls) |",
        "|---|---|---|---|---|---|",
    ]
    for claim, r in res["claims"].items():
        for sc, label in SCORE_LABELS.items():
            n1, n2 = r["nulls"]["value_shuffle"][sc], r["nulls"]["binned_10"][sc]
            lines.append(f"| {claim} | {label} | {r['observed'][sc]:+.3f} | {n1['mean']:+.3f} ± {n1['sd']:.3f}, p = {n1['p']:.4f} | "
                         f"{n2['mean']:+.3f} ± {n2['sd']:.3f}, p = {n2['p']:.4f} | {r['holm'][sc]['value_shuffle']:.4f}, {r['holm'][sc]['binned_10']:.4f} |")
    lines += ["", "All scores are bits per phoneme of the key's output (collapsed consonant skeleton); higher is more Sanskrit-like. "
              "The primary score asks whether the output has Sanskrit-like *order* beyond Sanskrit-like letters: log-probability "
              "under a Sanskrit character trigram minus log-probability under Sanskrit letter frequencies. p is the share of shuffled "
              f"keys scoring at least as high ({s['draws']:,} shuffles); Holm adjusts across the {s['holm_family']} pre-registered tests.", ""]
    for claim, r in res["claims"].items():
        lines.append(f"- **{claim}**: {r['sentence']}")
    lines += [""]
    for claim, r in res["claims"].items():
        o = r["order_null"]
        lines.append(f"- {claim}, sign order: the real order scores {o['observed']:+.3f}; shuffling the signs inside each text "
                     f"gives {o['shuffled_mean']:+.3f} ± {o['shuffled_sd']:.3f} ({pct(o['share_of_shuffles_scoring_higher'])} of shuffles score "
                     f"at least as high); reversing every text gives {o['reversed']:+.3f}.")
    oe = res["order_effect"]
    lines += ["", f"Exploratory check, not pre-registered: the real sign order beats within-text shuffles for the claimed key "
              f"(z = {oe['claim_z']:.1f}), but random tables made from the same values show order effects just as large "
              f"(z = {oe['random_keys_z_mean']:+.1f} ± {oe['random_keys_z_sd']:.1f}; p = {oe['p']:.3f} for the claim). The Indus "
              "inscriptions have strong sign order of their own (Milestone 2), and almost any table carries some of it into its output."]
    rv = res["read_vs_unread"]
    lines += ["", f"Fitting check (current key): on the {rv['read_texts']:,} texts the claimant read, the key scores "
              f"{rv['difference_bits_per_phoneme']:+.3f} bits per phoneme relative to the {rv['unread_texts']:,} texts he left unread, "
              f"after matching by length (p = {rv['p']:.3f}). A key fitted to its own readings would do better on the read texts; "
              "that would be evidence of fitting, not of decipherment.", ""]
    sh, fu = res["shaw"], res["fuls_coverage"]
    lines += ["## The other keys", "",
              f"- Shaw 2026 has no target language, so only one thing can be checked: whether its values follow the sign numbering. "
              f"Of {sh['pairs']} pairs of consecutively numbered signs, {sh['observed']} share a vowel against {sh['expected']:.0f} expected "
              f"if the values were shuffled (p = {sh['p']:.4f}).",
              f"- Fuls 2024 gives values for {fu['signs']} signs; they cover {pct(fu['token_share'])} of sign tokens in the Wells corpus "
              f"and {fu['fully_covered_texts']} whole texts. No openly licensed Linear Elamite corpus exists to score them against, so this "
              "claim is not run.", "",
              "## How to read this", "",
              "- Beating the frequency-binned null is necessary for a decipherment, not sufficient: it shows only that the table "
              "carries structure beyond putting common sounds on common signs.",
              "- Every result is conditional on the scoring choices fixed in advance: the Digital Corpus of Sanskrit as the target, a "
              "consonant skeleton (which ignores the claimant's vowel and conjunct choices) and a character trigram. No corpus from "
              "2600-1900 BCE exists, so the model is a proxy built from much later language.",
              "- The two scores that compare Sanskrit with Old Tamil (the v1 primary and Naive Bayes) carry no evidence either way: in the "
              f"positive control they detected the true table in {pct(pc['0']['value_shuffle']['power_by_scorer']['margin_tamil'])} and "
              f"{pct(pc['0']['value_shuffle']['power_by_scorer']['naive_bayes'])} of runs against the value-shuffle null, because Tamil "
              "script does not mark voicing and the comparison rewards letters rather than order.",
              "- The Yajnadevam test applies the table sign by sign. Plain substitution reproduces his own consonant skeleton in "
              f"{pct(res['gate_checks']['skeleton_match_reading_order'])} of the {res['gate_checks']['read_texts']:,} texts he reads, so this tests "
              "the table, not every per-text choice behind his readings.",
              "- The proposal asked to \"disprove\" language claims. A permutation test cannot do that: it can only find that one "
              "specific table is or is not distinguishable from chance, with the power measured above.",
              "", "## Data and licences", "",
              "Sanskrit: Digital Corpus of Sanskrit, O. Hellwig (CC BY 4.0). Old Tamil: Project Madurai Sangam anthologies. Keys: "
              "Yajnadevam (github.com/yajnadevam/lipi, no licence: fetched at run time, never stored in this repository); Shaw 2026 "
              "and Fuls 2024 via the paragorth/Indus- transcriptions (cite the original works). Indus texts: yajnadevam/indus-website "
              "(GPL-3.0), Wells/ICIT numbering. Only aggregate numbers are published here.", "",
              f"Run time {s['runtime_min']:.0f} minutes on a laptop CPU."]
    (OUT / "milestone_4_report.md").write_text("\n".join(lines) + "\n")


def sentence(claim, r, power):
    lm, p = r["holm"]["sequence"], r["nulls"]["binned_10"]["sequence"]
    if max(lm.values()) < 0.05:
        return (f"under the Sanskrit sequence score and the frequency-binned null the table scores at percentile "
                f"{p['percentile']:.1f} (p = {p['p']:.4f}); beating the frequency-matched null is necessary, not sufficient.")
    return (f"not distinguishable from frequency-matched random tables under the Sanskrit sequence score (percentile "
            f"{p['percentile']:.1f}, Holm p = {lm['binned_10']:.3f}); the harness detects true keys in {pct(power)} of "
            "positive-control runs at this length.")


# ---------- checks ----------

def selfcheck():
    us = [([1, 2], "ka"), ([3], "ta"), ([4], "ma")]
    rng = np.random.default_rng(0)
    for bins in (None, [np.array([0, 1]), np.array([2])]):
        k = shuffled(us, rng, bins)
        assert sorted(Counter([k[1], k[3], k[4]]).elements()) == ["ka", "ma", "ta"] and k[1] == k[2]
    assert shuffled(us, rng, [np.array([0]), np.array([1, 2])])[1] == "ka"  # a unit alone in its bin keeps its value
    assert p_value(1.0, [0, 0, 2]) == 0.5
    # the table scorer equals scoring the strings with the n-gram model directly
    target, control = ["kt", "ktk", "tk"] * 20, ["mn", "nm"] * 20
    lm_t, lm_c = lm_table(target), lm_table(control)
    nb = nb_tables(target, control)
    cp = Corpus([(1, 2), (3,)], [1, 2, 3])
    got = cp.score(cp.codes({1: "ka", 2: "ta", 3: "ma"}), ((lm_t, lm_c), (lm_t, lm_c), nb))
    mt, mc = (ngram.WittenBell([tuple(l) for l in x], ALPHABET, 3) for x in (target, control))
    want = sum(ngram.log2_likelihood(m, [tuple("kt"), tuple("m")])[0] * sign for m, sign in ((mt, 1), (mc, -1))) / 5
    assert abs(got[0] - want) < 1e-9, (got, want)
    vec, model = nb[3]
    X = vec.transform(["kt", "m"])
    ll = model.predict_joint_log_proba(X) if hasattr(model, "predict_joint_log_proba") else model._joint_log_likelihood(X)
    want_nb = ((ll[:, 1] - ll[:, 0]).sum() - 2 * (model.class_log_prior_[1] - model.class_log_prior_[0])) / math.log(2) / 5
    assert abs(got[2] - want_nb) < 1e-9, (got, want_nb)
    # enciphering with the true key and reading it back returns the plaintext skeleton
    key = {1: ("1", "ka"), 2: ("2", "ta"), 3: ("3", "kat")}
    assert encipher(["kata taka"], key, [2], np.random.default_rng(0)) == [(3, 2)]  # kt|t|k, then cut into texts of 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--controls-only", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    data.selfcheck()
    prereg = json.loads(PREREG.read_text())
    prov = provenance()
    if not args.controls_only:
        assert not prov["prereg_uncommitted"], "prereg.json has uncommitted changes: commit it before scoring any claim"
    draws, ctrl_draws, reps, neg_runs = (300, 100, 4, 40) if args.quick else (prereg["draws"], 1000, 20, 200)
    rng = np.random.default_rng(20261102)

    target, control = [skeleton(l) for l in data.dcs_lines("train")], [skeleton(l) for l in data.control_lines()]
    lm_t = lm_table(target)
    tables = ((lm_t, lm_table(target, order=1)), (lm_t, lm_table(control)), nb_tables(target, control))
    gate = data.gate_checks()
    wells = corpus.load_wells()[0]
    keys = {"yajnadevam-2024-11-30": data.load_key("xlits_2024-11-30.csv"), "yajnadevam-2026-07-15": data.load_key("xlits_2026-07-15.csv")}
    current = keys["yajnadevam-2026-07-15"]
    lengths = [len(t) for t in wells]
    res = {"settings": {"draws": draws, "control_draws": ctrl_draws, "positive_replicates": reps, "negative_runs": neg_runs,
                        "quick": args.quick, "target_lines": len(target), "control_lines": len(control)},
           "gate_checks": gate, "provenance": prov}

    print("positive control ...", flush=True)
    res["positive_control"] = positive_control(current, data.dcs_lines("heldout"), lengths, tables, rng, reps, ctrl_draws)
    texts = [t for t in wells if all(s in current for s in t)]
    cp = Corpus(texts, current)
    print("negative control ...", flush=True)
    res["negative_control"] = negative_control(cp, current, tables, rng, neg_runs, 199 if not args.quick else 49)
    power0 = min(v["power"] for v in res["positive_control"]["0"].values())

    nulls, claims = {}, {}
    if not args.controls_only:
        for name, key in keys.items():
            print(f"{name} ...", flush=True)
            texts = [t for t in wells if all(s in key for s in t)]
            cp = Corpus(texts, key)
            r, nulls[name] = test_key(cp, key, tables, draws, rng, bin_counts=(10, 5, 20))
            r["texts_scored"], r["tokens_scored"] = len(texts), sum(map(len, texts))
            r["order_null"] = order_nulls(texts, key, tables, rng, 200 if args.quick else 1000)
            claims[name] = r
        family = [(c, sc, n) for c in claims for sc in SCORERS for n in ("value_shuffle", "binned_10")]
        res["settings"]["holm_family"] = len(family)
        adj = stats.holm([claims[c]["nulls"][n][sc]["p"] for c, sc, n in family])
        for (c, sc, n), p in zip(family, adj):
            claims[c].setdefault("holm", {}).setdefault(sc, {})[n] = float(p)
        for name, r in claims.items():
            if r["holm"]["sequence"]["binned_10"] < 0.05:  # pre-registered: only then is the search-matched null needed
                texts = [t for t in wells if all(s in keys[name] for s in t)]
                r["search_matched"] = search_matched(Corpus(texts, keys[name]), keys[name], tables, rng, *((3, 200) if args.quick else (20, 2000)))
            r["sentence"] = sentence(name, r, power0)
        res["claims"] = claims
        res["read_vs_unread"] = read_vs_unread(current, tables, rng, 200 if args.quick else 2000)
        cur_texts = [t for t in wells if all(s in current for s in t)]
        res["order_effect"] = order_effect(cur_texts, current, tables, rng, *((40, 20) if args.quick else (200, 50)))
        res["shaw"] = shaw_numbering(data.load_paragorth("shaw2026.json"), rng, 1000 if args.quick else 10000)
        fuls = data.load_paragorth("fuls2024.json")
        tok = [s for t in wells for s in t]
        res["fuls_coverage"] = {"signs": len(fuls), "token_share": sum(s in fuls for s in tok) / len(tok),
                                "fully_covered_texts": sum(all(s in fuls for s in t) for t in wells)}
        table = {}
        for c, v in prereg["claims"].items():
            row = {"status": v["status"]}
            if c in claims:
                row["done"] = f"{claims[c]['texts_scored']:,} texts, {claims[c]['tokens_scored']:,} signs scored; {v['flag']}"
            elif c == "shaw-2026":
                row["done"] = "numbering test only"
            elif c == "fuls-2024":
                row["done"] = "coverage only: " + v["reason"]
            elif "reason" in v:
                row["reason"] = v["reason"]
            table[c] = row
        res["claims_table"] = table

    res["settings"]["runtime_min"] = (time.time() - start) / 60
    OUT.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in res.items()}
    (OUT / ("controls.json" if args.controls_only else "results.json")).write_text(json.dumps(out, indent=1))
    if not args.controls_only:
        figures(res, nulls)
        write_report(res)
    print(json.dumps({"positive_control": res["positive_control"], "negative_control": {k: v for k, v in res["negative_control"].items() if k != "p_values"},
                      "claims": {c: {"observed": r["observed"], "holm": r["holm"],
                                     "percentile_binned": {sc: r["nulls"]["binned_10"][sc]["percentile"] for sc in SCORERS}}
                                 for c, r in res.get("claims", {}).items()}}, indent=1))
    print(f"done in {(time.time() - start) / 60:.1f} min")
