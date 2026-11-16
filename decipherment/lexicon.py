"""Milestone 4 extension: dictionary match rate of a claimed table's output.

    python decipherment/lexicon.py           # about 10 minutes on a laptop CPU
    python decipherment/lexicon.py --quick   # a fast check, writes to reports/quick

Exploratory, NOT pre-registered. The Milestone 4 score reads letter order; this one reads word matches, which is what
decipherers usually present as evidence. A table is applied to the inscriptions and its consonant-skeleton output is
compared with a lexicon of Sanskrit word skeletons (words seen at least twice in the DCS training split). Short skeletons
match by chance, so a table is only ever compared with shuffled versions of itself, never with zero.

Writes reports/milestone_4/ (lexicon.json, lexicon_report.md): aggregates only.
"""
import argparse
import json
import sys
import time
from collections import Counter
from functools import lru_cache

import numpy as np

import data
import harness
from data import skeleton
from harness import corrupt, encipher, freq_bins, p_value, shuffled, units

sys.path.insert(0, str(data.ROOT / "sequence_analysis"))
import corpus  # noqa: E402
import stats  # noqa: E402

MIN_LEN, MIN_COUNT, MAX_LEN, END = 3, 2, 12, "$"
SCORES = ("whole_text_match", "coverage")
NULLS = ("value_shuffle", "binned_10")
sk = lru_cache(maxsize=None)(skeleton)


# ---------- lexicon and scores ----------

def build_lexicon(lines):
    """Skeletons of words (length MIN_LEN..MAX_LEN) seen at least MIN_COUNT times: (set, prefix trie)."""
    counts = Counter(sk(w) for l in lines for w in l.split())
    words = {w for w, c in counts.items() if MIN_LEN <= len(w) <= MAX_LEN and c >= MIN_COUNT}
    trie = {}
    for w in words:
        node = trie
        for ch in w:
            node = node.setdefault(ch, {})
        node[END] = True
    return words, trie


def cover(s, trie):
    """Letters of s covered by greedy left-to-right longest lexicon matches; a letter no entry starts at is skipped."""
    i, n, got = 0, len(s), 0
    while i < n:
        node, j, best = trie, i, 0
        while j < n and s[j] in node:
            node = node[s[j]]
            j += 1
            if END in node:
                best = j - i
        if best:
            got, i = got + best, i + best
        else:
            i += 1
    return got


def scores(texts, key_values, lex):
    """(share of outputs of length >= MIN_LEN that are lexicon entries, share of output letters covered)."""
    words, trie = lex
    outs = ["".join(key_values[s] for s in t) for t in texts]
    long_ = [o for o in outs if len(o) >= MIN_LEN]
    total = sum(map(len, outs))
    return np.array([sum(o in words for o in long_) / max(1, len(long_)), sum(cover(o, trie) for o in outs) / max(1, total)])


def skel_values(values):
    return {s: sk(v) for s, v in values.items()}


def null(texts, us, draws, rng, lex, bins=None):
    return np.array([scores(texts, skel_values(shuffled(us, rng, bins)), lex) for _ in range(draws)])


def test(texts, key, draws, rng, lex, nulls=NULLS):
    """Observed scores and shuffle nulls (value shuffle, and shuffle within 10 frequency bins) for one key."""
    us = units(key)
    counts = Counter(s for t in texts for s in t)
    obs = scores(texts, skel_values({s: cv[1] for s, cv in key.items()}), lex)
    arrs = {"value_shuffle": null(texts, us, draws, rng, lex)}
    if "binned_10" in nulls:
        arrs["binned_10"] = null(texts, us, draws, rng, lex, freq_bins(us, counts, 10))
    return {"observed": dict(zip(SCORES, map(float, obs))),
            "nulls": {n: {sc: {"mean": float(a[:, i].mean()), "sd": float(a[:, i].std()), "p": p_value(obs[i], a[:, i]),
                               "percentile": float(100 * (a[:, i] < obs[i]).mean())} for i, sc in enumerate(SCORES)}
                      for n, a in arrs.items()}}


# ---------- controls ----------

def positive_control(key, plain, lengths, lex, rng, replicates, draws, corruptions=(0, 0.1, 0.2, 0.3)):
    """Sanskrit enciphered with the key, then tested with the key corrupted by a share: how often is p < 0.05."""
    out = {}
    for share in corruptions:
        ps = {(n, sc): [] for n in NULLS for sc in SCORES}
        for _ in range(replicates):
            texts = encipher([plain[i] for i in rng.permutation(len(plain))[:400]], key, lengths, rng)
            r = test(texts, corrupt(key, share, rng) if share else key, draws, rng, lex)
            for n in NULLS:
                for sc in SCORES:
                    ps[(n, sc)].append(r["nulls"][n][sc]["p"])
        out[str(share)] = {n: {sc: float(np.mean(np.array(ps[(n, sc)]) < 0.05)) for sc in SCORES} for n in NULLS}
    return out


def negative_control(texts, key, lex, rng, runs, draws):
    """A random shuffle of the key tested against its own shuffles: p should be uniform, so about 5% fall below 0.05."""
    us = units(key)
    ps = {sc: [] for sc in SCORES}
    for _ in range(runs):
        obs = scores(texts, skel_values(shuffled(us, rng)), lex)
        nl = null(texts, us, draws, rng, lex)
        for i, sc in enumerate(SCORES):
            ps[sc].append(p_value(obs[i], nl[:, i]))
    return {"runs": runs, "draws": draws, "share_below_0.05": {sc: float(np.mean(np.array(v) < 0.05)) for sc, v in ps.items()}}


# ---------- report ----------

def pct(x):
    return f"{100 * x:.1f}%"


def write_report(res, out):
    s = res["settings"]
    L = ["# Dictionary match rate (exploratory, not pre-registered)", "",
         "A claimed table is applied to the Wells/ICIT inscriptions and its output is reduced to the consonant skeleton used in "
         "Milestone 4. Two scores are computed: the share of outputs (length 3 or more) that are exactly a word of the lexicon, "
         "and the share of output letters covered by a greedy longest-match segmentation into lexicon words. Each is compared with "
         "the same score for shuffled versions of the table, either shuffled over all signs or only within 10 frequency bins.", "",
         f"Lexicon: skeletons of words of {MIN_LEN} to {MAX_LEN} letters seen at least {MIN_COUNT} times in the DCS training split: "
         f"{res['lexicon']['sanskrit']:,} entries (Old Tamil control lexicon: {res['lexicon']['tamil']:,}). Short skeletons match by chance, "
         "which is why only the comparison with shuffled tables means anything. Vowels are ignored, so one skeleton stands for many words.", "",
         "## Controls", "",
         f"Positive control: Sanskrit enciphered with the claimed key, {s['control_replicates']} replicates per row, {s['control_draws']} draws each. "
         "Share of replicates with p < 0.05 when the key is correct and when a share of its values is replaced.", "",
         "| Values replaced | Score | Value-shuffle null | Binned null |", "|---|---|---|---|"]
    for share, r in res["positive_control"].items():
        for sc in SCORES:
            L.append(f"| {pct(float(share))} | {sc.replace('_', ' ')} | {pct(r['value_shuffle'][sc])} | {pct(r['binned_10'][sc])} |")
    nc = res["negative_control"]
    L += ["", f"Negative control: {nc['runs']} random shuffles of the key, each tested against its own shuffles ({nc['draws']} draws). "
          "Share with p < 0.05 (about 5% expected): " + ", ".join(f"{sc.replace('_', ' ')} {pct(v)}" for sc, v in nc["share_below_0.05"].items()) + ".", ""]
    weak = [sc for sc in SCORES if res["positive_control"]["0"]["binned_10"][sc] < 0.8]
    if weak:
        L += [f"The positive control detects a correct key less than 80% of the time for: {', '.join(w.replace('_', ' ') for w in weak)}. "
              "For those scores a null result on a claimed key is uninformative.", ""]
    L += ["## Claimed keys", "", "| Key | Lexicon | Score | Observed | Value-shuffle: mean (sd), percentile, p, Holm p | Binned: mean (sd), percentile, p, Holm p |", "|---|---|---|---|---|---|"]
    for name, r in res["claims"].items():
        for lang in ("sanskrit", "tamil"):
            for sc in SCORES:
                cells = []
                for n in NULLS:
                    d = r[lang]["nulls"][n][sc]
                    holm = f"{res['holm'][name][sc][n]:.3f}" if lang == "sanskrit" else "-"
                    cells.append(f"{d['mean']:.4f} ({d['sd']:.4f}), {d['percentile']:.1f}, {d['p']:.4f}, {holm}")
                L.append(f"| {name} ({r['texts']:,} texts) | {lang} | {sc.replace('_', ' ')} | {r[lang]['observed'][sc]:.4f} | {cells[0]} | {cells[1]} |")
    L += ["", f"Holm correction over {res['settings']['holm_family']} tests (2 keys, 2 scores, 2 nulls, Sanskrit lexicon). The Old Tamil rows are a control "
          "for what the score measures: a table that matches the Tamil lexicon as well as the Sanskrit one is not showing anything Sanskrit.", "",
          "## Reading", "", res["reading"], "",
          "## Limits", "",
          "Skeletons ignore vowels, so many unrelated words share one skeleton. The lexicon comes from five post-Vedic texts only. Greedy segmentation "
          "can miss better covers. The test can reject a specific table under this score; it cannot show whether the script records a language. "
          "Not pre-registered and outside the corrected family of Milestone 4 tests.", "",
          f"Settings: {s['draws']} draws per null, seed {s['seed']}, runtime {s['runtime_min']:.1f} min, quick run: {s['quick']}.", ""]
    (out / "lexicon_report.md").write_text("\n".join(L))


def reading(res):
    bits = []
    for name, r in res["claims"].items():
        hit = {n: [sc for sc in SCORES if res["holm"][name][sc][n] < 0.05] for n in NULLS}
        names = lambda l: " and ".join(sc.replace("_", " ") for sc in l)
        if hit["binned_10"]:
            bits.append(f"{name}: beats tables shuffled within frequency bins on {names(hit['binned_10'])} (Holm corrected).")
        elif hit["value_shuffle"]:
            pcts = ", ".join(f"percentile {r['sanskrit']['nulls']['binned_10'][sc]['percentile']:.0f} on {sc.replace('_', ' ')}" for sc in SCORES)
            bits.append(f"{name}: beats fully shuffled tables on {names(hit['value_shuffle'])} but not tables shuffled within frequency bins ({pcts}), "
                        "so the excess is explained by how often each sign occurs, not by which sound it carries.")
        else:
            bits.append(f"{name}: not distinguishable from shuffled versions of itself on either score after Holm correction.")
    t = [(n, sc) for n in res["claims"] for sc in SCORES if res["claims"][n]["tamil"]["nulls"]["binned_10"][sc]["p"] < 0.05]
    bits.append("Against the Old Tamil lexicon the frequency-binned test is significant for " + (", ".join(f"{n} {sc.replace('_', ' ')}" for n, sc in t) + ", so that score does not single out Sanskrit." if t else "no key and score."))
    return " ".join(bits)


def selfcheck():
    lex = build_lexicon(["katama katamana katamana katama zaza"])  # skeletons ktm x2, ktmn x2; "ss" is too short
    assert lex[0] == {"ktm", "ktmn"}
    assert cover("ktmnxktm", lex[1]) == 7 and cover("nnn", lex[1]) == 0 and cover("ktmn", lex[1]) == 4  # longest match wins
    key = {1: ("a", "ka"), 2: ("b", "ta"), 3: ("c", "ma"), 4: ("d", "na")}
    vals = skel_values({s: cv[1] for s, cv in key.items()})
    assert vals == {1: "k", 2: "t", 3: "m", 4: "n"}
    got = scores([(1, 2, 3), (1, 2), (4, 4, 4)], vals, lex)  # ktm is a word, kt is too short to score, nnn is not
    assert got[0] == 0.5 and abs(got[1] - 3 / 8) < 1e-12, got
    got = scores([(1, 2, 3), (1, 2, 3, 4, 4)], vals, lex)  # ktmnn: ktmn covered, last n not
    assert got[0] == 0.5 and abs(got[1] - 7 / 8) < 1e-12, got
    rng = np.random.default_rng(0)
    assert sorted(skel_values(shuffled(units(key), rng)).values()) == ["k", "m", "n", "t"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    draws, c_reps, c_draws, n_runs, n_draws = (100, 3, 60, 10, 60) if args.quick else (2000, 20, 500, 100, 199)
    seed = 20261116
    rng = np.random.default_rng(seed)
    lexes = {"sanskrit": build_lexicon(data.dcs_lines("train")), "tamil": build_lexicon(data.control_lines())}
    wells = corpus.load_wells()[0]
    keys = {"yajnadevam-2024-11-30": data.load_key("xlits_2024-11-30.csv"), "yajnadevam-2026-07-15": data.load_key("xlits_2026-07-15.csv")}
    res = {"lexicon": {k: len(v[0]) for k, v in lexes.items()}, "claims": {}}
    for name, key in keys.items():
        print(name, "...", flush=True)
        texts = [t for t in wells if all(x in key for x in t)]
        res["claims"][name] = {"texts": len(texts), "tokens": sum(map(len, texts)),
                               **{lang: test(texts, key, draws, rng, lex) for lang, lex in lexes.items()}}
    family = [(c, sc, n) for c in res["claims"] for sc in SCORES for n in NULLS]
    adj = stats.holm([res["claims"][c]["sanskrit"]["nulls"][n][sc]["p"] for c, sc, n in family])
    res["holm"] = {c: {sc: {} for sc in SCORES} for c in res["claims"]}
    for (c, sc, n), p in zip(family, adj):
        res["holm"][c][sc][n] = float(p)
    current = keys["yajnadevam-2026-07-15"]
    print("positive control ...", flush=True)
    res["positive_control"] = positive_control(current, data.dcs_lines("heldout"), [len(t) for t in wells], lexes["sanskrit"], rng, c_reps, c_draws)
    print("negative control ...", flush=True)
    res["negative_control"] = negative_control([t for t in wells if all(x in current for x in t)], current, lexes["sanskrit"], rng, n_runs, n_draws)
    res["reading"] = reading(res)
    res["settings"] = {"draws": draws, "control_replicates": c_reps, "control_draws": c_draws, "holm_family": len(family), "seed": seed,
                       "quick": args.quick, "pre_registered": False, "runtime_min": (time.time() - start) / 60}
    out = data.ROOT / "reports/quick" if args.quick else harness.OUT  # reports/quick is gitignored
    out.mkdir(parents=True, exist_ok=True)
    (out / "lexicon.json").write_text(json.dumps(res, indent=1))
    write_report(res, out)
    print(res["reading"])
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'lexicon_report.md'}")
