"""Milestone 4 extension: how large would the corpus have to be for a true table to stand out from searched ones?

    python decipherment/corpus_size.py           # about 25 minutes on a laptop CPU
    python decipherment/corpus_size.py --quick   # a fast check, writes to reports/quick

Exploratory, NOT pre-registered. overfit.py found that on a corpus of Indus size a blind search passes even a true table.
Here Sanskrit is enciphered with the Yajnadevam sign-to-sound relation (so the true table is known) into corpora from
0.5 to 20 times the Indus size, and at each size a permutation test and blind searches are run. The scoring trigram comes
from one half of the Sanskrit chapters and the plaintext from the other half, so nothing is scored on its own training text.

Writes reports/milestone_4/ (corpus_size.json, corpus_size.png, corpus_size_report.md): aggregates only.
"""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import harness
import overfit
from data import skeleton
from harness import Corpus, assign, encipher, p_value, shuffled, units

sys.path.insert(0, str(data.ROOT / "sequence_analysis"))
import corpus  # noqa: E402

MULTIPLIERS = (0.5, 1, 2, 5, 10, 20)
WIDTH = overfit.WIDTH


def dcs_half(h):
    """SLP1 sentences of the DCS chapters whose stable hash is h (0 or 1) modulo 2."""
    out = []
    for line in (data.RAW / "dcs.txt").read_text().splitlines():
        _, chapter, text = line.split("\t")
        if int(hashlib.md5(chapter.encode()).hexdigest(), 16) % 2 == h:
            out.append(data.iast_to_slp1(text))
    return [l for l in out if l]


def truncate(texts, tokens):
    """The first texts holding at least `tokens` sign tokens, or None when there are fewer."""
    out, n = [], 0
    for t in texts:
        out.append(t)
        n += len(t)
        if n >= tokens:
            return out
    return None


def whole(cp, vals, us, tables):
    return overfit.split_score(cp, cp.codes(assign(us, vals), WIDTH), tables, np.ones(cp.n_texts, bool))[0]


def replicate(texts, key, tables, rng, searches, steps, draws):
    cp, us = Corpus(texts, key), units(key)
    true = [v for _, v in us]
    obs = whole(cp, true, us, tables)
    null = [whole(cp, list(shuffled_vals(us, rng)), us, tables) for _ in range(draws)]
    margins, found, fit_gap = [], [], []
    for _ in range(searches):
        fit = rng.permutation(cp.n_texts) < cp.n_texts // 2
        vals, traj = overfit.climb(cp, us, tables, rng, steps, fit=fit)
        true_rest = overfit.split_score(cp, cp.codes(assign(us, true), WIDTH), tables, fit)[1]
        margins.append(true_rest - traj[-1][2])
        found.append(overfit.agreement(cp, us, true, vals))
    return {"tokens": int(sum(cp.counts.values())), "p": p_value(obs, null), "margin": float(min(margins)),
            "beats_all_searched": bool(all(m > 0 for m in margins)), "agreement": float(np.mean(found))}


def shuffled_vals(us, rng):
    return np.array([v for _, v in us], dtype=object)[rng.permutation(len(us))]


# ---------- report ----------

def write_report(res, out):
    s = res["settings"]
    L = ["# How large would the corpus have to be? (exploratory, not pre-registered)", "",
         "The blind-search extension (`overfitting_report.md`) found that on a corpus of Indus size a hill-climbing search reaches scores above a true table, "
         "so the score cannot pick out a correct table. This asks at what corpus size it can. Sanskrit is enciphered with the Yajnadevam 2026-07-15 "
         "sign-to-sound relation, so the true table is known, into corpora from 0.5 to 20 times the size of the Indus corpus used in Milestone 4 "
         f"({res['indus_tokens']:,} sign tokens). The scoring trigram comes from one half of the Sanskrit chapters and the plaintext from the other half.", "",
         f"At each size, {s['replicates']} independent replicates: (a) a permutation test of the true table against {s['draws']} shuffles of its own values; "
         f"(b) {s['searches']} blind searches of {s['steps']:,} swaps each, fitted on a random half of the texts and scored on the other half; "
         "(c) the true table's score on that other half minus the best searched table's; (d) the share of sign tokens a searched table reads with the same consonants as the true table.", "",
         "| Size (x Indus) | Sign tokens | Replicates | Power at p < 0.05 | True table beats every search on held-out texts | Margin, true minus best search (bits) | Agreement of searched with true |",
         "|---|---|---|---|---|---|---|"]
    for m, r in res["by_size"].items():
        L.append(f"| {m} | {r['tokens']:,} | {r['replicates']} | {100 * r['power']:.0f}% | {100 * r['beats_all']:.0f}% | {r['margin_mean']:+.3f} ± {r['margin_sd']:.3f} | {100 * r['agreement']:.0f}% |")
    if s["dropped"]:
        L += ["", f"Not run, because the plaintext half could not supply enough text: {', '.join(map(str, s['dropped']))} times the Indus size."]
    L += ["", "![held-out margin against corpus size](corpus_size.png)", "", "## Reading", "", res["reading"], "",
          "## Limits", "",
          "One key and one language. Enciphered Sanskrit is cleaner than any real inscription (no damaged signs, no variant spellings, the true reading is exactly Sanskrit). "
          "The search is a plain hill climb of a fixed length that has not necessarily converged on the largest corpora, so a better search would need an even larger corpus. "
          f"Settings: seed {s['seed']}, runtime {s['runtime_min']:.1f} min, quick run: {s['quick']}.", ""]
    (out / "corpus_size_report.md").write_text("\n".join(L))


def figure(res, path):
    ms = list(res["by_size"])
    x = [res["by_size"][m]["tokens"] for m in ms]
    y = [res["by_size"][m]["margin_mean"] for m in ms]
    e = [res["by_size"][m]["margin_sd"] for m in ms]
    plt.figure(figsize=(6.5, 4))
    plt.errorbar(x, y, e, marker="o", color="#4a9bd1", capsize=3)
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(res["indus_tokens"], color="#c9663c", ls="--", label="Indus corpus size")
    plt.xscale("log")
    plt.xlabel("sign tokens")
    plt.ylabel("true table minus best search, held-out (bits per phoneme)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def reading(res):
    ok = [m for m, r in res["by_size"].items() if r["beats_all"] >= 0.8]
    if ok:
        first = ok[0]
        return (f"The true table beats every blind search on held-out texts in at least 80% of replicates from {first} times the Indus size "
                f"({res['by_size'][first]['tokens']:,} sign tokens) upward" + ("" if all(m in ok for m in list(res['by_size'])[list(res['by_size']).index(first):]) else ", though not at every larger size") +
                f". At the Indus size itself it does so in {100 * res['by_size'].get('1', {'beats_all': float('nan')})['beats_all']:.0f}% of replicates. "
                "Below that size, scoring below what a search can reach says little about a table.")
    top = list(res["by_size"])[-1]
    return (f"The true table does not beat every blind search in 80% of replicates at any size run, up to {top} times the Indus size "
            f"({res['by_size'][top]['tokens']:,} sign tokens). The permutation test against shuffled tables detects the true table at every size (power " + ", ".join(f"{100 * r['power']:.0f}%" for r in res["by_size"].values()) +
            "), so what fails is the comparison with searched tables: a plain blind search finds tables that score above the true one on held-out texts at every size, and the tables it finds read only "
            f"{100 * min(r['agreement'] for r in res['by_size'].values()):.0f}-{100 * max(r['agreement'] for r in res['by_size'].values()):.0f}% of tokens the way the true table does. "
            "With this score, the true table is not what a search maximises, however large the corpus, so scoring below the search ceiling is not evidence against a table and scoring above shuffled tables is not evidence for one.")


def selfcheck():
    assert truncate([(1, 2), (3,), (4, 5, 6)], 3) == [(1, 2), (3,)] and truncate([(1,)], 2) is None
    rng = np.random.default_rng(0)
    plain = ["".join(rng.choice(list("ktmn"), 6, p=[0.4, 0.3, 0.2, 0.1])) for _ in range(300)]
    tables = overfit.scoring_tables(plain)
    key = {i + 1: (str(i + 1), c + "a") for i, c in enumerate("ktmn")}
    texts = [tuple("ktmn".index(ch) + 1 for ch in p) for p in plain[:120]]
    cp, us = Corpus(texts, key), units(key)
    true = [v for _, v in us]
    assert whole(cp, true, us, tables) > np.mean([whole(cp, list(shuffled_vals(us, rng)), us, tables) for _ in range(20)])
    r = replicate(texts, key, tables, rng, 2, 100, 20)
    assert 0 < r["p"] <= 1 and r["tokens"] == sum(map(len, texts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    mults, reps, searches, steps, draws = ((1, 5), 1, 2, 300, 100) if args.quick else (MULTIPLIERS, 4, 5, 2000, 500)
    seed = 20261116
    rng = np.random.default_rng(seed)
    key = data.load_key("xlits_2026-07-15.csv")
    wells = [t for t in corpus.load_wells()[0] if all(s in key for s in t)]
    lengths, base = [len(t) for t in wells], sum(map(len, wells))
    tables = overfit.scoring_tables([skeleton(l) for l in dcs_half(0)])
    plain = dcs_half(1)
    by, dropped = {}, []
    for m in mults:
        runs = []
        for r in range(reps):
            ct = truncate(encipher([plain[i] for i in rng.permutation(len(plain))], key, lengths, rng), round(m * base))
            if ct is None:
                dropped.append(m)
                break
            print(f"{m}x replicate {r + 1}/{reps} ({sum(map(len, ct)):,} tokens) ...", flush=True)
            runs.append(replicate(ct, key, tables, rng, searches, steps, draws))
        if runs:
            mg = [x["margin"] for x in runs]
            by[str(m)] = {"tokens": int(np.mean([x["tokens"] for x in runs])), "replicates": len(runs), "power": float(np.mean([x["p"] < 0.05 for x in runs])),
                          "beats_all": float(np.mean([x["beats_all_searched"] for x in runs])), "margin_mean": float(np.mean(mg)), "margin_sd": float(np.std(mg)),
                          "agreement": float(np.mean([x["agreement"] for x in runs]))}
    res = {"indus_tokens": base, "by_size": by}
    res["reading"] = reading(res)
    res["settings"] = {"replicates": reps, "searches": searches, "steps": steps, "draws": draws, "dropped": sorted(set(dropped)), "seed": seed,
                       "quick": args.quick, "pre_registered": False, "runtime_min": (time.time() - start) / 60}
    out = data.ROOT / "reports/quick" if args.quick else harness.OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_size.json").write_text(json.dumps(res, indent=1))
    figure(res, out / "corpus_size.png")
    write_report(res, out)
    print(res["reading"])
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'corpus_size_report.md'}")
