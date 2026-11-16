"""Milestone 2 extension: do the stroke signs behave like numerals?

    python sequence_analysis/numerals.py           # about 1 minute on a laptop CPU
    python sequence_analysis/numerals.py --quick   # a fast check, writes to reports/quick

Exploratory. Signs drawn as a row of equal thin bars are often called numerals. Real numerals predict several things, and
each can fail: they are followed by a limited set of counted items, numerals of different value are interchangeable, their
frequency falls with value, they are not scattered randomly through a text, and in an additive system they run together.
Each is tested against pseudo sets of non-stroke signs of matching frequency. The stroke signs are found from the glyph
drawings by a stated rule, not a hand-typed list.

Writes reports/milestone_2/ (numerals.json, numerals.png, numerals_contact_sheet.png, numerals_report.md): aggregates and sign numbers only.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import stats
from corpus import ROOT, ebuds, load_lines

sys.path.insert(0, str(ROOT / "grapheme_extraction"))
import synthetic_generator as gen  # noqa: E402

OUT = ROOT / "reports/milestone_2"
MIN_TOKENS = 10  # a sign needs this many tokens to have a context distribution worth comparing
POOL = 20  # frequency-matched candidates per stroke sign


# ---------- finding the stroke signs ----------

def stroke_info(img, fill=0.5):
    """None, or {count, long, rows} when every component of the drawing is a thin near-vertical bar and the bars are of about
    the same size: height at least 3 times width (5 times for a lone bar), the component filling at least `fill` of its bounding
    box (a bar, not an arc or a loop), heights within 20% of each other and widths within a factor of 2.
    long: bars taller than a third of the image."""
    n, _, st, cen = cv2.connectedComponentsWithStats((img > 127).astype(np.uint8), connectivity=8)
    bars = [(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], cen[i, 1], st[i, cv2.CC_STAT_AREA]) for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= 6]
    if not bars or any(h < 3 * w or a < fill * w * h for w, h, _, a in bars) or (len(bars) == 1 and bars[0][1] < 5 * bars[0][0]):
        return None
    if max(w for w, _, _, _ in bars) > 2 * min(w for w, _, _, _ in bars):
        return None
    bars = [b[:3] for b in bars]
    hs = [h for _, h, _ in bars]
    if (max(hs) - min(hs)) / max(hs) >= 0.2:
        return None
    ys = [y for _, _, y in bars]
    return {"count": len(bars), "long": bool(np.median(hs) > img.shape[0] / 3), "rows": 1 if max(ys) - min(ys) < 0.5 * np.median(hs) else 2}


def contact_sheet(info, path):
    cols = 8
    rows = (len(info) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (s, d) in zip(axes.ravel(), info.items()):
        ax.imshow(cv2.imread(str(gen.GLYPH_DIR / f"{s}.png"), cv2.IMREAD_GRAYSCALE), cmap="gray")
        ax.set_title(f"{s}: {d['count']}{' L' if d['long'] else ' S'}{d['rows']}", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close()


# ---------- statistics on one set of signs ----------

def interchange(texts, X):
    """Pairs of texts of equal length that differ in exactly one position, where both signs there are in X."""
    slots = defaultdict(list)
    for t in texts:
        for i, s in enumerate(t):
            slots[t[:i] + (None,) + t[i + 1:]].append(s)
    total = 0
    for v in slots.values():
        k = Counter(s for s in v if s in X)
        total += (sum(k.values()) ** 2 - sum(c * c for c in k.values())) // 2
    return total


def measures(texts, X, vec, cnt):
    big = [s for s in X if cnt[s] >= MIN_TOKENS]
    tokens = sum(cnt[s] for s in X)
    first = sum(t[0] in X for t in texts)
    last = sum(t[-1] in X for t in texts)
    return {"entropy_next": float(np.mean([stats.entropy(vec[s]) for s in big])) if big else float("nan"),
            "js_between": float(np.mean([stats.js_divergence(vec[a], vec[b]) for a, b in combinations(big, 2)])) if len(big) > 1 else float("nan"),
            "interchange": interchange(texts, X), "at_end": last / tokens, "at_start": first / tokens,
            "adjacent": sum(a in X and b in X for t in texts for a, b in zip(t, t[1:]))}


def tail(obs, null, side):
    null = np.asarray(null)
    null = null[~np.isnan(null)]
    if side == "lower":
        return float((1 + (null <= obs).sum()) / (len(null) + 1))
    if side == "upper":
        return float((1 + (null >= obs).sum()) / (len(null) + 1))
    return float((1 + (abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / (len(null) + 1))


TESTS = [("entropy_next", "lower", "Counted things: stroke signs are followed by a narrower set of signs (entropy of the next sign, bits)"),
         ("js_between", "lower", "Counted things: different stroke signs are followed by the same signs (mean Jensen-Shannon divergence)"),
         ("interchange", "upper", "Interchangeability: text pairs differing only in which stroke sign stands in one position"),
         ("at_end", "two", "Position: share of stroke-sign tokens at the end of a text"),
         ("adjacent", "two", "Runs: adjacent pairs of stroke signs (additive systems show them, positional ones do not)")]


# ---------- report ----------

def write_report(res, out):
    s = res["settings"]
    L = ["# Do the stroke signs behave like numerals? (exploratory)", "",
         "Signs drawn as a row of equal thin bars are often called numerals. If they are, several things should hold, and each can fail. "
         f"The stroke signs are found from the 417 glyph drawings by a rule: every connected part is a thin near-vertical bar (height at least 3 times width) and the bars are within 20% of one another in height. "
         f"The rule selected {len(res['stroke_signs'])} signs (contact sheet: `numerals_contact_sheet.png`; label = sign number, bar count, S or L for short or long bars, and the number of rows). "
         f"{res['adjustment']}", "",
         f"Texts: the {res['texts']:,} distinct single-line inscriptions of the filtered corpus (Milestone 2). Each stroke-sign measure is compared with {s['draws']:,} pseudo sets of the same size made of non-stroke signs, "
         f"each chosen at random from the {POOL} non-stroke signs nearest in frequency to the stroke sign it replaces. Contexts need at least {MIN_TOKENS} tokens, so {res['signs_with_context']} stroke signs enter the context tests. "
         "All six p-values form one Benjamini-Hochberg family.", "",
         "| Stroke sign | Count | Bars | Rows | Tokens |", "|---|---|---|---|---|"]
    for sg, d in res["stroke_signs"].items():
        L.append(f"| {sg} | {d['count']} | {'long' if d['long'] else 'short'} | {d['rows']} | {d['tokens']} |")
    L += ["", "## Results", "", "| Prediction | Stroke signs | Pseudo sets: mean (sd) | p | BH q |", "|---|---|---|---|---|"]
    for name, side, label in TESTS:
        r = res["tests"][name]
        L.append(f"| {label} | {r['observed']:.3f} | {r['null_mean']:.3f} ({r['null_sd']:.3f}) | {r['p']:.4f} | {r['q']:.3f} |")
    f = res["frequency"]
    L.append(f"| Frequency falls with value: Spearman rank correlation of bar count with tokens | {f['rho']:+.2f} | - | {f['p']:.4f} | {f['q']:.3f} |")
    L += ["", f"Frequency by count (tokens): short bars {f['short']}; long bars {f['long']}. Separate correlations: short {f['rho_short']}, long {f['rho_long']}.", "",
          f"Share of tokens at the start of a text: stroke signs {res['tests']['at_end']['at_start']:.3f}, all signs {res['all_start']:.3f}; at the end: stroke signs {res['tests']['at_end']['observed']:.3f}, all signs {res['all_end']:.3f}.", "",
          "![stroke signs](numerals.png)", "", "## Reading", ""] + [f"- {x}" for x in res["reading"]] + [
          "", "This does not decide whether the signs are numerals. It lists which numeral-like behaviours are present and which are not.", "",
          "## Limits", "", "The detection rule fixes what counts as a stroke sign; a different rule gives a different set. Many stroke signs are rare, so context distributions are thin. "
          "Pseudo sets are matched on frequency only. Texts are short. "
          f"Settings: seed {s['seed']}, runtime {s['runtime_min']:.1f} min, quick run: {s['quick']}.", ""]
    (out / "numerals_report.md").write_text("\n".join(L))


def figure(res, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for kind, color in (("short", "#4a9bd1"), ("long", "#c9663c")):
        d = res["frequency"][kind + "_by_count"]
        ax.plot(list(map(int, d)), list(d.values()), marker="o", color=color, label=f"{kind} bars")
    ax.set_xlabel("bar count")
    ax.set_ylabel("tokens")
    ax.set_yscale("log")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def reading(res):
    out = []
    for name, side, label in TESTS:
        r = res["tests"][name]
        held = r["q"] < 0.05
        if side == "two":
            out.append(f"{label}: {'differs from' if held else 'does not differ from'} matched pseudo sets (observed {r['observed']:.3f} against {r['null_mean']:.3f}, q {r['q']:.3f}); this test has no single numeral-like direction.")
        else:
            out.append(f"{label}: {'holds' if held else 'does not hold'} (observed {r['observed']:.3f} against {r['null_mean']:.3f} for matched pseudo sets, q {r['q']:.3f}); "
                       f"a numeral-like result would be {'lower' if side == 'lower' else 'higher'}.")
    f = res["frequency"]
    out.append(f"Frequency by value: rank correlation {f['rho']:+.2f} (q {f['q']:.3f}); numerals would fall with value, so a negative correlation is the numeral-like result.")
    return out


def selfcheck():
    img = lambda: np.zeros((150, 150), np.uint8)
    a = img()
    for x in (20, 40, 60):
        a[30:90, x:x + 6] = 255
    assert stroke_info(a) == {"count": 3, "long": True, "rows": 1}
    b = img()
    cv2.circle(b, (75, 75), 40, 255, 6)
    assert stroke_info(b) is None
    c = img()
    for x in (20, 40, 60):
        c[10:50, x:x + 6] = 255
        c[90:130, x:x + 6] = 255
    assert stroke_info(c) == {"count": 6, "long": False, "rows": 2}
    e = img()
    cv2.ellipse(e, (75, 75), (10, 60), 0, 0, 360, 255, 3)  # tall but hollow: a loop, not a bar
    assert stroke_info(e) is None and stroke_info(e, fill=0) == {"count": 1, "long": True, "rows": 1}
    d = img()
    d[30:90, 20:26] = 255
    d[30:60, 40:46] = 255
    assert stroke_info(d) is None
    assert interchange([(1, 5), (2, 5), (1, 6), (3, 5)], {1, 2}) == 1  # only (1,5)/(2,5); 3 is outside the set
    assert tail(0.0, [0.0, 1.0, 2.0], "lower") == 0.5 and tail(2.0, [0.0, 1.0, 2.0], "upper") == 0.5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    draws, seed = (100 if args.quick else 1000), 20261116
    rng = np.random.default_rng(seed)
    glyphs = {i: cv2.imread(str(gen.GLYPH_DIR / f"{i}.png"), cv2.IMREAD_GRAYSCALE) for i in range(1, gen.N_SIGNS + 1)}
    info = {i: d for i, g in glyphs.items() if (d := stroke_info(g))}
    loose = sorted(i for i, g in glyphs.items() if stroke_info(g, fill=0) and i not in info)
    texts = [t for t in ebuds(load_lines())[0]]
    cnt = Counter(s for t in texts for s in t)
    info = {i: d for i, d in info.items() if cnt[i] > 0}
    out = Path(ROOT / "reports/quick") if args.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    contact_sheet(info, out / "numerals_contact_sheet.png")
    vocab = sorted(cnt)
    col = {s: i for i, s in enumerate(vocab + ["END"])}
    vec = {s: np.zeros(len(col)) for s in vocab}
    for t in texts:
        for a, b in zip(t, t[1:] + ("END",)):
            vec[a][col[b]] += 1
    S = set(info)
    others = sorted(s for s in vocab if s not in S)
    pool = {s: sorted(others, key=lambda o: abs(np.log(cnt[o]) - np.log(cnt[s])))[:POOL] for s in S}
    obs = measures(texts, S, vec, cnt)
    nulls = defaultdict(list)
    for _ in range(draws):
        chosen = set()
        for s in sorted(S):
            free = [o for o in pool[s] if o not in chosen] or pool[s]
            chosen.add(free[rng.integers(len(free))])
        for k, v in measures(texts, chosen, vec, cnt).items():
            nulls[k].append(v)
    tests = {name: {"observed": obs[name], "null_mean": float(np.nanmean(nulls[name])), "null_sd": float(np.nanstd(nulls[name])), "p": tail(obs[name], nulls[name], side),
                    **({"at_start": obs["at_start"]} if name == "at_end" else {})} for name, side, _ in TESTS}
    by = {kind: Counter() for kind in ("short", "long")}
    for s, d in info.items():
        by["long" if d["long"] else "short"][d["count"]] += cnt[s]
    pairs = [(d["count"], cnt[s]) for s, d in info.items()]
    rho_p = spearmanr(*zip(*pairs)) if len(pairs) > 2 else (float("nan"), 1.0)
    rho = lambda kind: round(float(spearmanr(*zip(*[(c, n) for c, n in by[kind].items()]))[0]), 2) if len(by[kind]) > 2 else None
    fam = [tests[n]["p"] for n, _, _ in TESTS] + [float(rho_p[1]) if rho_p[1] == rho_p[1] else 1.0]
    q = stats.benjamini_hochberg(fam)
    for (n, _, _), qq in zip(TESTS, q):
        tests[n]["q"] = float(qq)
    res = {"texts": len(texts), "settings": {"draws": draws, "seed": seed, "quick": args.quick, "runtime_min": 0},
           "adjustment": (f"A first version of the rule, without the fill condition, also selected {len(loose)} curved or looped signs ({', '.join(map(str, loose))}); requiring each part to fill at least half of its bounding box, a lone bar to be at least 5 times as tall as wide, and bar widths to agree within a factor of 2 removed them. "
                          "Bars are called long when taller than a third of the drawing."), "signs_with_context": sum(cnt[s] >= MIN_TOKENS for s in S),
           "stroke_signs": {str(s): {**d, "tokens": cnt[s]} for s, d in sorted(info.items())}, "tests": tests,
           "frequency": {"rho": float(rho_p[0]), "p": float(rho_p[1]) if rho_p[1] == rho_p[1] else 1.0, "q": float(q[-1]), "rho_short": rho("short"), "rho_long": rho("long"),
                         "short": dict(sorted(by["short"].items())), "long": dict(sorted(by["long"].items())),
                         "short_by_count": {str(k): v for k, v in sorted(by["short"].items())}, "long_by_count": {str(k): v for k, v in sorted(by["long"].items())}},
           "all_start": len(texts) / sum(cnt.values()), "all_end": len(texts) / sum(cnt.values())}  # one start and one end token per text
    res["reading"] = reading(res)
    res["settings"]["runtime_min"] = (time.time() - start) / 60
    (out / "numerals.json").write_text(json.dumps(res, indent=1))
    figure(res, out / "numerals.png")
    write_report(res, out)
    print("\n".join(res["reading"]))
    print(f"{len(info)} stroke signs; done in {(time.time() - start) / 60:.1f} min -> {out / 'numerals_report.md'}")
