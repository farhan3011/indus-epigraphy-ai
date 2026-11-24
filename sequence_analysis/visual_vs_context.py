"""Milestones 1 and 2 together: do signs that look alike behave alike?

    python sequence_analysis/visual_vs_context.py           # a few minutes on a laptop
    python sequence_analysis/visual_vs_context.py --quick   # a fast check, writes to reports/quick

Exploratory. If a sign list splits one sign into several drawn variants, the variants should look alike and also occur in the
same contexts. Visual similarity: cosine of DINOv2 embeddings of the clean IM-417 drawings (one per sign), with raw pixels as a
baseline. Context similarity: 1 minus the mean Jensen-Shannon divergence of three distributions (the sign before, the sign after,
the position in the text). The two are compared with a Mantel test (sign labels permuted).

Writes reports/milestone_2/ (visual_vs_context.json, visual_vs_context.png, visual_vs_context_report.md): sign numbers and
aggregates only.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

import stats
from corpus import ROOT, ebuds, load_lines

sys.path.insert(0, str(ROOT / "grapheme_extraction"))
import synthetic_generator as gen  # noqa: E402

OUT = ROOT / "reports/milestone_2"
MIN_TOKENS = 20
SENSITIVITY = (10, 50)
POS_BINS = 5
TOP_PAIRS = 50
CANDIDATES = 20


# ---------- similarities ----------

def contexts(texts, signs):
    """Per sign: counts of the sign before (START included), the sign after (END included) and the position bin."""
    idx = {s: i for i, s in enumerate(sorted({s for t in texts for s in t}))}
    n = len(idx) + 1
    prev, nxt, pos = defaultdict(lambda: np.zeros(n)), defaultdict(lambda: np.zeros(n)), defaultdict(lambda: np.zeros(POS_BINS))
    for t in texts:
        for i, s in enumerate(t):
            if s not in signs:
                continue
            prev[s][idx[t[i - 1]] if i else n - 1] += 1
            nxt[s][idx[t[i + 1]] if i + 1 < len(t) else n - 1] += 1
            pos[s][min(POS_BINS - 1, int((i + 0.5) / len(t) * POS_BINS))] += 1
    return {s: (prev[s], nxt[s], pos[s]) for s in signs}


def context_similarity(ctx, signs):
    k = len(signs)
    M = np.ones((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            M[a, b] = M[b, a] = 1 - np.mean([stats.js_divergence(x, y) for x, y in zip(ctx[signs[a]], ctx[signs[b]])])
    return M


def cosine(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def drawings(signs):
    """Clean 150x150 drawings, black on white."""
    return np.stack([255 - cv2.imread(str(gen.GLYPH_DIR / f"{s}.png"), cv2.IMREAD_GRAYSCALE) for s in signs])


def dino_similarity(signs):
    from classifier import embed, load_backbone, to_tensor
    imgs = drawings(signs)
    emb = embed(load_backbone(), to_tensor([cv2.resize(g, (224, 224), interpolation=cv2.INTER_AREA) for g in imgs])).numpy()
    return cosine(emb)


def pixel_similarity(signs):
    return cosine(drawings(signs).reshape(len(signs), -1).astype(float) / 255)


# ---------- tests ----------

def upper(M):
    return M[np.triu_indices(len(M), 1)]


def mantel(A, B, perms, rng):
    """Spearman correlation of the two upper triangles, and its p-value with the sign labels of B permuted."""
    ra = rankdata(upper(A))
    ra = (ra - ra.mean()) / ra.std()
    RB = np.zeros_like(B)
    RB[np.triu_indices(len(B), 1)] = rankdata(upper(B))
    RB = RB + RB.T
    def rho(P):
        rb = upper(P)
        return float(np.mean(ra * (rb - rb.mean()) / rb.std()))
    obs = rho(RB)
    null = []
    for _ in range(perms):
        p = rng.permutation(len(B))
        null.append(rho(RB[np.ix_(p, p)]))
    return obs, float((1 + np.sum(np.array(null) >= obs)) / (perms + 1))


def top_pairs_test(V, C, rng, draws):
    """Mean context similarity of the most visually similar pairs, against random sets of pairs."""
    v, c = upper(V), upper(C)
    top = np.argsort(-v)[:TOP_PAIRS]
    obs = float(c[top].mean())
    null = np.array([c[rng.choice(len(c), TOP_PAIRS, replace=False)].mean() for _ in range(draws)])
    return {"observed": obs, "null_mean": float(null.mean()), "null_sd": float(null.std()), "p": float((1 + np.sum(null >= obs)) / (draws + 1))}


def candidates(V, C, signs):
    """Pairs in the top 5% of both similarities, best first by the sum of percentiles."""
    iu = np.triu_indices(len(V), 1)
    v, c = V[iu], C[iu]
    pv, pc = rankdata(v) / len(v), rankdata(c) / len(c)
    keep = np.flatnonzero((pv >= 0.95) & (pc >= 0.95))
    keep = keep[np.argsort(-(pv[keep] + pc[keep]))][:CANDIDATES]
    return [{"signs": [signs[iu[0][k]], signs[iu[1][k]]], "visual": round(float(v[k]), 3), "context": round(float(c[k]), 3)} for k in keep]


def analyse(texts, cnt, threshold, perms, draws, rng, with_details):
    signs = sorted(s for s in cnt if cnt[s] >= threshold)
    C = context_similarity(contexts(texts, set(signs)), signs)
    V, P = dino_similarity(signs), pixel_similarity(signs)
    rho, p = mantel(V, C, perms, rng)
    rho_px, p_px = mantel(P, C, perms, rng)
    out = {"signs": len(signs), "pairs": len(signs) * (len(signs) - 1) // 2, "dino_spearman": rho, "dino_p": p, "pixel_spearman": rho_px, "pixel_p": p_px}
    if with_details:
        out["top_pairs"] = top_pairs_test(V, C, rng, draws)
        out["candidates"] = candidates(V, C, signs)
        out["_scatter"] = (upper(V), upper(C))
    return out


# ---------- report ----------

def figure(res, path):
    v, c = res["main"].pop("_scatter")
    plt.figure(figsize=(5.5, 4.2))
    plt.scatter(v, c, s=3, color="0.6", alpha=0.4, label="sign pairs")
    for cand in res["main"]["candidates"]:
        plt.scatter(cand["visual"], cand["context"], s=18, color="#c9663c")
    plt.scatter([], [], s=18, color="#c9663c", label="candidate variant pairs")
    plt.xlabel("visual similarity (DINOv2 cosine of the drawings)"); plt.ylabel("context similarity (1 - mean JS divergence)")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


def write_report(res, out):
    m, tp, s = res["main"], res["main"]["top_pairs"], res["settings"]
    lines = [
        "# Do signs that look alike behave alike? (exploratory)", "",
        f"Signs with at least {MIN_TOKENS} tokens in the {res['texts']:,} EBUDS texts: {m['signs']} signs, {m['pairs']:,} pairs. Visual similarity is the "
        "cosine of DINOv2-small embeddings of the clean IM-417 drawings (one per sign); a raw-pixel cosine is the baseline. Context "
        "similarity is 1 minus the mean Jensen-Shannon divergence of three distributions: the sign before (text start included), the "
        f"sign after (text end included) and the relative position in the text ({POS_BINS} bins).", "",
        "| Measure | Spearman correlation with context similarity | Mantel p (sign labels permuted) |", "|---|---|---|",
        f"| DINOv2 visual similarity | {m['dino_spearman']:+.3f} | {m['dino_p']:.4f} |",
        f"| Pixel similarity | {m['pixel_spearman']:+.3f} | {m['pixel_p']:.4f} |", "",
        f"The {TOP_PAIRS} most visually similar pairs have a mean context similarity of {tp['observed']:.3f}, against {tp['null_mean']:.3f} ± "
        f"{tp['null_sd']:.3f} for random sets of {TOP_PAIRS} pairs (p = {tp['p']:.4f}).", "",
        "Sensitivity to the token threshold:", "",
        "| Minimum tokens | Signs | DINOv2 Spearman | p | Pixel Spearman | p |", "|---|---|---|---|---|---|",
        *[f"| {k} | {r['signs']} | {r['dino_spearman']:+.3f} | {r['dino_p']:.4f} | {r['pixel_spearman']:+.3f} | {r['pixel_p']:.4f} |"
          for k, r in sorted(res["sensitivity"].items(), key=lambda kv: int(kv[0]))], "",
        f"## Candidate variant pairs ({len(m['candidates'])} pairs in the top 5% on both)", "",
        "| Signs (Mahadevan) | Visual | Context |", "|---|---|---|",
        *[f"| {a} and {b} | {c['visual']:.3f} | {c['context']:.3f} |" for c in m["candidates"] for a, b in [c["signs"]]], "",
        "## Reading", "", *[f"- {x}" for x in res["reading"]], "",
        "## Limits", "",
        "- Compound signs contain other signs, so they look alike for a reason that has nothing to do with variation.",
        "- The drawings are normalised sign-list forms, not carved forms.",
        "- A correlation near zero would mean visual form does not predict use, which is also a result. A positive one does not show that two "
        "signs are variants: signs of a class (for example the stroke signs) can look and behave alike and still be distinct.", "",
        f"Settings: {s['perms']:,} label permutations, {s['draws']:,} random pair sets, seed {s['seed']}; run time {s['runtime_min']:.1f} minutes.", ""]
    (out / "visual_vs_context_report.md").write_text("\n".join(lines))


def reading(res):
    m, tp = res["main"], res["main"]["top_pairs"]
    sig = lambda p: "a reliable" if p < 0.05 else "no reliable"
    return [f"There is {sig(m['dino_p'])} association between how alike two signs look (DINOv2) and how alike they are used "
            f"(Spearman {m['dino_spearman']:+.3f}, Mantel p = {m['dino_p']:.4f}); raw pixels give {m['pixel_spearman']:+.3f} (p = {m['pixel_p']:.4f}).",
            f"The most similar-looking pairs are {'more' if tp['observed'] > tp['null_mean'] else 'not more'} alike in use than random pairs "
            f"(p = {tp['p']:.4f}).",
            "The candidate pairs are where to look for sign-list variants; each needs checking by eye against carved examples."]


def selfcheck():
    texts = [(1, 2, 3), (1, 2, 3), (4, 2, 5), (6, 7, 8)]
    ctx = contexts(texts, {1, 2, 4, 6, 7})
    C = context_similarity(ctx, [1, 4, 6, 7])
    assert np.allclose(np.diag(C), 1)
    toy = [(1, 9), (2, 9)]  # 1 and 2: same position, same follower, same (start) predecessor
    assert abs(context_similarity(contexts(toy, {1, 2}), [1, 2])[0, 1] - 1) < 1e-12
    assert C[0, 3] < 0.5  # signs 1 and 7 share no neighbour and no position
    rng = np.random.default_rng(0)
    A = rng.random((6, 6)); A = A + A.T
    p = rng.permutation(6)
    assert np.allclose(A[np.ix_(p, p)], A[np.ix_(p, p)].T)
    r, _ = mantel(A, A, 50, rng)
    assert abs(r - 1) < 1e-9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    perms, draws, seed = (500, 500, 20261124) if args.quick else (10000, 10000, 20261124)
    rng = np.random.default_rng(seed)
    texts = ebuds(load_lines())[0]
    cnt = Counter(s for t in texts for s in t)
    out = Path(ROOT / "reports/quick") if args.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    res = {"texts": len(texts), "main": analyse(texts, cnt, MIN_TOKENS, perms, draws, rng, True),
           "sensitivity": {str(k): analyse(texts, cnt, k, perms, draws, rng, False) for k in SENSITIVITY},
           "settings": {"perms": perms, "draws": draws, "seed": seed, "quick": args.quick, "min_tokens": MIN_TOKENS}}
    figure(res, out / "visual_vs_context.png")
    res["reading"] = reading(res)
    res["settings"]["runtime_min"] = (time.time() - start) / 60
    (out / "visual_vs_context.json").write_text(json.dumps(res, indent=1))
    write_report(res, out)
    print("\n".join(res["reading"]))
    print(f"done in {res['settings']['runtime_min']:.1f} min -> {out / 'visual_vs_context_report.md'}")
