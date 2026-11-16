"""Milestone 2 extension: how much of an object's site, type and field symbol can be read from its inscription alone?

    python sequence_analysis/predict_object.py           # about 4 minutes on a laptop CPU
    python sequence_analysis/predict_object.py --quick   # a fast check, writes to reports/quick

Exploratory. Milestone 2 found differences by site, object type and field symbol; this measures how much information is
there. A logistic regression on sign and sign-pair counts (plus text length) is scored by 5-fold cross-validation grouped
by text, so copies of one inscription never sit on both sides, against a majority baseline, a text-length baseline, and
labels permuted within the training folds.

Writes reports/milestone_2/ (predict_object.json, predict_object.png, predict_object_report.md): aggregates and sign numbers only.
"""
import argparse
import json
import os
import time
import warnings
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")  # before numpy: BLAS threads oversubscribe the cores on small models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from corpus import ROOT, ebuds_rows, load_lines
from run_milestone2 import icon_group

warnings.filterwarnings("ignore", category=ConvergenceWarning)
OUT = ROOT / "reports/milestone_2"
MIN_ROWS = 30
TARGETS = {"site": lambda d: d.site, "object type": lambda d: d.object_type, "field symbol": lambda d: d.fs_description.map(icon_group)}
MODELS = {"signs and pairs": ((1, 2), True), "signs only": ((1, 1), True), "text length only": ((1, 1), False)}


def doc(signs, with_signs=True):
    return " ".join([f"LEN{len(signs)}"] + ([str(s) for s in signs] if with_signs else []))


def make_model(ngram):
    return CountVectorizer(token_pattern=r"\S+", ngram_range=ngram, lowercase=False, min_df=3), LogisticRegression(C=0.3, max_iter=300, class_weight="balanced")


def cross_validate(docs, y, groups, ngram, seed, rng=None):
    """5-fold grouped CV; with rng, the labels of each training fold are permuted before fitting. (balanced accuracy, macro F1, truth, prediction)."""
    truth, pred = [], []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(docs, y, groups):
        vec, clf = make_model(ngram)
        ytr = y[tr] if rng is None else rng.permutation(y[tr])
        clf.fit(vec.fit_transform(docs[tr]), ytr)
        truth += list(y[te])
        pred += list(clf.predict(vec.transform(docs[te])))
    return balanced_accuracy_score(truth, pred), f1_score(truth, pred, average="macro"), truth, pred


def prepare(rows, label):
    y = label(rows)
    keep = y.map(y.value_counts()) >= MIN_ROWS
    return rows[keep], y[keep].to_numpy(), {k: int(v) for k, v in y.value_counts().items() if v < MIN_ROWS}


def top_signs(docs, y, k=10):
    vec, clf = make_model((1, 1))
    clf.fit(vec.fit_transform(docs), y)
    names = np.array(vec.get_feature_names_out())
    sign = np.array([n.isdigit() for n in names])
    coef = clf.coef_ if len(clf.classes_) > 2 else np.vstack([-clf.coef_, clf.coef_])
    return {str(c): [int(n) for n in names[sign][np.argsort(-abs(coef[i][sign]))[:k]]] for i, c in enumerate(clf.classes_)}


def write_report(res, out):
    s = res["settings"]
    L = ["# Does the text predict the object? (exploratory)", "",
         f"Milestone 2 found differences in the sign sequences by site, object type and field symbol. This asks how much can be predicted. Rows: the {res['rows']:,} inscribed lines of the filtered corpus "
         "(damaged, doubtful and multi-line texts removed, copies kept because each is its own object), with their metadata. Features: counts of signs and of adjacent sign pairs, plus the text length. "
         f"Model: logistic regression (C 0.3) with balanced class weights; features seen in fewer than 3 training texts are ignored. Score: 5-fold cross-validation grouped by text (copies of one inscription never straddle a fold), mean over {s['repeats']} repeats of the fold split. "
         f"Metric: balanced accuracy (mean recall per class; chance is 1 over the number of classes). Permuted baseline: labels shuffled within the training folds, {s['permutations']} times (so p cannot fall below {1 / (s['permutations'] + 1):.3f}). Classes with fewer than {MIN_ROWS} rows are left out.", "",
         "| Target | Classes | Rows | Chance | Majority | Text length only | Signs only | Signs and pairs | Macro F1 | Permuted labels (mean, max) | p |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, r in res["targets"].items():
        m = r["models"]
        L.append(f"| {t} | {r['classes']} | {r['rows']:,} | {100 / r['classes']:.0f}% | {100 * r['majority']:.0f}% | {100 * m['text length only']['bal_acc']:.0f}% | {100 * m['signs only']['bal_acc']:.0f}% | "
                 f"{100 * m['signs and pairs']['bal_acc']:.0f}% ± {100 * m['signs and pairs']['bal_acc_sd']:.1f} | {100 * m['signs and pairs']['macro_f1']:.0f}% | {100 * r['permuted']['mean']:.0f}%, {100 * r['permuted']['max']:.0f}% | {r['permuted']['p']:.3f} |")
    L += ["", "Classes left out for having fewer than 30 rows: " + "; ".join(f"{t}: {', '.join(f'{k} ({v})' for k, v in r['dropped'].items()) or 'none'}" for t, r in res["targets"].items()) + ".", "",
          "## Confounds", "", "Site, object type and field symbol are correlated with each other. Balanced accuracy of the signs-and-pairs model within the largest class of each other variable:", "",
          "| Target | Within | Rows | Classes | Balanced accuracy | Chance |", "|---|---|---|---|---|---|"]
    for c in res["confounds"]:
        L.append(f"| {c['target']} | {c['within']} = {c['value']} | {c['rows']:,} | {c['classes']} | {100 * c['bal_acc']:.0f}% | {100 / c['classes']:.0f}% |")
    L += ["", "![confusion matrices](predict_object.png)", "", "## Signs with the largest weights", "", "Sign numbers, by absolute weight in a signs-only model fitted on all rows (not a causal statement; the weights of rare signs are noisy):", ""]
    for t, r in res["targets"].items():
        L.append(f"- {t}: " + "; ".join(f"{c}: {' '.join(map(str, v))}" for c, v in r["top_signs"].items()))
    L += ["", "## Reading", "", res["reading"], "", "## Limits", "",
          "Classes are very unbalanced and the metadata categories are Mahadevan's. Texts are short (median 4 signs), so a ceiling far below 100% is expected. "
          "Pairs of signs help only if order carries information beyond counts. Repeats of the fold split are not independent resamples of the data. "
          f"Settings: seed {s['seed']}, runtime {s['runtime_min']:.1f} min, quick run: {s['quick']}.", ""]
    (out / "predict_object_report.md").write_text("\n".join(L))


def reading(res):
    out = []
    for t, r in res["targets"].items():
        a, m = r["models"]["signs and pairs"]["bal_acc"], r["models"]["signs only"]["bal_acc"]
        out.append(f"{t}: {100 * a:.0f}% balanced accuracy against {100 / r['classes']:.0f}% chance and {100 * r['permuted']['max']:.0f}% at most for permuted labels "
                   f"(p {r['permuted']['p']:.3f}); text length alone gives {100 * r['models']['text length only']['bal_acc']:.0f}%, and adding pairs to single signs changes it by {100 * (a - m):+.1f} points.")
    return " ".join(out)


def selfcheck():
    rng = np.random.default_rng(0)
    texts = [tuple(int(x) for x in rng.integers(10, 60, 4)) + (i % 2,) for i in range(200)]  # sign 0 or 1 decides the label
    texts = [t + (k,) for k, t in enumerate(texts)]
    y = np.array([t[4] for t in texts])
    docs = np.array([doc(t[:4] + (t[4],)) for t in texts])
    groups = np.arange(len(texts))
    assert cross_validate(docs, y, groups, (1, 1), 0)[0] > 0.95
    assert cross_validate(docs, y, groups, (1, 1), 0, rng=np.random.default_rng(1))[0] < 0.7
    g = np.array([i // 2 for i in range(len(texts))])  # every text twice: a group never straddles two folds
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(docs, y, g):
        assert not set(g[tr]) & set(g[te])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    start = time.time()
    selfcheck()
    repeats, perms, seed = (2, 5, 20261116) if args.quick else (3, 30, 20261116)
    rng = np.random.default_rng(seed)
    rows = ebuds_rows(load_lines())
    res = {"rows": len(rows), "targets": {}, "confounds": []}
    label = {t: f(rows).reset_index(drop=True) for t, f in TARGETS.items()}
    rows = rows.reset_index(drop=True)
    for t, f in TARGETS.items():
        sub, y, dropped = prepare(rows, f)
        docs = np.array([doc(s) for s in sub.signs])
        groups = np.array([" ".join(map(str, s)) for s in sub.signs])
        r = {"classes": len(set(y)), "rows": len(y), "dropped": dropped, "majority": float(balanced_accuracy_score(y, [Counter(y).most_common(1)[0][0]] * len(y))), "models": {}}
        for name, (ng, with_signs) in MODELS.items():
            d = docs if with_signs else np.array([doc(s, False) for s in sub.signs])
            runs = [cross_validate(d, y, groups, ng, seed + k) for k in range(repeats)]
            r["models"][name] = {"bal_acc": float(np.mean([x[0] for x in runs])), "bal_acc_sd": float(np.std([x[0] for x in runs])), "macro_f1": float(np.mean([x[1] for x in runs]))}
            if name == "signs and pairs":
                truth, pred = runs[0][2], runs[0][3]
        obs = cross_validate(docs, y, groups, (1, 2), seed)[0]
        null = [cross_validate(docs, y, groups, (1, 2), seed, rng=rng)[0] for _ in range(perms)]
        r["permuted"] = {"mean": float(np.mean(null)), "max": float(np.max(null)), "p": float((1 + sum(x >= obs for x in null)) / (perms + 1))}
        r["top_signs"] = top_signs(docs, y)
        classes = sorted(set(y))
        r["confusion"] = {"classes": classes, "matrix": confusion_matrix(truth, pred, labels=classes, normalize="true").round(3).tolist()}
        res["targets"][t] = r
        for other, f2 in TARGETS.items():
            if other == t:
                continue
            big = f2(sub).value_counts().index[0]
            part = sub[f2(sub).to_numpy() == big]
            yy = f(part).to_numpy()
            keep = np.array([c >= MIN_ROWS for c in map(Counter(yy).get, yy)])
            if len(set(yy[keep])) > 1:
                ds = np.array([doc(s) for s in part.signs])[keep]
                gs = np.array([" ".join(map(str, s)) for s in part.signs])[keep]
                res["confounds"].append({"target": t, "within": other, "value": str(big), "rows": int(keep.sum()), "classes": len(set(yy[keep])),
                                         "bal_acc": float(np.mean([cross_validate(ds, yy[keep], gs, (1, 2), seed + k)[0] for k in range(2)]))})
        print(t, {k: round(v["bal_acc"], 3) for k, v in r["models"].items()}, flush=True)
    res["reading"] = reading(res)
    res["settings"] = {"repeats": repeats, "permutations": perms, "seed": seed, "quick": args.quick, "runtime_min": (time.time() - start) / 60}
    out = ROOT / "reports/quick" if args.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "predict_object.json").write_text(json.dumps(res, indent=1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (t, r) in zip(axes, res["targets"].items()):
        ax.imshow(r["confusion"]["matrix"], cmap="Blues", vmin=0, vmax=1)
        n = len(r["confusion"]["classes"])
        ax.set_xticks(range(n), r["confusion"]["classes"], rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(n), r["confusion"]["classes"], fontsize=7)
        ax.set_title(f"{t} (rows: true class, share predicted)", fontsize=9)
    plt.tight_layout()
    plt.savefig(out / "predict_object.png", dpi=120)
    plt.close()
    write_report(res, out)
    print(res["reading"])
    print(f"done in {(time.time() - start) / 60:.1f} min -> {out / 'predict_object_report.md'}")
