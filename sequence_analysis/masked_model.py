"""Milestone 3: predict a hidden sign from the rest of its inscription with a small masked-sign transformer,
compared with the Milestone 2 bigram on exactly the same folds and trials.

    python sequence_analysis/masked_model.py                       # evaluation + report; ~15 min on a laptop CPU
    python sequence_analysis/masked_model.py --quick               # 1 seed, short training, for a fast check
    python sequence_analysis/masked_model.py predict "267 99 ? 342"   # fill a gap, reading order, '?' = missing

Writes reports/milestone_3/ (results.json, figures, milestone_3_report.md). Only aggregates go there:
per-text predictions on damaged corpus texts go to data/derived/, which is gitignored.
"""
import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import ngram
from corpus import ROOT, ebuds, load_lines
from ngram import BOS, EOS, WittenBell

OUT = ROOT / "reports/milestone_3"
DERIVED = ROOT / "data/derived"
MODEL_FILE = ROOT / "models/masked_sign_model.pt"
M2_RESULTS = ROOT / "reports/milestone_2/results.json"
PAD, MASK, START, END, UNK = range(5)  # token ids; signs follow
MAX_LEN = 16  # longest EBUDS text is 14 signs, plus start and end
CONFIG = {"d": 96, "layers": 2, "heads": 4, "dropout": 0.2, "lr": 2e-3, "weight_decay": 0.01, "batch": 64,
          "max_epochs": 400, "patience": 40, "inner_val_share": 0.1}
TEMPERATURES = np.round(np.arange(0.5, 3.01, 0.05), 2)


# ---------- models ----------

class MaskedSignModel(nn.Module):
    """Transformer encoder over [start, signs..., end]; predicts the sign at each masked position."""

    def __init__(self, n_tokens, n_out, d, layers, heads, dropout):
        super().__init__()
        self.tok, self.pos = nn.Embedding(n_tokens, d), nn.Embedding(MAX_LEN, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc, self.norm, self.out = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False), nn.LayerNorm(d), nn.Linear(d, n_out)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1]))
        return self.out(self.norm(self.enc(h, src_key_padding_mask=x == PAD)))


class Tokens:
    """Signs seen in training get their own token; anything else in a context becomes UNK.
    Outputs always range over the full candidate vocabulary, so every fold scores the same candidates."""

    def __init__(self, train, vocab):
        self.vocab, self.out = vocab, {s: i for i, s in enumerate(vocab)}
        self.tok = {s: 5 + i for i, s in enumerate(sorted({s for t in train for s in t}))}
        self.n = 5 + len(self.tok)

    def encode(self, text, gaps=()):
        ids = [START] + [MASK if i in gaps else self.tok.get(s, UNK) for i, s in enumerate(text)] + [END]
        return ids + [PAD] * (MAX_LEN - len(ids))


def batches(tokens, texts, rng, size):
    """One random masked position per text per epoch, matching the one-gap test."""
    order = rng.permutation(len(texts))
    for i in range(0, len(order), size):
        chunk = [texts[k] for k in order[i:i + size]]
        gaps = [int(rng.integers(len(t))) for t in chunk]
        x = torch.tensor([tokens.encode(t, {g}) for t, g in zip(chunk, gaps)])
        y = torch.tensor([tokens.out[t[g]] for t, g in zip(chunk, gaps)])
        yield x, torch.tensor(gaps) + 1, y


def trials(texts):
    """Every position of every text: (text index, position)."""
    return [(k, j) for k, t in enumerate(texts) for j in range(len(t))]


@torch.no_grad()
def net_logits(model, tokens, texts, pairs):
    model.eval()
    x = torch.tensor([tokens.encode(texts[k], {j}) for k, j in pairs])
    out = torch.cat([model(x[i:i + 512])[torch.arange(len(x[i:i + 512])), torch.tensor([j + 1 for _, j in pairs[i:i + 512]])]
                     for i in range(0, len(x), 512)])
    return out.numpy()


def train_net(train, val, vocab, seed, cfg=CONFIG):
    """Early stopping on the masked-sign log loss of the inner validation texts (every position)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokens = Tokens(train, vocab)
    model = MaskedSignModel(tokens.n, len(vocab), cfg["d"], cfg["layers"], cfg["heads"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    val_pairs = trials(val)
    val_y = np.array([tokens.out[val[k][j]] for k, j in val_pairs])
    best, best_state, since = math.inf, None, 0
    for epoch in range(cfg["max_epochs"]):
        model.train()
        for x, pos, y in batches(tokens, train, rng, cfg["batch"]):
            loss = nn.functional.cross_entropy(model(x)[torch.arange(len(y)), pos], y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if not val:
            continue
        logits = net_logits(model, tokens, val, val_pairs)
        nll = float(-log_softmax(logits)[np.arange(len(val_y)), val_y].mean())
        if nll < best - 1e-4:
            best, best_state, since = nll, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            since += 1
            if since >= cfg["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, tokens, epoch + 1


def bigram_scores(model, vocab, texts, pairs):
    """Unnormalised P(w | prev) * P(next | w) for every candidate w, the Milestone 2 restoration score."""
    rows = []
    for k, j in pairs:
        s = [BOS, *texts[k], EOS]
        rows.append([model.prob(w, (s[j],)) * model.prob(s[j + 2], (w,)) for w in vocab])
    return np.log(np.array(rows))


def unigram_scores(train, vocab, pairs):
    c = Counter(s for t in train for s in t)
    p = np.array([c[w] + 0.5 for w in vocab], float)
    return np.tile(np.log(p / p.sum()), (len(pairs), 1))


# ---------- scoring ----------

def log_softmax(z, temperature=1.0):
    z = z / temperature
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def fit_temperature(scores, y):
    """Temperature with the lowest log loss on held-out trials."""
    return float(min(TEMPERATURES, key=lambda T: -log_softmax(scores, T)[np.arange(len(y)), y].mean()))


def ece(conf, correct, bins=15):
    """Expected calibration error of the top-1 confidence, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def per_trial(logp, y):
    """Per-trial outcomes from log-probabilities over the candidates."""
    p = np.exp(logp)
    order = np.argsort(-p, 1)
    rank = np.argmax(order == y[:, None], 1)
    k90 = np.argmax(np.cumsum(np.take_along_axis(p, order, 1), 1) >= 0.9, 1) + 1
    return {"hit1": rank == 0, "hit5": rank < 5, "nll_bits": -logp[np.arange(len(y)), y] / math.log(2),
            "in90": rank < k90, "in90_strict": rank < k90 - 1, "set90": k90, "conf": p.max(1)}


def summary(t):
    return {"top1": float(t["hit1"].mean()), "top5": float(t["hit5"].mean()), "bits_per_sign": float(t["nll_bits"].mean()),
            "hit_in_90pct_set": float(t["in90"].mean()), "hit_in_90pct_set_strict": float(t["in90_strict"].mean()),
            "mean_90pct_set_size": float(t["set90"].mean()), "ece_top1": ece(t["conf"], t["hit1"])}


def paired_bootstrap(a, b, groups, n=2000, seed=0):
    """Difference in mean a - b, resampling whole test texts. Returns (diff, low, high)."""
    ids = np.unique(groups)
    sa, sb, cnt = (np.bincount(np.searchsorted(ids, groups), weights=v, minlength=len(ids)) for v in (a, b, np.ones(len(a))))
    pick = np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))
    d = (sa[pick].sum(1) - sb[pick].sum(1)) / cnt[pick].sum(1)
    return float(a.mean() - b.mean()), *np.percentile(d, [2.5, 97.5]).tolist()


# ---------- evaluation ----------

def evaluate(texts, vocab, seeds, cfg):
    """5-fold CV over distinct texts, the Milestone 2 folds. Temperatures and early stopping use an inner split
    of each training fold; the test fold is only scored."""
    idx = {s: i for i, s in enumerate(vocab)}
    rows = {name: [] for name in ("unigram", "bigram", "bigram_calibrated", *(f"net_seed{s}" for s in seeds),
                                  *(f"net_seed{s}_calibrated" for s in seeds), *(f"average_seed{s}" for s in seeds))}
    ys, groups, epochs, temps = [], [], [], {"bigram": [], "net": []}
    offset = 0
    for f, (train, test) in enumerate(ngram.folds(texts)):
        rng = random.Random(f)
        inner = rng.sample(range(len(train)), round(CONFIG["inner_val_share"] * len(train)))
        inner_set = set(inner)
        fit, val = [t for i, t in enumerate(train) if i not in inner_set], [train[i] for i in inner]
        pairs, val_pairs = trials(test), trials(val)
        y = np.array([idx[test[k][j]] for k, j in pairs])
        val_y = np.array([idx[val[k][j]] for k, j in val_pairs])
        ys.append(y)
        groups.append(np.array([offset + k for k, _ in pairs]))
        offset += len(test)

        bigram = bigram_scores(WittenBell(train, vocab, 2, backoff=True), vocab, test, pairs)
        t_bi = fit_temperature(bigram_scores(WittenBell(fit, vocab, 2, backoff=True), vocab, val, val_pairs), val_y)
        temps["bigram"].append(t_bi)
        rows["unigram"].append(log_softmax(unigram_scores(train, vocab, pairs)))
        rows["bigram"].append(log_softmax(bigram))
        rows["bigram_calibrated"].append(log_softmax(bigram, t_bi))
        for s in seeds:
            model, tokens, n_epochs = train_net(fit, val, vocab, seed=1000 * s + f, cfg=cfg)
            epochs.append(n_epochs)
            t_net = fit_temperature(net_logits(model, tokens, val, val_pairs), val_y)
            temps["net"].append(t_net)
            net = net_logits(model, tokens, test, pairs)
            rows[f"net_seed{s}"].append(log_softmax(net))
            rows[f"net_seed{s}_calibrated"].append(log_softmax(net, t_net))
            avg = np.log((np.exp(log_softmax(net, t_net)) + np.exp(log_softmax(bigram, t_bi))) / 2)
            rows[f"average_seed{s}"].append(avg)
        print(f"fold {f}: {len(test)} texts, {len(pairs)} trials, epochs {epochs[-len(seeds):]}", flush=True)
    y, groups = np.concatenate(ys), np.concatenate(groups)
    outcomes = {name: per_trial(np.concatenate(v), y) for name, v in rows.items()}
    return outcomes, groups, epochs, temps


def collapse(outcomes, seeds):
    """Mean and sd over seeds for the network rows; single values for the rest."""
    res = {}
    for base in ("unigram", "bigram", "bigram_calibrated"):
        res[base] = summary(outcomes[base])
    for base in ("net", "net_calibrated", "average"):
        names = [f"{base.split('_')[0]}_seed{s}" + ("_calibrated" if base == "net_calibrated" else "") for s in seeds]
        per = [summary(outcomes[n]) for n in names]
        res[base] = {k: float(np.mean([p[k] for p in per])) for k in per[0]} | {f"{k}_sd": float(np.std([p[k] for p in per])) for k in per[0]}
    return res


def reliability(outcomes, names, path):
    plt.figure(figsize=(5, 4.6))
    plt.plot([0, 1], [0, 1], color="0.6", lw=1, ls=":", label="perfect calibration")
    edges = np.linspace(0, 1, 11)
    for name, label, colour in names:
        t = outcomes[name]
        idx = np.clip(np.digitize(t["conf"], edges[1:-1]), 0, 9)
        xs = [t["conf"][idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ys = [t["hit1"][idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        plt.plot(xs, ys, marker="o", color=colour, label=f"{label} (ECE {ece(t['conf'], t['hit1']):.3f})")
    plt.xlabel("confidence in the top guess"); plt.ylabel("share of top guesses that are right")
    plt.legend(fontsize=7); plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()


# ---------- final model and its uses ----------

def train_final(texts, vocab, seed=0):
    """Trained on every EBUDS text (10% held out for early stopping and the temperature)."""
    rng = random.Random(seed)
    inner = set(rng.sample(range(len(texts)), round(CONFIG["inner_val_share"] * len(texts))))
    fit, val = [t for i, t in enumerate(texts) if i not in inner], [texts[i] for i in sorted(inner)]
    model, tokens, _ = train_net(fit, val, vocab, seed)
    pairs = trials(val)
    temperature = fit_temperature(net_logits(model, tokens, val, pairs), np.array([tokens.out[val[k][j]] for k, j in pairs]))
    return model, tokens, temperature


def fill(model, tokens, temperature, text, k=5):
    """text: signs in reading order with None for each gap. All gaps are masked together and each is
    predicted on its own (no joint search). Returns {gap position: [(sign, probability), ...]}."""
    gaps = {i for i, s in enumerate(text) if s is None}
    x = torch.tensor([tokens.encode([s or 0 for s in text], gaps)])
    model.eval()
    with torch.no_grad():
        logits = model(x)[0].numpy()
    out = {}
    for g in sorted(gaps):
        p = np.exp(log_softmax(logits[g + 1][None], temperature)[0])
        top = np.argsort(-p)[:k]
        out[g] = [(tokens.vocab[i], float(p[i])) for i in top]
    return out


def save(model, tokens, temperature):
    MODEL_FILE.parent.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "vocab": tokens.vocab, "tok": tokens.tok, "temperature": temperature, "config": CONFIG}, MODEL_FILE)


def load():
    d = torch.load(MODEL_FILE, weights_only=False)
    tokens = Tokens([], d["vocab"])
    tokens.tok, tokens.n = d["tok"], 5 + len(d["tok"])
    c = d["config"]
    model = MaskedSignModel(tokens.n, len(tokens.vocab), c["d"], c["layers"], c["heads"], c["dropout"])
    model.load_state_dict(d["state"])
    return model, tokens, d["temperature"]


def damaged_texts(lines, vocab):
    """Single-line, no doubtful reading, exactly one illegible sign, every other sign in the vocabulary; distinct."""
    keep = lines[(lines.line == "0") & ~lines.doubtful & lines.signs.map(lambda s: s.count(0) == 1 and 2 <= len(s) <= MAX_LEN - 2)]
    v = set(vocab)
    return sorted({s for s in keep.signs if all(x in v for x in s if x)})


# ---------- report ----------

def pct(x):
    return f"{100 * x:.1f}%"


def write_report(res):
    m2 = res["milestone_2_restoration"]
    r, s = res["models"], res["settings"]
    def row(name, label, sd=False):
        v = r[name]
        f = (lambda k: f"{pct(v[k])} ± {100 * v[k + '_sd']:.1f}") if sd else (lambda k: pct(v[k]))
        return (f"| {label} | {f('top1')} | {f('top5')} | {v['bits_per_sign']:.2f} | {pct(v['hit_in_90pct_set'])} "
                f"({v['mean_90pct_set_size']:.0f} signs) | {v['ece_top1']:.3f} |")
    d1, d2 = res["difference_top1_vs_bigram"], res["difference_top1_average_vs_bigram"]
    dm = res["damaged_texts"]
    lines = [
        "# Milestone 3 report: predicting a hidden sign",
        "",
        f"A small masked-sign transformer ({s['parameters']:,} parameters) is trained to fill one hidden sign from the rest of "
        "its inscription, and compared with the Milestone 2 bigram on the same task. Both use the same 5 folds of the "
        f"{s['texts']:,} EBUDS texts and score the same {s['trials']:,} trials: every position of every held-out text, ranked "
        f"over the same {s['candidates']} candidate signs. Early stopping and the temperature that calibrates confidence are "
        "chosen on a 10% inner split of each training fold; the test fold is only scored. Network rows are the mean ± sd over "
        f"{len(s['seeds'])} training seeds.",
        "",
        f"The bigram reproduces Milestone 2 exactly (top-1 {pct(r['bigram']['top1'])} here, {pct(m2['top1'])} there), so the "
        "folds, candidates and trials are the same.",
        "",
        "## Results",
        "",
        "| Model | Top-1 | Top-5 | Bits per hidden sign | True sign inside the 90% set (its mean size) | Calibration error (ECE) |",
        "|---|---|---|---|---|---|",
        row("unigram", "Most frequent signs (no context)"),
        row("bigram", "Bigram, Milestone 2"),
        row("bigram_calibrated", "Bigram, temperature-calibrated"),
        row("net", "Masked transformer", sd=True),
        row("net_calibrated", "Masked transformer, temperature-calibrated", sd=True),
        row("average", "Average of calibrated bigram and transformer", sd=True),
        "",
        f"Top-1 difference from the bigram, with a 95% interval from resampling whole test texts (seed 0): transformer "
        f"{100 * d1[0]:+.1f} points [{100 * d1[1]:+.1f}, {100 * d1[2]:+.1f}]; average {100 * d2[0]:+.1f} points "
        f"[{100 * d2[1]:+.1f}, {100 * d2[2]:+.1f}].",
        "",
        "For context, Milestone 2 also ran a trigram backoff (top-1 "
        f"{pct(res['milestone_2_other']['witten-bell backoff n=3']['top1'])}) and Kneser-Ney bigram "
        f"({pct(res['milestone_2_other']['kneser-ney n=2']['top1'])}). Yadav et al. 2010 report that the true sign falls inside "
        "the 90% candidate set 74% of the time; that is the column it compares with, not top-1.",
        "",
        "## Reading the numbers",
        "",
        f"- {res['verdict']}",
        "- Calibration (the proposal's \"validation confidence ratings\"): ECE is the average gap between how confident the top "
        "guess is and how often it is right; 0 is perfect. `reliability.png` shows the same thing per confidence band.",
        "- The 90% set is the smallest group of signs whose probabilities add up to 90%. It is the useful output for an "
        "epigrapher: a short list that should contain the right sign nine times in ten.",
        "",
        "## Damaged inscriptions",
        "",
        f"The final model, trained on all {s['texts']:,} texts, was run on the {dm['texts']} distinct single-line corpus texts with "
        "exactly one illegible sign. They have no answer key, so this shows what the model offers an epigrapher, not how "
        f"accurate it is. Its top guess carries a median confidence of {pct(dm['median_top1_confidence'])}; "
        f"{dm['confident_over_50pct']} texts get a top guess above 50% and {dm['confident_over_80pct']} above 80%. The 90% "
        f"candidate set holds a median of {dm['median_90pct_set_size']:.0f} signs. The most frequent top guesses are "
        f"{', '.join(str(x) for x, _ in dm['most_common_top1'])}. Per-text suggestions are in `data/derived/` on the machine that "
        "ran this (not published: they contain corpus texts).",
        "",
        "## Use it",
        "",
        "```bash",
        "python sequence_analysis/masked_model.py predict \"267 99 ? 342\"   # reading order, ? marks each gap",
        "```",
        "",
        "Several gaps are masked together and each is predicted on its own; there is no joint search over gap combinations.",
        "",
        "## Settings",
        "",
        f"Transformer: {s['config']['layers']} layers, width {s['config']['d']}, {s['config']['heads']} heads, dropout "
        f"{s['config']['dropout']}, AdamW (learning rate {s['config']['lr']}, weight decay {s['config']['weight_decay']}), one random "
        "masked sign per text per epoch, early stopping after "
        f"{s['config']['patience']} epochs without improvement (median {s['median_epochs']} epochs). Inputs are the signs "
        "in reading order between start and end tokens; a context sign unseen in a training fold becomes an unknown token. "
        f"Temperatures chosen: bigram median {s['median_temperature_bigram']}, transformer median {s['median_temperature_net']}. "
        "Trained on CPU. The configuration was fixed in advance; no search over settings was run.",
        "",
    ]
    (OUT / "milestone_3_report.md").write_text("\n".join(lines))


# ---------- checks ----------

def selfcheck():
    assert abs(ece(np.array([0.9, 0.9, 0.2, 0.2]), np.array([1, 1, 0, 0])) - 0.15) < 1e-12
    lp = np.log(np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]]))
    t = per_trial(lp, np.array([1, 2]))
    assert list(t["hit1"]) == [False, True] and list(t["set90"]) == [3, 2] and list(t["in90"]) == [True, True]
    texts, vocab = [(1, 2, 3), (1, 2, 3), (3, 2, 1)], [1, 2, 3]
    b = np.exp(log_softmax(bigram_scores(WittenBell(texts, vocab, 2, backoff=True), vocab, texts, [(0, 1)])))
    assert abs(b.sum() - 1) < 1e-9 and b.argmax() == 1
    d = paired_bootstrap(np.array([1., 1, 0, 0]), np.array([0., 0, 0, 0]), np.array([0, 0, 1, 1]), n=200)
    assert d[0] == 0.5 and d[1] <= 0.5 <= d[2]
    # the network can learn three texts by heart
    small = [(10, 20, 30), (40, 50, 60), (70, 80)]
    model, tokens, _ = train_net(small, [], [10, 20, 30, 40, 50, 60, 70, 80], seed=0,
                                 cfg=CONFIG | {"max_epochs": 300, "dropout": 0.0, "batch": 8})
    got = [fill(model, tokens, 1.0, [None if i == j else s for i, s in enumerate(t)], k=1)[j][0][0] for t in small for j in range(len(t))]
    assert got == [s for t in small for s in t], got


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", nargs="?", default="evaluate", choices=["evaluate", "predict"])
    ap.add_argument("text", nargs="?", help="for predict: signs in reading order, ? for each gap")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.command == "predict":
        if not MODEL_FILE.exists():
            sys.exit("no trained model yet: run `python sequence_analysis/masked_model.py` first")
        model, tokens, temperature = load()
        text = [None if t == "?" else int(t) for t in args.text.split()]
        for g, top in fill(model, tokens, temperature, text).items():
            print(f"gap at position {g + 1}: " + ", ".join(f"{s} ({p:.0%})" for s, p in top))
        sys.exit()

    selfcheck()
    seeds = [0] if args.quick else [0, 1, 2]
    cfg = CONFIG | ({"max_epochs": 60, "patience": 10} if args.quick else {})
    lines = load_lines()
    texts, _ = ebuds(lines)
    vocab = sorted({s for t in texts for s in t})
    outcomes, groups, epochs, temps = evaluate(texts, vocab, seeds, cfg)
    m2 = json.loads(M2_RESULTS.read_text())
    res = {"models": collapse(outcomes, seeds), "milestone_2_restoration": m2["restoration"],
           "milestone_2_other": m2["restoration_by_model"]}
    assert abs(res["models"]["bigram"]["top1"] - m2["restoration"]["top1"]) < 1e-9, "bigram does not reproduce Milestone 2"
    assert abs(res["models"]["bigram"]["top5"] - m2["restoration"]["top5"]) < 1e-9
    res["difference_top1_vs_bigram"] = paired_bootstrap(outcomes["net_seed0_calibrated"]["hit1"].astype(float), outcomes["bigram"]["hit1"].astype(float), groups)
    res["difference_top1_average_vs_bigram"] = paired_bootstrap(outcomes["average_seed0"]["hit1"].astype(float), outcomes["bigram"]["hit1"].astype(float), groups)
    lo, hi = res["difference_top1_vs_bigram"][1:]
    res["verdict"] = ("The transformer's top-1 is reliably higher than the bigram's." if lo > 0 else
                      "The transformer's top-1 is reliably lower than the bigram's. With about 1,100 training texts per fold it does not "
                      "even match the neighbour statistics a bigram counts directly, and averaging the two only ties the bigram, so "
                      "the wider context the network can read adds nothing measurable on this corpus." if hi < 0 else
                      "The transformer and the bigram are statistically tied on top-1: the interval for the difference "
                      "includes zero. A likely reason: with about 1,100 training texts per fold, the wider context a transformer "
                      "can use adds little to what the two immediate neighbours already give.")

    OUT.mkdir(parents=True, exist_ok=True)
    reliability(outcomes, [("bigram", "bigram", "#4a9bd1"), ("net_seed0", "transformer, raw", "#c9663c"),
                           ("net_seed0_calibrated", "transformer, calibrated", "#7a3b1d")], OUT / "reliability.png")

    model, tokens, temperature = train_final(texts, vocab)
    save(model, tokens, temperature)
    damaged = damaged_texts(lines, vocab)
    DERIVED.mkdir(parents=True, exist_ok=True)
    confs, sizes, top1s = [], [], []
    with open(DERIVED / "damaged_text_suggestions.tsv", "w") as f:
        f.write("text_reading_order\tgap_position\ttop5\n")
        for t in damaged:
            gap = t.index(0)
            top = fill(model, tokens, temperature, [None if s == 0 else s for s in t], k=len(vocab))[gap]
            p = np.array([q for _, q in top])
            confs.append(p[0]); top1s.append(top[0][0]); sizes.append(int(np.argmax(np.cumsum(p) >= 0.9) + 1))
            f.write(" ".join("?" if s == 0 else str(s) for s in t) + f"\t{gap + 1}\t" + " ".join(f"{s}:{q:.2f}" for s, q in top[:5]) + "\n")
    res["damaged_texts"] = {"texts": len(damaged), "median_top1_confidence": float(np.median(confs)),
                            "confident_over_50pct": int(sum(c > 0.5 for c in confs)), "confident_over_80pct": int(sum(c > 0.8 for c in confs)),
                            "median_90pct_set_size": float(np.median(sizes)), "most_common_top1": Counter(top1s).most_common(5)}
    res["settings"] = {"texts": len(texts), "trials": int(len(groups)), "candidates": len(vocab), "seeds": seeds, "config": cfg,
                       "parameters": sum(p.numel() for p in model.parameters()), "median_epochs": int(np.median(epochs)),
                       "median_temperature_bigram": float(np.median(temps["bigram"])), "median_temperature_net": float(np.median(temps["net"])),
                       "quick": args.quick}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_report(res)
    for name in ("unigram", "bigram", "bigram_calibrated", "net", "net_calibrated", "average"):
        v = res["models"][name]
        print(f"{name:20s} top1 {v['top1']:.3f} top5 {v['top5']:.3f} bits {v['bits_per_sign']:.2f} in90 {v['hit_in_90pct_set']:.3f} ece {v['ece_top1']:.3f}")
    print("diff vs bigram:", res["difference_top1_vs_bigram"], "average:", res["difference_top1_average_vs_bigram"])
    print(f"report -> {OUT / 'milestone_3_report.md'}")
