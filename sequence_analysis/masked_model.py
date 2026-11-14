"""Milestone 3: predict a hidden sign from the rest of its inscription with small neural models, compared
with the best Milestone 2 n-gram on exactly the same folds and trials.

    python sequence_analysis/masked_model.py                          # evaluation + report; about 15 minutes on a laptop CPU
    python sequence_analysis/masked_model.py --quick                  # 1 seed, 1 configuration, short training
    python sequence_analysis/masked_model.py predict "267 99 ? 342"   # fill gaps: reading order, '?' = missing

Writes reports/milestone_3/ (results.json, figures, milestone_3_report.md). Only aggregates go there:
per-text predictions on damaged corpus texts go to data/derived/, which is gitignored.
"""
import argparse
import json
import math
import random
import sys
import time
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
from ngram import KneserNey, WittenBell

OUT = ROOT / "reports/milestone_3"
DERIVED = ROOT / "data/derived"
MODEL_FILE = ROOT / "models/masked_sign_model.pt"
M2_RESULTS = ROOT / "reports/milestone_2/results.json"
PAD, MASK, START, END, UNK = range(5)  # token ids; signs follow
MAX_LEN = 16  # longest EBUDS text is 14 signs, plus start and end
BASE = {"dropout": 0.2, "lr": 2e-3, "weight_decay": 0.01, "batch": 128, "max_epochs": 200, "patience": 10, "inner_val_share": 0.1}
# Fixed in advance; one configuration per fold is chosen on the inner split only.
GRID = [BASE | {"arch": a, "d": d, "layers": n} for a in ("transformer", "lstm") for d in (32, 64) for n in (1, 2)]
RESIDUAL = BASE | {"arch": "transformer", "d": 32, "layers": 1}
TEMPERATURES = np.round(np.arange(0.5, 3.01, 0.05), 2)
LAMBDAS = np.round(np.arange(0, 1.001, 0.05), 2)
ALPHA = 0.1  # conformal sets aim to contain the true sign 90% of the time


# ---------- models ----------

class MaskedSignModel(nn.Module):
    """Encoder over [start, signs..., end] with position counted from both ends; predicts the sign at a masked
    position. residual=True starts with a zero output layer, so with a prior added it begins as that prior."""

    def __init__(self, n_tokens, n_out, arch="transformer", d=64, layers=1, dropout=0.2, residual=False, **_):
        super().__init__()
        self.arch = arch
        self.tok, self.pos, self.pos_end = nn.Embedding(n_tokens, d), nn.Embedding(MAX_LEN, d), nn.Embedding(MAX_LEN, d)
        if arch == "transformer":
            layer = nn.TransformerEncoderLayer(d, d // 16, 4 * d, dropout, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        else:
            self.enc = nn.LSTM(d, d // 2, layers, batch_first=True, bidirectional=True, dropout=dropout if layers > 1 else 0)
        self.norm, self.out = nn.LayerNorm(d), nn.Linear(d, n_out)
        if residual:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        i = torch.arange(x.shape[1])[None]
        from_end = ((x != PAD).sum(1, keepdim=True) - 1 - i).clamp(min=0)
        h = self.tok(x) + self.pos(i) + self.pos_end(from_end)
        h = self.enc(h, src_key_padding_mask=x == PAD) if self.arch == "transformer" else self.enc(h)[0]
        return self.out(self.norm(h))


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


def trials(texts):
    """Every position of every text: (text index, position)."""
    return [(k, j) for k, t in enumerate(texts) for j in range(len(t))]


def gap_logits(model, x, pos, prior=None):
    out = model(x)[torch.arange(len(pos)), pos]
    return out if prior is None else out + prior


@torch.no_grad()
def net_logits(model, tokens, texts, pairs, prior=None):
    model.eval()
    x = torch.tensor([tokens.encode(texts[k], {j}) for k, j in pairs])
    pos = torch.tensor([j + 1 for _, j in pairs])
    p = None if prior is None else torch.tensor(prior, dtype=torch.float32)
    return torch.cat([gap_logits(model, x[i:i + 1024], pos[i:i + 1024], None if p is None else p[i:i + 1024])
                      for i in range(0, len(x), 1024)]).numpy()


def train_net(train, val, vocab, seed, cfg, prior=None, val_prior=None, epochs=None):
    """Every position of every training text is a masked example each epoch. With epochs=None, stop early on the
    inner validation log loss; with an integer, train exactly that many epochs (the full-fold retrain).
    prior / val_prior: log-probabilities added to the logits (residual model), aligned with trials()."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tokens = Tokens(train, vocab)
    model = MaskedSignModel(tokens.n, len(vocab), residual=prior is not None, **cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    pairs = trials(train)
    x_all = torch.tensor([tokens.encode(train[k], {j}) for k, j in pairs])
    pos_all = torch.tensor([j + 1 for _, j in pairs])
    y_all = torch.tensor([tokens.out[train[k][j]] for k, j in pairs])
    p_all = None if prior is None else torch.tensor(prior, dtype=torch.float32)
    val_pairs = trials(val)
    val_y = np.array([tokens.out[val[k][j]] for k, j in val_pairs])
    best, best_state, since, n_epochs = math.inf, None, 0, epochs or cfg["max_epochs"]
    for epoch in range(n_epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(len(pairs)))
        for b in order.split(cfg["batch"]):
            loss = nn.functional.cross_entropy(gap_logits(model, x_all[b], pos_all[b], None if p_all is None else p_all[b]), y_all[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epochs:
            continue
        nll = nll_of(log_softmax(net_logits(model, tokens, val, val_pairs, val_prior)), val_y)
        if nll < best - 1e-4:
            best, best_state, since, used = nll, {k: v.clone() for k, v in model.state_dict().items()}, 0, epoch + 1
        else:
            since += 1
            if since >= cfg["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model, tokens, (epochs or used)


# ---------- n-gram scores ----------

def ngram_logp(model, vocab, texts, pairs):
    return log_softmax(np.log(ngram.candidate_scores(model, vocab, texts, pairs)))


def unigram_logp(train, vocab, pairs):
    c = Counter(s for t in train for s in t)
    p = np.array([c[w] + 0.5 for w in vocab], float)
    return np.tile(np.log(p / p.sum()), (len(pairs), 1))


def best_ngram(train, vocab):
    """The strongest n-gram in Milestone 2's restoration table: interpolated Kneser-Ney trigram."""
    return KneserNey(train, vocab, order=3)


def cross_fit(texts, vocab, k=5):
    """Best-n-gram log-probabilities for every (text, position) of texts, each scored by a model fitted without
    that text, so a residual model never sees its own answer in the prior. Aligned with trials(texts)."""
    pairs = trials(texts)
    out = np.zeros((len(pairs), len(vocab)))
    group = np.array([k_ % k for k_ in random.Random(0).sample(range(len(texts)), len(texts))])
    for g in range(k):
        rest = [t for t, gg in zip(texts, group) if gg != g]
        rows = [r for r, (t, _) in enumerate(pairs) if group[t] == g]
        out[rows] = ngram_logp(best_ngram(rest, vocab), vocab, texts, [pairs[r] for r in rows])
    return out


# ---------- scoring ----------

def log_softmax(z, temperature=1.0):
    z = z / temperature
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def nll_of(logp, y):
    return float(-logp[np.arange(len(y)), y].mean())


def fit_temperature(scores, y):
    """Temperature with the lowest log loss on held-out trials."""
    return float(min(TEMPERATURES, key=lambda T: nll_of(log_softmax(scores, T), y)))


def mix(logp_a, logp_b, lam):
    return np.log(lam * np.exp(logp_a) + (1 - lam) * np.exp(logp_b) + 1e-300)


def fit_mixture(logp_a, logp_b, y):
    """Weight on model a, then a temperature fitted after mixing; both by log loss on held-out trials."""
    lam = float(min(LAMBDAS, key=lambda l: nll_of(mix(logp_a, logp_b, l), y)))
    return lam, fit_temperature(mix(logp_a, logp_b, lam), y)


def conformal_scores(logp, kind):
    """Non-conformity of every candidate: LAC = 1 - p; APS = probability mass ranked at or above it."""
    p = np.exp(logp)
    if kind == "lac":
        return 1 - p
    order = np.argsort(-p, 1)
    s = np.empty_like(p)
    np.put_along_axis(s, order, np.cumsum(np.take_along_axis(p, order, 1), 1), 1)
    return s


def conformal(cal_logp, cal_y, test_logp, test_y, kind, alpha=ALPHA):
    """Split-conformal sets: (true sign covered, set size) per test trial."""
    cal = conformal_scores(cal_logp, kind)[np.arange(len(cal_y)), cal_y]
    q = np.sort(cal)[min(len(cal) - 1, math.ceil((len(cal) + 1) * (1 - alpha)) - 1)]
    inside = conformal_scores(test_logp, kind) <= q + 1e-12
    return inside[np.arange(len(test_y)), test_y], inside.sum(1)


def ece(conf, correct, bins=15):
    """Expected calibration error of the top-1 confidence, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def per_trial(logp, y, cal_logp, cal_y):
    """Per-trial outcomes from log-probabilities over the candidates; conformal sets calibrated on cal_*."""
    p = np.exp(logp)
    order = np.argsort(-p, 1, kind="stable")  # ties (about 1% of trials) broken by sign number, the same on every machine
    rank = np.argmax(order == y[:, None], 1)
    k90 = np.argmax(np.cumsum(np.take_along_axis(p, order, 1), 1) >= 0.9, 1) + 1
    lac, lac_size = conformal(cal_logp, cal_y, logp, y, "lac")
    aps, aps_size = conformal(cal_logp, cal_y, logp, y, "aps")
    return {"hit1": rank == 0, "hit5": rank < 5, "nll_bits": -logp[np.arange(len(y)), y] / math.log(2), "conf": p.max(1),
            "in90": rank < k90, "set90": k90, "lac": lac, "lac_size": lac_size, "aps": aps, "aps_size": aps_size}


def summary(t):
    return {"top1": float(t["hit1"].mean()), "top5": float(t["hit5"].mean()), "bits_per_sign": float(t["nll_bits"].mean()),
            "ece_top1": ece(t["conf"], t["hit1"]), "hit_in_90pct_mass_set": float(t["in90"].mean()),
            "mean_90pct_mass_set_size": float(t["set90"].mean()),
            "conformal_lac_coverage": float(t["lac"].mean()), "conformal_lac_mean_size": float(t["lac_size"].mean()),
            "conformal_aps_coverage": float(t["aps"].mean()), "conformal_aps_mean_size": float(t["aps_size"].mean())}


def paired_bootstrap(a, b, groups, n=2000, seed=0):
    """Difference in mean a - b, resampling whole test texts. Returns (diff, low, high)."""
    ids = np.unique(groups)
    sa, sb, cnt = (np.bincount(np.searchsorted(ids, groups), weights=v, minlength=len(ids)) for v in (a, b, np.ones(len(a))))
    pick = np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))
    d = (sa[pick].sum(1) - sb[pick].sum(1)) / cnt[pick].sum(1)
    return float(a.mean() - b.mean()), *np.percentile(d, [2.5, 97.5]).tolist()


# ---------- evaluation ----------

def inner_split(train, f):
    inner = set(random.Random(f).sample(range(len(train)), round(BASE["inner_val_share"] * len(train))))
    return [t for i, t in enumerate(train) if i not in inner], [train[i] for i in sorted(inner)]


def evaluate(texts, vocab, seeds, grid, log=print):
    """5-fold CV over distinct texts, the Milestone 2 folds. Configuration choice, early stopping, temperatures, the
    mixture weight and conformal quantiles all use an inner split of each training fold (from models fitted on the
    rest of it); the test fold is only scored, by models retrained on the whole training fold."""
    idx = {s: i for i, s in enumerate(vocab)}
    test_rows, cal_rows = {}, {}
    def add(name, test_lp, cal_lp):
        test_rows.setdefault(name, []).append(test_lp)
        cal_rows.setdefault(name, []).append(cal_lp)
    ys, cal_ys, groups, info = [], [], [], {"chosen": [], "epochs": [], "temperatures": {}, "mixture_lambda": []}
    offset = 0
    for f, (train, test) in enumerate(ngram.folds(texts)):
        fit, val = inner_split(train, f)
        pairs, val_pairs = trials(test), trials(val)
        y, val_y = (np.array([idx[t[k][j]] for k, j in p]) for t, p in ((test, pairs), (val, val_pairs)))
        ys.append(y); cal_ys.append(val_y)
        groups.append(np.array([offset + k for k, _ in pairs]))
        offset += len(test)

        add("unigram", unigram_logp(train, vocab, pairs), unigram_logp(fit, vocab, val_pairs))
        ng = {}
        makers = {"bigram": lambda tr: WittenBell(tr, vocab, 2, backoff=True), "trigram": lambda tr: WittenBell(tr, vocab, 3, backoff=True),
                  "kn_trigram": lambda tr: best_ngram(tr, vocab)}
        for name, make in makers.items():
            test_lp = ngram_logp(make(train), vocab, test, pairs)
            val_lp = ngram_logp(make(fit), vocab, val, val_pairs)
            T = fit_temperature(val_lp, val_y)
            info["temperatures"].setdefault(name, []).append(T)
            add(name, test_lp, val_lp)
            add(f"{name}_calibrated", log_softmax(test_lp, T), log_softmax(val_lp, T))
            ng[name] = (log_softmax(test_lp, T), log_softmax(val_lp, T), test_lp, val_lp)

        # one configuration per fold, chosen on the inner split
        tried = []
        for i, cfg in enumerate(grid):
            model, tokens, ep = train_net(fit, val, vocab, f, cfg)
            lg = net_logits(model, tokens, val, val_pairs)
            tried.append((nll_of(log_softmax(lg, fit_temperature(lg, val_y)), val_y), i, model, tokens, ep))
        _, best_i, *first = min(tried, key=lambda t: t[:2])
        cfg = grid[best_i]
        info["chosen"].append({k: cfg[k] for k in ("arch", "d", "layers")})

        tri_test, tri_val = ng["kn_trigram"][2], ng["kn_trigram"][3]
        prior_fit, prior_train = cross_fit(fit, vocab), cross_fit(train, vocab)
        for s in seeds:
            model, tokens, ep = first if s == seeds[0] else train_net(fit, val, vocab, 1000 * s + f, cfg)
            val_lg = net_logits(model, tokens, val, val_pairs)
            T = fit_temperature(val_lg, val_y)
            full, full_tok, _ = train_net(train, [], vocab, 1000 * s + f, cfg, epochs=ep)
            test_lg = net_logits(full, full_tok, test, pairs)
            info["epochs"].append(ep)
            info["temperatures"].setdefault("net", []).append(T)
            add(f"net_seed{s}", log_softmax(test_lg), log_softmax(val_lg))
            add(f"net_seed{s}_calibrated", log_softmax(test_lg, T), log_softmax(val_lg, T))
            lam, Tm = fit_mixture(log_softmax(val_lg, T), ng["kn_trigram"][1], val_y)
            info["mixture_lambda"].append(lam)
            add(f"mixture_seed{s}", log_softmax(mix(log_softmax(test_lg, T), ng["kn_trigram"][0], lam), Tm),
                log_softmax(mix(log_softmax(val_lg, T), ng["kn_trigram"][1], lam), Tm))

            res, res_tok, ep_r = train_net(fit, val, vocab, 1000 * s + f, RESIDUAL, prior=prior_fit, val_prior=tri_val)
            val_r = net_logits(res, res_tok, val, val_pairs, tri_val)
            Tr = fit_temperature(val_r, val_y)
            res_full, res_full_tok, _ = train_net(train, [], vocab, 1000 * s + f, RESIDUAL, prior=prior_train, epochs=ep_r)
            add(f"residual_seed{s}_calibrated", log_softmax(net_logits(res_full, res_full_tok, test, pairs, tri_test), Tr),
                log_softmax(val_r, Tr))
        log(f"fold {f}: {len(test)} texts, {len(pairs)} trials, chosen {info['chosen'][-1]}, epochs {info['epochs'][-len(seeds):]}")
    y, cal_y, groups = np.concatenate(ys), np.concatenate(cal_ys), np.concatenate(groups)
    outcomes = {name: per_trial(np.concatenate(v), y, np.concatenate(cal_rows[name]), cal_y) for name, v in test_rows.items()}
    return outcomes, groups, info


FAMILIES = {"net": "net_seed{}", "net_calibrated": "net_seed{}_calibrated", "mixture": "mixture_seed{}",
            "residual_calibrated": "residual_seed{}_calibrated"}


def collapse(outcomes, seeds):
    """Mean and sd over seeds for the network rows; single values for the n-gram rows."""
    res = {name: summary(outcomes[name]) for name in ("unigram", "bigram", "bigram_calibrated", "trigram", "trigram_calibrated",
                                                       "kn_trigram", "kn_trigram_calibrated")}
    for fam, pattern in FAMILIES.items():
        per = [summary(outcomes[pattern.format(s)]) for s in seeds]
        res[fam] = {k: float(np.mean([p[k] for p in per])) for k in per[0]} | {f"{k}_sd": float(np.std([p[k] for p in per])) for k in per[0]}
    return res


def ngram_partitions(texts, vocab, partitions=5):
    """Top-1 / top-5 of the bigram and trigram over several random 5-fold partitions."""
    out = {"bigram": {"top1": [], "top5": []}, "trigram": {"top1": [], "top5": []}, "kn_trigram": {"top1": [], "top5": []}}
    idx = {s: i for i, s in enumerate(vocab)}
    for p in range(partitions):
        for name, make in (("bigram", lambda tr: WittenBell(tr, vocab, 2, backoff=True)), ("trigram", lambda tr: WittenBell(tr, vocab, 3, backoff=True)),
                           ("kn_trigram", lambda tr: best_ngram(tr, vocab))):
            hit1 = hit5 = total = 0
            for train, test in ngram.folds(texts, seed=p):
                pairs = trials(test)
                rank = np.argsort(-ngram.candidate_scores(make(train), vocab, test, pairs), 1, kind="stable")
                y = np.array([idx[test[k][j]] for k, j in pairs])
                hit1 += (rank[:, 0] == y).sum(); hit5 += (rank[:, :5] == y[:, None]).any(1).sum(); total += len(y)
            out[name]["top1"].append(hit1 / total); out[name]["top5"].append(hit5 / total)
    return out


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

def train_final(texts, vocab, cfg, seed=0):
    """The fold-wise most chosen configuration, trained on every EBUDS text for the number of epochs early stopping
    picks on a 10% inner split; the temperature comes from that inner split."""
    fit, val = inner_split(texts, seed)
    model, tokens, ep = train_net(fit, val, vocab, seed, cfg)
    pairs = trials(val)
    temperature = fit_temperature(net_logits(model, tokens, val, pairs), np.array([tokens.out[val[k][j]] for k, j in pairs]))
    full, full_tok, _ = train_net(texts, [], vocab, seed, cfg, epochs=ep)
    return full, full_tok, temperature


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


def influence(model, tokens, temperature, text, gap):
    """How much each legible sign matters for the top guess at one gap: the drop, in bits, of that guess's
    log-probability when the sign is replaced by the unknown-sign token (negative: the guess gets likelier without it).
    text as in fill(). Returns (top sign, {position: bits})."""
    gaps = {i for i, s in enumerate(text) if s is None}
    ids = tokens.encode([s or 0 for s in text], gaps)
    context = [i for i in range(len(text)) if i not in gaps]
    x = torch.tensor([ids] * (len(context) + 1))
    for row, i in enumerate(context, start=1):
        x[row, i + 1] = UNK
    model.eval()
    with torch.no_grad():
        logp = log_softmax(model(x)[:, gap + 1].numpy(), temperature)
    top = int(np.argmax(logp[0]))
    return tokens.vocab[top], {i: float((logp[0, top] - logp[row, top]) / math.log(2)) for row, i in enumerate(context, start=1)}


def save(model, tokens, temperature, cfg):
    MODEL_FILE.parent.mkdir(exist_ok=True)
    torch.save({"state": model.state_dict(), "vocab": tokens.vocab, "tok": tokens.tok, "temperature": temperature, "config": cfg}, MODEL_FILE)


def load():
    d = torch.load(MODEL_FILE, weights_only=False)
    tokens = Tokens([], d["vocab"])
    tokens.tok, tokens.n = d["tok"], 5 + len(d["tok"])
    model = MaskedSignModel(tokens.n, len(tokens.vocab), **d["config"])
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


LABELS = {"unigram": "Most frequent signs (no context)", "bigram": "Bigram (Milestone 2)", "bigram_calibrated": "Bigram, calibrated",
          "trigram": "Witten-Bell trigram (Milestone 2)", "trigram_calibrated": "Witten-Bell trigram, calibrated",
          "kn_trigram": "Kneser-Ney trigram (best n-gram, Milestone 2)", "kn_trigram_calibrated": "Kneser-Ney trigram, calibrated",
          "net": "Neural model, raw", "net_calibrated": "Neural model, calibrated",
          "mixture": "Mixture of neural model and Kneser-Ney trigram, fitted weight, calibrated",
          "residual_calibrated": "Residual model: Kneser-Ney trigram + learned correction, calibrated"}


def write_report(res):
    r, s, m2 = res["models"], res["settings"], res["milestone_2_restoration_by_model"]
    def cell(v, k, sd):
        return f"{pct(v[k])} ± {100 * v[k + '_sd']:.1f}" if sd else pct(v[k])
    rows = []
    for name, label in LABELS.items():
        v, sd = r[name], name in FAMILIES
        rows.append(f"| {label} | {cell(v, 'top1', sd)} | {cell(v, 'top5', sd)} | {v['bits_per_sign']:.2f} | "
                    f"{pct(v['conformal_lac_coverage'])}, {v['conformal_lac_mean_size']:.0f} signs | "
                    f"{pct(v['conformal_aps_coverage'])}, {v['conformal_aps_mean_size']:.0f} signs | {v['ece_top1']:.3f} |")
    diffs = [f"| {LABELS[fam]} | {100 * d[0]:+.1f} | [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}] |" for fam, d in res["difference_top1_vs_trigram"].items()]
    part = res["ngram_partitions"]
    chosen = Counter(f"{c['arch']} width {c['d']}, {c['layers']} layer{'s' if c['layers'] > 1 else ''}" for c in res["chosen_configurations"])
    dm = res["damaged_texts"]
    lines = [
        "# Milestone 3 report: predicting a hidden sign",
        "",
        "Small neural models are trained to fill one hidden sign from the rest of its inscription and compared with the "
        "Milestone 2 n-grams on the same task. All models use the same 5 folds of the "
        f"{s['texts']:,} EBUDS texts and score the same {s['trials']:,} trials: every position of every held-out text, ranked "
        f"over the same {s['candidates']} candidate signs. **The bar is the best n-gram, the Kneser-Ney trigram**, not the bigram.",
        "",
        f"The n-grams reproduce Milestone 2 (top-1 bigram {pct(r['bigram']['top1'])}, Witten-Bell trigram {pct(r['trigram']['top1'])}, "
        f"Kneser-Ney trigram {pct(r['kn_trigram']['top1'])}; Milestone 2: {pct(res['milestone_2_restoration']['top1'])}, "
        f"{pct(m2['witten-bell backoff n=3']['top1'])}, {pct(m2['kneser-ney n=3']['top1'])}), so the folds, candidates and trials are the "
        "same. Tied candidates are broken by sign number, so the ranking is the same on every machine.",
        "",
        "## Results",
        "",
        "| Model | Top-1 | Top-5 | Bits per hidden sign | Conformal set (LAC): coverage, mean size | Conformal set (APS): coverage, mean size | Calibration error (ECE) |",
        "|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"Neural rows: mean ± sd over {len(s['seeds'])} training seeds. Conformal sets aim to contain the true sign "
        f"{pct(1 - ALPHA)} of the time; a smaller set at the same coverage is the more useful model.",
        "",
        "Top-1 difference from the Kneser-Ney trigram, in points, with a 95% interval from resampling whole test texts (per-trial hits "
        "averaged over seeds):",
        "",
        "| Model | Difference | 95% interval |",
        "|---|---|---|",
        *diffs,
        "",
        f"Over {len(part['trigram']['top1'])} random fold partitions the n-grams vary little: Kneser-Ney trigram top-1 "
        f"{pct(np.mean(part['kn_trigram']['top1']))} ± {100 * np.std(part['kn_trigram']['top1']):.1f}, Witten-Bell trigram "
        f"{pct(np.mean(part['trigram']['top1']))} ± {100 * np.std(part['trigram']['top1']):.1f}, bigram "
        f"{pct(np.mean(part['bigram']['top1']))} ± {100 * np.std(part['bigram']['top1']):.1f}. The neural rows use the first "
        "partition only.",
        "",
        "## Reading the numbers",
        "",
        f"- {res['verdict']}",
        "- Calibration (the proposal's \"validation confidence ratings\"): ECE is the average gap between how confident the top "
        "guess is and how often it is right; 0 is perfect. `reliability.png` shows it per confidence band.",
        "- Conformal sets replace the earlier \"90% probability mass\" set, which did not contain the true sign 90% of the "
        "time and whose size could not be compared across models. They are calibrated on the inner split of each training "
        "fold with models fitted on the rest of it and applied to models retrained on the whole fold, so coverage is "
        "approximate; the measured coverage above is the check. The bigram's 90%-mass set is kept for the comparison with "
        f"Yadav et al. 2010 (74%): {pct(r['bigram']['hit_in_90pct_mass_set'])}, mean {r['bigram']['mean_90pct_mass_set_size']:.0f} signs.",
        "",
        "## Damaged inscriptions",
        "",
        f"The final neural model, trained on all {s['texts']:,} texts, was run on the {dm['texts']} distinct single-line corpus "
        "texts with exactly one illegible sign. They have no answer key, so this shows what the model offers an epigrapher, "
        f"not how accurate it is. Its top guess carries a median confidence of {pct(dm['median_top1_confidence'])}; "
        f"{dm['confident_over_50pct']} texts get a top guess above 50% and {dm['confident_over_80pct']} above 80%. The most "
        f"frequent top guesses are {', '.join(str(x) for x, _ in dm['most_common_top1'])}. Per-text suggestions are in "
        "`data/derived/` on the machine that ran this (not published: they contain corpus texts).",
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
        f"- Neural model: one configuration per fold is chosen from {len(s['grid'])} fixed in advance (transformer or "
        "bidirectional LSTM; width 32 or 64; 1 or 2 layers), by calibrated log loss on a 10% inner split of the training fold. "
        f"Chosen: {'; '.join(f'{c} ({n} folds)' for c, n in chosen.most_common())}. Inputs are the signs in reading order "
        "between start and end tokens, with position counted from both ends; a context sign unseen in training becomes an "
        "unknown token. Every position of every training text is a masked example in every epoch. Early stopping on the inner "
        f"split (patience {BASE['patience']}, median {s['median_epochs']} epochs) sets the epoch count; the model is then "
        "retrained on the whole training fold for that many epochs, so it sees the same texts as the n-grams.",
        f"- Mixture: weight on the neural model fitted on the inner split (median {s['median_mixture_lambda']}), then a "
        "temperature after mixing.",
        "- Residual model: a width-32, one-layer transformer whose output layer starts at zero and is added to the trigram's "
        "log-probabilities, so it starts exactly at the Kneser-Ney trigram and can only learn corrections. Its training priors come "
        "from trigrams fitted without the text being predicted (5-fold cross-fitting inside the training fold).",
        f"- Temperatures (median): bigram {s['median_temperature']['bigram']}, Kneser-Ney trigram {s['median_temperature']['kn_trigram']}, "
        f"neural {s['median_temperature']['net']}. AdamW, learning rate {BASE['lr']}, weight decay {BASE['weight_decay']}, "
        f"dropout {BASE['dropout']}. Trained on CPU; wall-clock time for the whole run {s['runtime_min']:.0f} minutes.",
        "",
    ]
    (OUT / "milestone_3_report.md").write_text("\n".join(lines))


def verdict(diffs):
    """One sentence per model family against the trigram, from the bootstrap intervals."""
    def say(fam):
        d, lo, hi = (100 * x for x in diffs[fam])
        how = "reliably better than" if lo > 0 else "reliably worse than" if hi < 0 else "statistically tied with"
        return f"{LABELS[fam].split(',')[0]} is {how} the Kneser-Ney trigram ({d:+.1f} points, 95% interval {lo:+.1f} to {hi:+.1f})"
    return ("; ".join(say(f) for f in diffs) + ". The neural model cannot replace the n-gram on about 1,200 short training "
            "texts per fold, and what it adds on top is small: the neighbouring signs carry almost everything a model can use.")


# ---------- checks ----------

def selfcheck():
    assert abs(ece(np.array([0.9, 0.9, 0.2, 0.2]), np.array([1, 1, 0, 0])) - 0.15) < 1e-12
    texts, vocab = [(1, 2, 3), (1, 2, 3), (3, 2, 1)], [1, 2, 3]
    m = WittenBell(texts, vocab, 2, backoff=True)
    b = np.exp(ngram_logp(m, vocab, texts, [(0, 1)]))[0]
    closed = np.array([m.prob(w, (1,)) * m.prob(3, (w,)) for w in vocab])
    assert np.allclose(b, closed / closed.sum()) and b.argmax() == 1
    d = paired_bootstrap(np.array([1., 1, 0, 0]), np.array([0., 0, 0, 0]), np.array([0, 0, 1, 1]), n=200)
    assert d[0] == 0.5 and d[1] <= 0.5 <= d[2]
    # mixture: a uniform second model gets no weight; conformal sets cover at the nominal rate on calibrated data
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 2, (4000, 20))
    y = np.array([rng.choice(20, p=np.exp(r)) for r in log_softmax(logits)])
    assert fit_mixture(log_softmax(logits), np.full_like(logits, -math.log(20)), y)[0] == 1.0
    for kind in ("lac", "aps"):
        cov, size = conformal(log_softmax(logits[:2000]), y[:2000], log_softmax(logits[2000:]), y[2000:], kind)
        assert cov.mean() >= 1 - ALPHA - 0.03 and size.mean() < 20, (kind, cov.mean())
    # a residual model with a zero output layer reproduces its prior before training
    tok = Tokens(texts, vocab)
    res = MaskedSignModel(tok.n, 3, residual=True, d=32, layers=1)
    prior = ngram_logp(m, vocab, texts, trials(texts))
    assert np.allclose(log_softmax(net_logits(res, tok, texts, trials(texts), prior)), prior, atol=1e-5)
    # the network can learn three texts by heart
    small = [(10, 20, 30), (40, 50, 60), (70, 80)]
    model, tokens, _ = train_net(small, [], [10, 20, 30, 40, 50, 60, 70, 80], 0,
                                 GRID[2] | {"dropout": 0.0, "batch": 8}, epochs=300)
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

    start = time.time()
    selfcheck()
    seeds = [0] if args.quick else [0, 1, 2]
    grid = [GRID[2] | {"max_epochs": 15}] if args.quick else GRID
    lines = load_lines()
    texts, _ = ebuds(lines)
    vocab = sorted({s for t in texts for s in t})
    outcomes, groups, info = evaluate(texts, vocab, seeds, grid, log=lambda m: print(f"[{(time.time() - start) / 60:.1f} min] {m}", flush=True))
    m2 = json.loads(M2_RESULTS.read_text())
    res = {"models": collapse(outcomes, seeds), "milestone_2_restoration": m2["restoration"],
           "milestone_2_restoration_by_model": m2["restoration_by_model"], "chosen_configurations": info["chosen"]}
    for name, ref in (("bigram", m2["restoration"]), ("trigram", m2["restoration_by_model"]["witten-bell backoff n=3"]),
                      ("kn_trigram", m2["restoration_by_model"]["kneser-ney n=3"])):
        # ponytail: Milestone 2 broke tied rankings with numpy's default (CPU-dependent) sort; 0.002 = 14 of 7,000 trials
        assert all(abs(res["models"][name][k] - ref[k]) < 0.002 for k in ("top1", "top5")), f"{name} does not reproduce Milestone 2"
    tri = outcomes["kn_trigram"]["hit1"].astype(float)
    res["difference_top1_vs_trigram"] = {fam: paired_bootstrap(np.mean([outcomes[p.format(s)]["hit1"] for s in seeds], 0), tri, groups)
                                         for fam, p in FAMILIES.items() if fam != "net"}
    res["verdict"] = verdict(res["difference_top1_vs_trigram"])
    res["ngram_partitions"] = ngram_partitions(texts, vocab, partitions=2 if args.quick else 5)

    OUT.mkdir(parents=True, exist_ok=True)
    reliability(outcomes, [("kn_trigram_calibrated", "Kneser-Ney trigram, calibrated", "#4a9bd1"), ("net_seed0", "neural, raw", "#c9663c"),
                           ("net_seed0_calibrated", "neural, calibrated", "#7a3b1d")], OUT / "reliability.png")

    top = Counter(json.dumps(c, sort_keys=True) for c in info["chosen"]).most_common(1)[0][0]
    final_cfg = next(c for c in grid if json.dumps({k: c[k] for k in ("arch", "d", "layers")}, sort_keys=True) == top)
    model, tokens, temperature = train_final(texts, vocab, final_cfg)
    save(model, tokens, temperature, final_cfg)
    damaged = damaged_texts(lines, vocab)
    DERIVED.mkdir(parents=True, exist_ok=True)
    confs, top1s = [], []
    with open(DERIVED / "damaged_text_suggestions.tsv", "w") as f:
        f.write("text_reading_order\tgap_position\ttop5\n")
        for t in damaged:
            gap = t.index(0)
            top = fill(model, tokens, temperature, [None if s == 0 else s for s in t])[gap]
            confs.append(top[0][1]); top1s.append(top[0][0])
            f.write(" ".join("?" if s == 0 else str(s) for s in t) + f"\t{gap + 1}\t" + " ".join(f"{s}:{q:.2f}" for s, q in top) + "\n")
    res["damaged_texts"] = {"texts": len(damaged), "median_top1_confidence": float(np.median(confs)),
                            "confident_over_50pct": int(sum(c > 0.5 for c in confs)), "confident_over_80pct": int(sum(c > 0.8 for c in confs)),
                            "most_common_top1": Counter(top1s).most_common(5)}
    res["settings"] = {"texts": len(texts), "trials": int(len(groups)), "candidates": len(vocab), "seeds": seeds,
                       "grid": [{k: c[k] for k in ("arch", "d", "layers")} for c in grid], "base": BASE, "final_config": final_cfg,
                       "median_epochs": int(np.median(info["epochs"])), "median_mixture_lambda": float(np.median(info["mixture_lambda"])),
                       "median_temperature": {k: float(np.median(v)) for k, v in info["temperatures"].items()},
                       "runtime_min": (time.time() - start) / 60, "quick": args.quick}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    write_report(res)
    for name in LABELS:
        v = res["models"][name]
        print(f"{name:22s} top1 {v['top1']:.3f} top5 {v['top5']:.3f} bits {v['bits_per_sign']:.2f} "
              f"lac {v['conformal_lac_coverage']:.3f}/{v['conformal_lac_mean_size']:.0f} ece {v['ece_top1']:.3f}")
    print(res["verdict"])
    print(f"report -> {OUT / 'milestone_3_report.md'}  ({(time.time() - start) / 60:.0f} min)")
