"""Demo app for Milestones 1 and 2.

    streamlit run app.py
    python app.py            # self-check: asserts, then runs the whole app once headless

Runs locally only: it reads the IM77 corpus and the trained weights from this machine.
"""
import itertools
import json
import os
import random
import sys
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "grapheme_extraction"), str(ROOT / "sequence_analysis")]

import corpus  # noqa: E402
import ngram  # noqa: E402
from classifier import crop_sign, load_classifier  # noqa: E402
from pipeline import extract, iou, levenshtein, load_detector  # noqa: E402
from synthetic_generator import GLYPH_DIR, degradation, load_glyphs, load_sequences, make_seal  # noqa: E402

M2 = ROOT / "reports/milestone_2"
DETECTORS = {"YOLOv8n": ROOT / "runs/yolov8n/weights/best.pt", "RT-DETR-L": ROOT / "runs/rtdetr-l/weights/best.pt",
             "OpenCV baseline (no learning)": "opencv"}
# Seeds chosen to show the real spread of results (exact match on synthetic seals is about 43%), not only successes.
PRESETS = {"A: clean read, 7 signs": 38, "B: clean read, 5 signs": 33, "C: one wrong sign of 8": 6,
           "D: three wrong signs of 5": 19, "E: spurious box, sequence fails": 8}
REAL = {"Impression (cropped from the Met photo)": ("met_impression.jpg", False),
        "Carved seal (same object, mirror image)": ("met_seal.jpg", True),
        "Original Met photo, seal and impression together": ("met_cropped.jpg", False)}
GREEN, RED, BLUE, GREY = (60, 170, 60), (40, 40, 220), (220, 130, 30), (150, 150, 150)  # BGR


# ---------- shared ----------

@st.cache_resource
def glyph(sign):
    """Reference drawing of a Mahadevan sign, black on white."""
    return 255 - cv2.imread(str(GLYPH_DIR / f"{sign}.png"), cv2.IMREAD_GRAYSCALE)


def strip(signs, hidden=None, tile=84):
    """Signs given in reading order, drawn as they sit on an impression: first-read sign on the right."""
    tiles = []
    for i, s in enumerate(signs):
        t = np.full((tile + 22, tile, 3), 255, np.uint8)
        if i == hidden:
            t[4:tile - 4, 4:tile - 4] = 225
            cv2.putText(t, "?", (tile // 2 - 12, tile // 2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (90, 90, 90), 3)
        else:
            t[6:tile - 6, 6:tile - 6] = cv2.cvtColor(cv2.resize(glyph(s), (tile - 12, tile - 12), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
            cv2.putText(t, str(s), (6, tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)
        tiles.append(t)
    return np.hstack(tiles[::-1])


# ---------- Milestone 1 ----------

@st.cache_resource
def detector(name, conf):
    return load_detector(DETECTORS[name], conf)


@st.cache_resource
def classifier():
    return load_classifier()


@st.cache_resource
def generator_inputs():
    return load_glyphs(), load_sequences()


def synthetic(seed):
    """Deterministic synthetic seal: (BGR image, truth [(sign, x0, y0, x1, y1)])."""
    random.seed(seed)
    np.random.seed(seed)
    degrade = degradation()
    degrade.set_random_seed(seed)
    return make_seal(*generator_inputs(), degrade)


def match(truth, signs):
    """Per predicted sign: the true sign ID it overlaps (IoU >= 0.5, one-to-one), or None."""
    pred = [(x, y, x + w, y + h) for x, y, w, h in (s["bbox_xywh"] for s in signs)]
    hit = [None] * len(pred)
    for g in truth:
        free = [p for p in range(len(pred)) if hit[p] is None]
        best = max(free, key=lambda p: iou(g[1:], pred[p]), default=None)
        if best is not None and iou(g[1:], pred[best]) >= 0.5:
            hit[best] = g[0]
    return hit


def draw(img, signs, hit=None):
    """Boxes and top-1 sign numbers. With ground truth: green = right sign, red = wrong sign or not a sign."""
    k = max(1.0, 900 / img.shape[1])
    img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    for i, s in enumerate(signs):
        x, y, w, h = (round(v * k) for v in s["bbox_xywh"])
        top1 = s["top_k"][0][0]
        colour = BLUE if hit is None else GREEN if hit[i] == top1 else RED
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
        (tw, th), _ = cv2.getTextSize(str(top1), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(img, (x, max(0, y - th - 8)), (x + tw + 6, max(th + 8, y)), colour, -1)
        cv2.putText(img, str(top1), (x + 3, max(th + 4, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def milestone1():
    st.caption("A photograph goes in; the detector boxes each sign, the classifier names it with a Mahadevan sign number "
               "(1-417), and the signs are listed in reading order, right to left.")
    left, right = st.columns([1, 2])
    with left:
        source = st.radio("Image", ["Synthetic seal (answer known)", "Real seal photo", "Upload a JPEG"])
        truth, mirror = None, False
        if source.startswith("Synthetic"):
            choice = st.selectbox("Example", [*PRESETS, "Random"])
            if choice == "Random":
                if st.button("New random seal") or "seed" not in st.session_state:
                    st.session_state.seed = random.SystemRandom().randrange(10 ** 6)
                seed = st.session_state.seed
            else:
                seed = PRESETS[choice]
            img, truth = synthetic(seed)
            st.caption(f"Generated from seed {seed}: sign drawings engraved on procedural stone, then blurred and distorted.")
        elif source.startswith("Real"):
            name, mirror = REAL[st.selectbox("Photo", list(REAL))]
            img = cv2.imread(str(ROOT / "data/samples" / name))
            st.caption("Indus stamp seal, ca. 2600-1900 B.C., The Metropolitan Museum of Art 49.40.1 (public domain).")
        else:
            up = st.file_uploader("JPEG or PNG", type=["jpg", "jpeg", "png"])
            img = cv2.imdecode(np.frombuffer(up.getvalue(), np.uint8), cv2.IMREAD_COLOR) if up else None
            if up and img is None:
                st.error("That file could not be read as an image.")
        if truth is None:
            mirror = st.checkbox("Photo shows the carved seal itself (mirror it first)", value=mirror)
        name = st.selectbox("Detector", [n for n, w in DETECTORS.items() if w == "opencv" or w.exists()])
        conf = st.slider("Detector confidence threshold", 0.05, 0.9, 0.25, 0.05)
    if img is None:
        return
    out = extract(img, detector(name, conf), classifier(), is_seal_matrix=mirror)
    view = cv2.flip(img, 1) if mirror else img  # boxes are in impression view
    hit = match(truth, out["signs"]) if truth else None
    with right:
        if truth is None:
            st.warning("The detector has only been trained on synthetic seals. On real photographs it merges neighbouring "
                       "signs and boxes the animal, so the numbers below are **not a reading**. Closing this gap needs "
                       "hand-labelled real seals.")
        st.image(draw(view, out["signs"], hit), channels="BGR", width=680)
        if truth:
            gold = [g[0] for g in sorted(truth, key=lambda g: -(g[1] + g[3]) / 2)]
            a, b, c = st.columns(3)
            a.metric("Signs found", f"{sum(h is not None for h in hit)} of {len(gold)}")
            b.metric("Signs named correctly", f"{sum(h == s['top_k'][0][0] for h, s in zip(hit, out['signs']))} of {len(gold)}")
            c.metric("Edits to fix the sequence", levenshtein(gold, out["sequence"]))
            st.caption("Green box: right sign. Red box: wrong sign, or not a sign at all.")
            st.write("**True sequence** (first-read sign on the right)")
            st.image(strip(gold), channels="BGR")
        if out["sequence"]:
            st.write("**Pipeline output**" + ("" if truth else " (not a reading)"))
            st.image(strip(out["sequence"]), channels="BGR")
            st.code(" ".join(map(str, out["sequence"])), language=None)

    st.subheader("Each detected sign, in reading order")
    gray = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
    for i, s in enumerate(out["signs"][:24]):
        x, y, w, h = s["bbox_xywh"]
        cols = st.columns([1, 1, 3, 2])
        cols[0].image(crop_sign(gray, (x, y, x + w, y + h)), caption=f"#{i + 1} crop", width=90)
        cols[1].image(glyph(s["top_k"][0][0]), caption=f"best match: {s['top_k'][0][0]}", width=90)
        cols[2].write("Top 5: " + ", ".join(f"**{sid}** ({p:.0%})" for sid, p in s["top_k"]))
        if hit:
            top5 = [sid for sid, _ in s["top_k"]]
            cols[3].write("Not a sign" if hit[i] is None else f"True sign {hit[i]}: " + (
                "correct" if hit[i] == top5[0] else "in the top 5" if hit[i] in top5 else "missed"))
    if len(out["signs"]) > 24:
        st.caption(f"{len(out['signs']) - 24} more boxes not listed.")
    with st.expander("Raw JSON output"):
        st.json(out, expanded=False)


# ---------- Milestone 2 ----------

@st.cache_resource
def sequence_model():
    """Bigram model trained on four fifths of the corpus; the demo texts are the held-out fifth,
    so the model has never seen the inscription it is asked about."""
    texts, _ = corpus.ebuds(corpus.load_lines())
    vocab = sorted({s for t in texts for s in t})
    train, test = next(ngram.folds(texts))
    return ngram.WittenBell(train, vocab, 2, backoff=True), vocab, [t for t in test if len(t) >= 4]


def bits_per_sign(model, text):
    ll, n = ngram.log2_likelihood(model, [text])
    return -ll / n


def candidates(model, vocab, text, j):
    """Probability of every sign at position j given its two neighbours: P(w | prev) * P(next | w), normalised."""
    s = [ngram.BOS, *text, ngram.EOS]
    score = np.array([model.prob(w, (s[j],)) * model.prob(s[j + 2], (w,)) for w in vocab])
    return score / score.sum()


def reorderings(text, n=500, seed=0):
    """Distinct reorderings of a text other than itself: all of them if short, else a random sample."""
    if len(text) <= 6:
        return sorted(set(itertools.permutations(text)) - {tuple(text)})
    rng, out = random.Random(seed), set()
    while len(out) < n:
        out.add(tuple(rng.sample(text, len(text))))
    return sorted(out - {tuple(text)})


def milestone2():
    model, vocab, held_out = sequence_model()
    res = json.loads((M2 / "results.json").read_text())
    st.caption("Statistics on the published sign sequences (Mahadevan 1977 concordance, 1,548 texts after filtering). "
               "They show that sign order is structured and not random. They say nothing about what the signs mean.")
    guess, shuffle, explore, results = st.tabs(["Guess the missing sign", "Real order vs shuffled", "Sign explorer", "Headline results"])

    if st.session_state.get("new_text") or "text" not in st.session_state:
        st.session_state.text = random.SystemRandom().randrange(len(held_out))
    text = held_out[st.session_state.text]

    with guess:
        st.write("One sign is hidden from an inscription the model has never seen. It ranks all "
                 f"{len(vocab)} signs for the gap using only the two neighbours.")
        st.button("Another inscription", key="new_text")
        j = st.slider("Hide which sign (1 = first read, on the right)", 1, len(text), min(2, len(text))) - 1
        st.image(strip(text, hidden=j), channels="BGR")
        p = candidates(model, vocab, text, j)
        order = np.argsort(-p)
        st.write("**The model's five best guesses**")
        for col, k in zip(st.columns(5), order[:5]):
            col.image(glyph(vocab[k]), width=90, caption=f"sign {vocab[k]}: {p[k]:.0%}")
        if st.checkbox("Reveal the answer"):
            rank = int(np.where(order == vocab.index(text[j]))[0][0]) + 1
            st.image(strip(text), channels="BGR")
            (st.success if rank <= 5 else st.info)(f"The hidden sign is {text[j]}. The model ranked it {rank} of {len(vocab)}.")
        r = res["restoration"]
        st.caption(f"Across every position of every text (5-fold, {r['trials']:,} trials) the first guess is right "
                   f"{r['top1']:.1%} of the time and the answer is in the top five {r['top5']:.1%} of the time. "
                   f"Guessing at random would be right {1 / len(vocab):.1%} of the time.")

    with shuffle:
        st.write("If sign order carried no structure, the model would find a real inscription no easier to predict than "
                 "the same signs in any other order. Lower means more predictable.")
        st.image(strip(text), channels="BGR")
        real = bits_per_sign(model, text)
        others = np.array([bits_per_sign(model, t) for t in reorderings(text)])
        if len(others):
            a, b, c = st.columns(3)
            a.metric("Real order", f"{real:.2f} bits per sign")
            b.metric("Typical reordering", f"{np.median(others):.2f} bits per sign")
            c.metric("Reorderings the real order beats", f"{(others > real).mean():.0%}", help=f"of {len(others)} distinct reorderings")
            bars = alt.Chart(pd.DataFrame({"bits": others})).mark_bar(color="#4c78a8").encode(
                alt.X("bits:Q", bin=alt.Bin(maxbins=30), title="Bits per sign (lower = more predictable)"),
                alt.Y("count():Q", title="Reorderings"))
            mark = pd.DataFrame({"bits": [real], "label": ["real order"]})
            rule = alt.Chart(mark).mark_rule(color="#222", strokeWidth=2).encode(x="bits:Q", tooltip=[alt.Tooltip("bits:Q", format=".2f")])
            label = alt.Chart(mark).mark_text(align="left", dx=6, dy=-90, color="#222").encode(x="bits:Q", text="label:N")
            st.altair_chart((bars + rule + label).properties(height=260), width="stretch")
        else:
            st.info("Every sign in this text is the same, so it has no other ordering.")
        st.caption("A single short text can lose to some of its reorderings; the claim is about the corpus. "
                   f"Corpus-wide, neighbouring signs share {res['structure_test']['observed_bits']:.2f} bits of information against "
                   f"{res['structure_test']['shuffled_mean_bits']:.2f} when signs are shuffled within each text "
                   f"(p = {res['structure_test']['p']:.3f}).")

    with explore:
        pos = pd.read_csv(M2 / "sign_positions.csv").set_index("sign")
        pairs = pd.read_csv(M2 / "sign_pairs.csv")
        sign = st.selectbox("Sign (most frequent first)", list(pos.index), index=list(pos.index).index(342),
                            format_func=lambda s: f"{s}  ({pos.loc[s, 'count']} occurrences)")
        row = pos.loc[sign]
        a, b = st.columns([1, 3])
        a.image(glyph(sign), width=130, caption=f"Mahadevan sign {sign}")
        where = pd.DataFrame({"Position in the text": ["First sign read", "Middle", "Last sign read"],
                              "Occurrences": [row.as_text_beginner, row.medial, row.as_text_ender]})
        b.altair_chart(alt.Chart(where).mark_bar(color="#4c78a8", cornerRadiusEnd=4).encode(
            alt.X("Occurrences:Q", axis=alt.Axis(tickCount=6)), alt.Y("Position in the text:N", sort=None, title=None),
            tooltip=["Position in the text", "Occurrences"]).properties(height=150), width="stretch")
        for title, mine, other in (("Most often followed by", "first_sign", "second_sign"), ("Most often preceded by", "second_sign", "first_sign")):
            st.write(f"**{title}**")
            top = pairs[pairs[mine] == sign].nlargest(5, "count")
            for col, r in zip(st.columns(5), top.itertuples()):
                col.image(glyph(getattr(r, other)), width=80)
                col.caption(f"sign {getattr(r, other)}: {r.count} times, {r.count / r.expected_if_independent:.1f}x chance")
            if top.empty:
                st.caption("No recorded neighbour.")
        st.caption("\"x chance\" compares the count with what the two signs' frequencies alone would give.")

    with results:
        rep = res["replication"]
        a, b, c = st.columns(3)
        a.metric("Texts / sign tokens / distinct signs", "1,548 / 7,000 / 377", help="Identical to Yadav et al. 2010")
        b.metric("Commonest pair of signs", f"{rep['top_pair'][0]} then {rep['top_pair'][1]}, {rep['top_pair'][2]} times")
        c.metric("Signs covering 80% of text endings / beginnings", f"{rep['coverage80']['enders']} / {rep['coverage80']['beginners']}")
        st.write("Text endings use far fewer signs than text beginnings, which is what a script written in one direction "
                 "looks like. The layout is consistent with right-to-left, given the concordance's own direction labels.")
        for name, caption in (("perplexity_by_order.png", "How well models of each order predict held-out texts"),
                              ("zipf.png", "Sign frequencies"), ("direction_null.png", "Direction test against its null"),
                              ("error_propagation.png", "What Milestone 1's error rate would do to these statistics")):
            if (M2 / name).exists():
                st.image(str(M2 / name), caption=caption)
        st.caption("Full method and every number: reports/milestone_2/milestone_2_report.md")


def selftest():
    signs = [{"bbox_xywh": [100, 10, 30, 50], "top_k": [[7, 1.0]]}, {"bbox_xywh": [300, 300, 30, 50], "top_k": [[9, 1.0]]}]
    assert match([(7, 98, 9, 131, 61), (5, 10, 10, 40, 60)], signs) == [7, None]  # second box overlaps no true sign
    model = ngram.WittenBell([(1, 2, 3), (1, 2, 3), (3, 2, 1)], [1, 2, 3], 2, backoff=True)
    p = candidates(model, [1, 2, 3], (1, 2, 3), 1)
    assert abs(p.sum() - 1) < 1e-9 and p.argmax() == 1  # the hidden middle sign, 2, is the best guess
    assert len(reorderings((1, 2, 3))) == 5 and (1, 2, 3) not in reorderings(tuple(range(9)), n=50)
    assert strip([1, 2, 3], hidden=1).shape == (106, 252, 3)
    from streamlit.testing.v1 import AppTest
    os.environ["INDUS_APP_UNDER_TEST"] = "1"
    for tab in ("m1", "m2"):
        at = AppTest.from_file(__file__, default_timeout=300)
        at.session_state["_only"] = tab
        at.run()
        assert not at.exception, at.exception
    print("ok")


if __name__ == "__main__":
    if not st.runtime.exists() and not os.environ.get("INDUS_APP_UNDER_TEST"):
        selftest()
        raise SystemExit
    st.set_page_config(page_title="Indus script: computational epigraphy", layout="wide")
    st.title("Computational epigraphy of the Indus script")
    only = st.session_state.get("_only")  # set by the self-check, which cannot click tabs
    m1, m2 = st.tabs(["Milestone 1: photo to sign numbers", "Milestone 2: sign-sequence statistics"])
    if only != "m2":
        with m1:
            milestone1()
    if only != "m1":
        with m2:
            milestone2()
