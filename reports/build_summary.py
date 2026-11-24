"""Build reports/SUMMARY.md, the one-page summary of every milestone and extension, from the JSON results only.

    python reports/build_summary.py

Every number below is read from a results file under reports/; a missing file is skipped with a line saying so.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent


def load(rel):
    p = R / rel
    return json.loads(p.read_text()) if p.exists() else None


def pct(x):
    return f"{100 * x:.1f}%"


def reading(d):
    r = d.get("reading", "")
    return " ".join(r) if isinstance(r, list) else r


def milestone1():
    e, real, rs, zs = load("milestone_1/eval_results.json"), load("milestone_1/real/real_results.json"), load("milestone_1/rescoring.json"), load("milestone_1_zero_shot/zero_shot.json")
    L = ["## Milestone 1: from a seal photograph to sign numbers", "",
         "**Question.** Can a seal photograph be turned into an ordered list of Mahadevan sign numbers?", "",
         "**Method.** A single-class detector (YOLOv8n or RT-DETR) boxes each sign and a frozen DINOv2 nearest-neighbour classifier names it. "
         "Trained on synthetic seals made from the IM-417 drawings; tested on held-out synthetic seals and, where labelled, on real photographs.", ""]
    if not e:
        return L + ["`milestone_1/eval_results.json` is missing: run `grapheme_extraction/evaluate.py`.", ""]
    m = e["detectors"]["yolov8n"]["metrics"]
    rows = [("Synthetic test: detection recall", pct(m["Detection recall (IoU 0.5)"][0])),
            ("Synthetic test: sign named correctly", pct(m["Top-1 sign ID on detected signs"][0])),
            ("Synthetic test: whole sequence exact", pct(m["Exact sequence match"][0]))]
    if rs:
        rows.append(("Synthetic corpus-text seals: sign error rate, top-1 then re-ranked by the n-gram",
                     f"{rs['corpus']['top1']['sign_error_rate'][0]:.3f} then {rs['corpus']['rescored']['sign_error_rate'][0]:.3f}"))
    if real:
        w = real["with_real"]
        rows += [(f"Real test photos ({real['data']['split']['test']}): detector recall", pct(w["detection"]["recall"][0])),
                 ("Real test photos: sign named correctly on true boxes", pct(w["classification"]["top1"][0])),
                 ("Real test photos: whole-pipeline sign error rate", f"{w['end_to_end']['sign_error_rate'][0]:.2f}")]
    if zs and zs["methods"]["FastSAM"].get("real"):
        rows.append(("Real test photos: best zero-shot segmenter recall (FastSAM)", pct(zs["methods"]["FastSAM"]["real"]["recall"])))
    L += ["| Measure | Value |", "|---|---|", *[f"| {a} | {b} |" for a, b in rows], "",
          "**What it shows.** The pipeline works on synthetic seals and the detector finds most signs on real photographs. "
          "**What it does not.** It does not yet name real signs reliably" + ("" if real else "; no real labelled photographs were found") + ", so it cannot read a real inscription.", "",
          "Reports: [synthetic](milestone_1/milestone_1_report.md)" + (", [real photographs](milestone_1/real/real_report.md)" if real else "")
          + (", [re-ranking](milestone_1/rescoring.md)" if rs else "") + (", [zero-shot](milestone_1_zero_shot/zero_shot_report.md)" if zs else "") + ".", ""]
    return L


def milestone2():
    m = load("milestone_2/results.json")
    L = ["## Milestone 2: positional bias and Markov-chain modelling", "",
         "**Question.** Is the order of Indus signs structured, and which way is it written?", "",
         "**Method.** N-gram models, pair statistics and direction tests on the 1,548 EBUDS texts of Mahadevan's 1977 corpus, with shuffled corpora "
         "as nulls. Published figures from Yadav et al. 2010 are reproduced first.", ""]
    if not m:
        return L + ["`milestone_2/results.json` is missing: run `sequence_analysis/run_milestone2.py`.", ""]
    st, d = m["structure_test"], m["direction"]["primary: labelled right-to-left, single-line"]["delta_S"]
    rows = [("Corpus reproduced (texts / tokens / signs)", f"{m['ebuds']['texts']:,} / {m['ebuds']['tokens']:,} / {m['ebuds']['signs']}"),
            ("Information shared by neighbouring signs, real vs shuffled (bits)", f"{st['observed_bits']:.2f} vs {st['shuffled_mean_bits']:.2f}, p = {st['p']:.3f}"),
            ("Best n-gram perplexity (Kneser-Ney trigram)", f"{m['language_models']['kneser-ney n=3']['cv_perplexity']:.2f}"),
            ("Hidden sign restored, first guess (Kneser-Ney trigram)", pct(m["restoration_by_model"]["kneser-ney n=3"]["top1"])),
            ("Direction statistic, delta S (negative = right to left)", f"{d['value']:+.3f}")]
    L += ["| Measure | Value |", "|---|---|", *[f"| {a} | {b} |" for a, b in rows], "",
          "**What it shows.** Sign order is structured and non-random, and the layout is consistent with right-to-left reading. "
          "**What it does not.** None of it shows that the signs record a language.", "", "Report: [milestone_2_report.md](milestone_2/milestone_2_report.md).", ""]
    return L


def milestone3():
    m = load("milestone_3/results.json")
    L = ["## Milestone 3: predicting hidden signs", "",
         "**Question.** Can a model fill a missing sign from the rest of its inscription, with honest confidence?", "",
         "**Method.** Small neural models (transformer or BiLSTM) against the best n-gram on identical held-out trials, 5-fold, "
         "with calibration and conformal candidate sets.", ""]
    if not m:
        return L + ["`milestone_3/results.json` is missing: run `sequence_analysis/masked_model.py`.", ""]
    r = m["models"]
    rows = [(name, pct(r[k]["top1"]), pct(r[k]["top5"])) for name, k in (("Kneser-Ney trigram", "kn_trigram"), ("Neural model", "net_calibrated"),
                                                                       ("Mixture of the two", "mixture")) if k in r]
    L += ["| Model | First guess right | Right in top 5 |", "|---|---|---|", *[f"| {a} | {b} | {c} |" for a, b, c in rows], "",
          f"**Reading.** {m['verdict']}", "", "Report: [milestone_3_report.md](milestone_3/milestone_3_report.md).", ""]
    return L


def milestone4():
    m = load("milestone_4/results.json")
    L = ["## Milestone 4: testing claimed decipherments", "",
         "**Question.** Does a published sign-to-sound table read the inscriptions better than chance?", "",
         "**Method.** A pre-registered permutation test: the claimed table against 10,000 shuffles of its own values, scored by how Sanskrit-like "
         "the order of its output is, with positive and negative controls.", ""]
    if not m:
        return L + ["`milestone_4/results.json` is missing: run `decipherment/harness.py`.", ""]
    c = m["claims"]["yajnadevam-2026-07-15"]
    pc, nc = m["positive_control"]["0"], m["negative_control"]
    rows = [("Known table detected (positive control)", pct(min(v["power"] for v in pc.values()))),
            ("Random tables flagged (negative control, 5% expected)", pct(nc["share_below_0.05"])),
            ("Yajnadevam key: percentile among frequency-matched shuffles", f"{c['nulls']['binned_10']['sequence']['percentile']:.1f}"),
            ("Yajnadevam key: Holm-corrected p", f"{c['holm']['sequence']['binned_10']:.2f}")]
    L += ["| Measure | Value |", "|---|---|", *[f"| {a} | {b} |" for a, b in rows], "",
          "**What it shows.** The one fully published key is not distinguishable from shuffled versions of itself. "
          "**What it does not.** It cannot show whether the script records a language.", "", "Report: [milestone_4_report.md](milestone_4/milestone_4_report.md).", ""]
    return L


EXTENSIONS = [("milestone_2/numerals.json", "Do the stroke signs behave like numerals?", "milestone_2/numerals_report.md"),
              ("milestone_2/predict_object.json", "Does the text predict the object?", "milestone_2/predict_object_report.md"),
              ("milestone_2/visual_vs_context.json", "Do signs that look alike behave alike?", "milestone_2/visual_vs_context_report.md"),
              ("milestone_4/overfitting.json", "What a blind search can reach", "milestone_4/overfitting_report.md"),
              ("milestone_4/lexicon.json", "Dictionary match rate", "milestone_4/lexicon_report.md"),
              ("milestone_4/linear_b.json", "Linear B as Greek: a real decipherment as control", "milestone_4/linear_b_report.md"),
              ("milestone_4/real_controls.json", "More controls on real scripts", "milestone_4/real_controls_report.md"),
              ("milestone_4/languages.json", "The same claim against four languages", "milestone_4/languages_report.md"),
              ("milestone_4/corpus_size.json", "How large would the corpus have to be?", "milestone_4/corpus_size_report.md")]


def extensions():
    L = ["## Exploratory extensions", "", "Not pre-registered. Each line is the result file's own reading.", ""]
    rx = load("milestone_3/restoration_extensions.json")
    if rx:
        L.append(f"- **Restoring damaged inscriptions** ([report](milestone_3/restoration_extensions_report.md)): two hidden signs both right "
                 f"{pct(rx['two_gaps']['all']['joint']['both_right'])} of the time with a joint search; a damaged stretch's length right "
                 f"{pct(rx['unknown_length']['length_right'])}; a miscopied sign located {pct(rx['miscopy']['position_found'])} of the time (chance {pct(rx['miscopy']['position_by_chance'])}).")
    for f, title, rep in EXTENSIONS:
        d = load(f)
        L.append(f"- **{title}** ([report]({rep})): {reading(d)}" if d else f"- {title}: `{f}` is missing, skipped.")
    return L + [""]


def findings():
    m2, m3, m4, ov, rc, rx = (load(p) for p in ("milestone_2/results.json", "milestone_3/results.json", "milestone_4/results.json",
                                                 "milestone_4/overfitting.json", "milestone_4/real_controls.json", "milestone_1/real/real_results.json"))
    out = []
    if m2:
        st = m2["structure_test"]
        out.append(f"Indus sign order is structured: neighbouring signs share {st['observed_bits']:.2f} bits of information against "
                   f"{st['shuffled_mean_bits']:.2f} when shuffled, and the layout is consistent with right-to-left reading.")
    if m3:
        r = m3["models"]
        out.append(f"A missing sign is guessed right {pct(r['kn_trigram']['top1'])} of the time by a trigram; neural models do not do better on a corpus this small.")
    if m4:
        c = m4["claims"]["yajnadevam-2026-07-15"]
        out.append(f"The one fully published decipherment key scores at percentile {c['nulls']['binned_10']['sequence']['percentile']:.0f} of shuffled "
                   f"versions of itself (Holm p {c['holm']['sequence']['binned_10']:.2f}): not distinguishable from chance.")
    if ov:
        out.append(f"A blind search passes the claimed key within about {ov['indus']['steps_until_median_search_passes_table']} swaps, so on this corpus "
                   "a good-reading table is easy to find by search alone.")
    if rc:
        out.append("The test finds true decipherments of Linear B and Ugaritic but also flags Linear A read as Greek, so passing it is not evidence for a language.")
    if rx:
        out.append(f"On real seal photographs the detector finds {pct(rx['with_real']['detection']['recall'][0])} of signs, but naming them remains the bottleneck.")
    return ["## What this project found", "", *out, ""]


if __name__ == "__main__":
    L = ["# Summary of results", "", "Built by `reports/build_summary.py` from the results files; rerun it after any milestone script.", "",
         *milestone1(), *milestone2(), *milestone3(), *milestone4(), *extensions(), *findings()]
    (R / "SUMMARY.md").write_text("\n".join(L))
    print(f"-> {R / 'SUMMARY.md'}")
