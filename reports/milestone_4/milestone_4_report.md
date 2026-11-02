# Milestone 4 report: testing claimed decipherments against chance

A claimed decipherment of the Indus script is a table from signs to sounds. This harness applies a table to the inscriptions, measures how Sanskrit-like the result is, and compares that with thousands of tables made by shuffling the claim's own sound values among its signs. The rules are in `decipherment/prereg.json`. Version 1 was committed before any claim was scored; version 2 changed the primary score after a quick v1 run had scored the claims, as disclosed below (pre-registration SHA-256 `647c654ca242…`, commit `e3d9af1368`). **The test can only show that a particular table does or does not carry a signal beyond chance; it cannot show that the script is or is not a language.**

## Revision of the pre-registration

Version 1 (commit 9d89ed1) made 'Sanskrit minus Old Tamil' the primary score. Its first end-to-end run (a quick run, 300 shuffles, which also scored the two claims) showed that the positive control had 0% power against the value-shuffle null: Tamil script does not mark voicing, so the Old Tamil model makes any output rich in g, j, d, b look Sanskrit, whatever the sign order, and shuffled keys can score higher than the true key on real enciphered Sanskrit. Version 2 makes the primary score the Sanskrit trigram minus Sanskrit letter frequencies (order beyond letters), chosen because it had 100% power in a 10-replicate positive-control comparison of three candidate scores; the claims' scores were not used to choose it. The v1 score and the Naive Bayes score are still computed and reported, and all three are in the Holm family. Because v2 was fixed after the v1 quick run had scored the claims, this is disclosed in the report.

## Which claims can be tested

| Claim | Status | What was done |
|---|---|---|
| yajnadevam-2024-11-30 | `TESTED` | 1,334 texts, 5,682 signs scored; per-sign substitution; the claimant's conjunct and vowel choices are not applied (the collapsed consonant skeleton makes them irrelevant) |
| yajnadevam-2026-07-15 | `TESTED` | 1,318 texts, 5,584 signs scored; per-sign substitution; the claimant's conjunct and vowel choices are not applied (the collapsed consonant skeleton makes them irrelevant) |
| shaw-2026 | `NO_TARGET_LANGUAGE` | numbering test only |
| fuls-2024 | `NOT_RUN` | coverage only: no openly licensed Linear Elamite target corpus; coverage of the Wells corpus is reported only |
| bonta-2023 | `NOT_RUN` | key recoverable only from per-text readings that are not openly available |
| anonymous-614-page-dictionary | `NOT_RUN` | key recoverable only from per-text readings |
| parpola-1994 | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |
| mahadevan | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |
| rao-1982 | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |
| rajaram-jha-2000 | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |
| kak-1988-1990 | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |
| mukhopadhyay | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |
| horner-2026 | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |
| meilhac-2026 | `NOT_TESTABLE_NO_TABLE` | no fixed sign-to-value table is published |

## Can the harness detect a real decipherment at all?

Positive control: held-out Sanskrit text from the Digital Corpus of Sanskrit was enciphered with the Yajnadevam key's own sign-to-sound relation (many signs per sound) and cut to the lengths of real inscriptions; the true table was then tested exactly like a claim (20 replicates, 1,000 shuffles per null). It was detected (p < 0.05) in 100% of runs against the value-shuffle null and 100% against the frequency-binned null; with 30% of its values wrong, 100% and 100% (`power_curve.png`). So a negative result below is informative, with one caveat: the enciphered text comes from the same corpus as the scoring model (held-out chapters), so this power is an upper bound for a real Bronze Age language.

Negative control: 200 shuffled tables were treated as claims. 2% came out "significant" at 0.05 (5% expected) and the p-values are consistent with uniform (KS p = 0.45; `negative_control_p.png`), so the test does not fire falsely.

## Results

| Key | Score | Observed | Value-shuffle null: mean ± sd, p | Frequency-binned null: mean ± sd, p | Holm p (both nulls) |
|---|---|---|---|---|---|
| yajnadevam-2024-11-30 | Sanskrit sequence score (primary) | -0.295 | -0.329 ± 0.113, p = 0.3942 | -0.472 ± 0.109, p = 0.0611 | 1.0000, 0.6720 |
| yajnadevam-2024-11-30 | Sanskrit minus Old Tamil (v1 primary) | +4.587 | +6.241 ± 1.433, p = 0.8748 | +5.384 ± 1.039, p = 0.7714 | 1.0000, 1.0000 |
| yajnadevam-2024-11-30 | Naive Bayes, Sanskrit vs Old Tamil | +6.798 | +9.724 ± 2.425, p = 0.8873 | +7.446 ± 1.360, p = 0.5946 | 1.0000, 1.0000 |
| yajnadevam-2026-07-15 | Sanskrit sequence score (primary) | -0.299 | -0.353 ± 0.111, p = 0.3212 | -0.485 ± 0.112, p = 0.0544 | 1.0000, 0.6527 |
| yajnadevam-2026-07-15 | Sanskrit minus Old Tamil (v1 primary) | +4.670 | +6.400 ± 1.412, p = 0.8892 | +5.540 ± 1.035, p = 0.8067 | 1.0000, 1.0000 |
| yajnadevam-2026-07-15 | Naive Bayes, Sanskrit vs Old Tamil | +6.934 | +9.979 ± 2.397, p = 0.9015 | +7.713 ± 1.350, p = 0.6439 | 1.0000, 1.0000 |

All scores are bits per phoneme of the key's output (collapsed consonant skeleton); higher is more Sanskrit-like. The primary score asks whether the output has Sanskrit-like *order* beyond Sanskrit-like letters: log-probability under a Sanskrit character trigram minus log-probability under Sanskrit letter frequencies. p is the share of shuffled keys scoring at least as high (10,000 shuffles); Holm adjusts across the 12 pre-registered tests.

- **yajnadevam-2024-11-30**: not distinguishable from frequency-matched random tables under the Sanskrit sequence score (percentile 93.9, Holm p = 0.672); the harness detects true keys in 100% of positive-control runs at this length.
- **yajnadevam-2026-07-15**: not distinguishable from frequency-matched random tables under the Sanskrit sequence score (percentile 94.6, Holm p = 0.653); the harness detects true keys in 100% of positive-control runs at this length.

Sensitivity runs (pre-registered, outside the corrected family): with 5 frequency bins instead of 10 the primary score's p is 0.191 (yajnadevam-2024-11-30), 0.170 (yajnadevam-2026-07-15); with 20 bins 0.034 (yajnadevam-2024-11-30), 0.012 (yajnadevam-2026-07-15). Finer bins come closer to significance; none of these would survive the correction applied to the main tests.

- yajnadevam-2024-11-30, sign order: the real order scores -0.295; shuffling the signs inside each text gives -0.393 ± 0.011 (0% of shuffles score at least as high); reversing every text gives -0.438.
- yajnadevam-2026-07-15, sign order: the real order scores -0.299; shuffling the signs inside each text gives -0.398 ± 0.012 (0% of shuffles score at least as high); reversing every text gives -0.442.

Exploratory check, not pre-registered: the real sign order beats within-text shuffles for the claimed key (z = 8.4), but random tables made from the same values show order effects just as large (z = +0.3 ± 6.2; p = 0.114 for the claim). The Indus inscriptions have strong sign order of their own (Milestone 2), and almost any table carries some of it into its output.

Fitting check (current key): on the 1,267 texts the claimant read, the key scores +0.210 bits per phoneme relative to the 308 texts he left unread, after matching by length (p = 0.005). A key fitted to its own readings would do better on the read texts; that would be evidence of fitting, not of decipherment.

## The other keys

- Shaw 2026 has no target language, so only one thing can be checked: whether its values follow the sign numbering. Of 353 pairs of consecutively numbered signs, 298 share a vowel against 137 expected if the values were shuffled (p = 0.0001).
- Fuls 2024 gives values for 20 signs; they cover 8% of sign tokens in the Wells corpus and 2 whole texts. No openly licensed Linear Elamite corpus exists to score them against, so this claim is not run.

## How to read this

- Beating the frequency-binned null is necessary for a decipherment, not sufficient: it shows only that the table carries structure beyond putting common sounds on common signs.
- Every result is conditional on the scoring choices fixed in advance: the Digital Corpus of Sanskrit as the target, a consonant skeleton (which ignores the claimant's vowel and conjunct choices) and a character trigram. No corpus from 2600-1900 BCE exists, so the model is a proxy built from much later language.
- The two scores that compare Sanskrit with Old Tamil (the v1 primary and Naive Bayes) carry no evidence either way: in the positive control they detected the true table in 0% and 0% of runs against the value-shuffle null, because Tamil script does not mark voicing and the comparison rewards letters rather than order.
- The Yajnadevam test applies the table sign by sign. Plain substitution reproduces his own consonant skeleton in 43% of the 1,756 texts he reads, so this tests the table, not every per-text choice behind his readings.
- The proposal asked to "disprove" language claims. A permutation test cannot do that: it can only find that one specific table is or is not distinguishable from chance, with the power measured above.

## Data and licences

Sanskrit: Digital Corpus of Sanskrit, O. Hellwig (CC BY 4.0). Old Tamil: Project Madurai Sangam anthologies. Keys: Yajnadevam (github.com/yajnadevam/lipi, no licence: fetched at run time, never stored in this repository); Shaw 2026 and Fuls 2024 via the paragorth/Indus- transcriptions (cite the original works). Indus texts: yajnadevam/indus-website (GPL-3.0), Wells/ICIT numbering. Only aggregate numbers are published here.

Run time 5 minutes on a laptop CPU.
