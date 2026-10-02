# Milestone 1 findings

Read with `milestone_1_report.md` (generated tables), `RUN_INFO.md` (machine and timings) and `figures/`. The run was commit `e0a9b50` on one RTX 5090, default settings.

**Scope:** Milestone 1 is complete on synthetic data only. Every score is on 400 held-out synthetic seals (1,464 signs). Nothing has been measured on real seals.

## Findings

1. **YOLOv8n and RT-DETR-L are tied.** AP50 is 0.995 for both, detection recall 0.999 for both, sign error rate 0.237 vs 0.238, exact sequence match 0.430 vs 0.443. Every difference is inside the 95% intervals. YOLOv8n trained in 1.7 minutes against 39.5 for RT-DETR and runs at 9 ms per image against 13, so it is the detector to keep.
2. **NMS did not drop touching signs.** YOLOv8n's recall on ground-truth signs that overlap a neighbour is 1.000. The original PRD chose RT-DETR on the expectation that NMS would delete adjacent signs; on this data it does not.
3. **The learned detectors beat classical segmentation by a wide margin.** The OpenCV baseline reaches recall 0.458 and gets the sign count right on 13.5% of seals, against 97.3% for YOLOv8n.
4. **The classifier is the bottleneck.** With near-perfect detection, 77% of detected signs get the right ID, which caps exact sequence match near 43%. On ground-truth boxes the classifier scores top-1 0.789 and top-5 0.949.
5. **Rare signs are weakest, but top-5 holds up.** Top-1 is 0.887 for the 50 most frequent signs, 0.739 for ranks 51-200 and 0.697 for rarer signs; top-5 is 0.992, 0.934 and 0.904. The correct sign is usually among the candidates, which is what Milestone 2's sequence rescoring needs.
6. **Confusions are between near-identical signs:** 342/343 (jar variants), 97/98 and 99/100 (stroke counts), 67/59 (fish variants). The classifier letterboxes each crop to a fixed size, which discards the relative size that separates some of these.
7. **Per-sign scores are thin.** 373 of the 417 signs occur in the test split; 43 of them have F1 of zero, mostly signs with one or two test crops.

## Real photographs (two Met photos, not scored)

Neither detector is usable on real seals.

- YOLOv8n drew two boxes on the carved seal, each covering two or three signs, and nothing on the impression.
- RT-DETR-L isolated one sign correctly on the impression (the triangle-shaped one), merged the others, and boxed the animal's horn as a sign on both objects.

Two photos cannot rank the detectors. They do confirm that scores on synthetic seals do not carry over: real signs are deep relief with shadows, and the generator draws thin engraved lines.

## What this means for the next milestones

- Milestones 2 to 4 run on the IM77 corpus text and do not depend on this pipeline.
- The synthetic sign error rate (about 0.24) is the only available operating point for Milestone 2's error-propagation study, and it is optimistic for real seals.
- To make the pipeline work on real photographs: labelled real seals (see `docs/milestone1_status.md`) or a generator that renders relief.
