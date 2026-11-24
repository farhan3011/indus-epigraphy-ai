# Hand-labelling guide: real Indus seal photographs

This guide is for whoever labels the photographs. It travels with the zip `indus_hand_labelling.zip`.

## Why this matters

Our sign-reading pipeline (Milestone 1) was trained only on computer-generated seals. On real photographs it fails: it merges neighbouring signs and boxes the animal. To fix that, and to measure honestly how well it works, we need real photographs where a person has marked every sign.

For each photograph you will do two things:

1. Draw a box around every sign.
2. Type that sign's number from the sign chart (Mahadevan's numbering, 1 to 417).

## What is in the zip

| Item | What it is |
|---|---|
| `images/` | 191 photographs to label. `wiki_*.jpg` (71) are openly licensed from Wikimedia Commons; `harappa_*.jpg` (120) are research-use-only photographs, mostly from Harappa.com. Gaps in the numbering are duplicates that were removed. |
| `sign_chart.pdf` | Every sign with its number. Page 1 is the 80 most common signs (84% of all signs written); the next pages list all 417 in number order. |
| `sign_chart_MIRRORED.pdf` | The same chart, mirrored. Use it only for photos of the carved seal stone itself (see step 3 below). |
| `labelling_config.xml` | The labelling screen layout. You paste it into Label Studio once. (Also kept in the repository as `grapheme_extraction/labelling_config.xml`.) |
| `example_labelling.png` | What a labelled sign looks like on screen. |
| `sources.csv` | Where each photo came from, its licence, and which photos show the same object. You do not need to edit it. |
| `HAND_LABELLING_GUIDE.md` | This guide. |

## Time needed

Roughly 900 signs across 191 photos. Expect 2 to 4 minutes per photo once you are used to it, so roughly 7 to 11 hours in total. Work in sessions; Label Studio saves every photo you submit.

## 1. Install Label Studio (once, about 10 minutes)

Label Studio is a free labelling program that runs in your web browser. Everything stays on your computer.

1. Install Python 3.10, 3.11 or 3.12 if you do not have it: <https://www.python.org/downloads/>
2. Open a terminal (Mac: Terminal; Windows: Command Prompt) and run:

   ```bash
   python3 -m venv labelling-env             # Windows: python -m venv labelling-env
   source labelling-env/bin/activate        # Windows: labelling-env\Scripts\activate
   pip install label-studio
   label-studio start
   ```

3. A browser tab opens at <http://localhost:8080>. Click **Sign up** and create an account. It is local to your computer; any email works.

Next time you only need the last line (after activating the environment again with the `source` or `activate` line).

## 2. Create the project (once)

1. Click **Create Project**, name it `Indus seals`.
2. Open the **Labeling Setup** tab, choose **Custom template** (bottom left), switch to **Code**, delete what is there, paste the whole content of `labelling_config.xml`, then **Save**.
3. Open **Data Import** and drag in the files from `images/`, at most 100 at a time: `wiki_*` first, then `harappa_*` in two batches.
4. Open any image to start labelling. Click the **settings** icon (sliders) at the bottom of the labelling screen and switch on:
   - **Select region after creating it** (so you can type the number straight away)
   - **Keep label selected after creating a region** (so you can draw box after box)

## 3. Label one photograph

![Example](example_labelling.png)

*An impression (the unicorn faces right), with the first two signs boxed: 391 (the wheel) and 180 (selected, its number in the right-hand panel).*

### Step A: say what the photo shows (right-hand panel)

This choice matters more than anything else. A carved seal is the mirror image of its impression, so a wrong choice gives every asymmetric sign the wrong number.

**The rule: look at which way the animal faces.**

| The animal faces | Choose | Chart to use |
|---|---|---|
| **right** (head on the right) | **impression**: a print in clay or plaster, or a modern print of a seal | `sign_chart.pdf` |
| **left** (head on the left) | **carved seal**: the seal stone itself | `sign_chart_MIRRORED.pdf` |

On Indus seals the animal faces right in the impression, so it faces left on the stone. Do not judge by whether the signs look raised or cut in: lighting in photographs makes cut-in signs look raised.

No animal (a tablet, a bar seal, an inscription close-up)? Use these clues:

- The very common "jar" sign 342 (a U with two little ears) usually ends a text. It sits at the **left** end of an impression and the **right** end of a carved seal.
- Check one asymmetric sign against both charts and see which one it matches.
- Still unsure? Choose **impression** and write "view unsure" in the notes.

Choose **skip** for photos with no legible sign, drawings rather than photos, display cases where the signs are too small to read, or photos that are far too blurred. Skipping is fine: a wrong label is worse than none.

A photo showing **both** a seal and its impression side by side: choose **impression** and label only the impression (the one whose animal faces right).

A photo showing **several separate seals or tablets**, all the same kind: label every one and also tick **several inscriptions in this photo**. If they are a mix of seals and impressions, label only the impressions.

### Step B: box every sign

Press **S** (or click the orange **sign** button above the photo), then drag a box around one sign.

- **Tight but complete.** The box touches the outermost strokes of the sign, nothing more.
- **One sign, one box.** A group of short parallel strokes standing together is **one** sign (for example `|||` is sign 89, four strokes `||||` is sign 104). Strokes stacked in two or three rows are also one sign (e.g. 112, 121).
- **Not signs:** the animal, the "standard" or incense-burner object in front of the unicorn, the border, cracks and chips. Do not box them.
- **Boxes may overlap** when signs touch. That is fine.
- **Damaged sign:** box the part you can see. If you cannot tell which sign it is, type `0`.
- Use the mouse wheel or the zoom buttons to zoom in on small signs.

### Step C: type the sign number

With the box selected, type its number in **"3. Sign number for the selected box"** and press Enter.

- Look at `sign_chart.pdf` page 1 first: most signs are among those 80.
- **Size matters.** Two short strokes (99) and two long strokes (87) are different signs, as are one short stroke (98) and one long stroke (86). Compare each sign with its neighbours. The chart draws all signs at one scale to help.
- Not sure? Type your best guess and tick **unsure**. That is far more useful than leaving it out.
- Cannot read it at all? Type `0`.
- To fix a box, click it to select it. Then drag its corners, change the number, or press Backspace to delete it.

### Step D: submit

Click **Submit** (or Ctrl+Enter / Cmd+Enter). Label Studio moves to the next photo. You can reopen any photo later from the list and click **Update** after changes.

The **notes** box is optional. Use it for a catalogue number if the photo's source shows one (for example `M-1186` or `H-95`), or for anything odd. Click **Add** under the notes box (or press Shift+Enter) or the note is not saved.

## 4. Checking your own work

Every 30 photos or so, reopen 3 at random and check:

- Every sign has a box, and nothing that is not a sign has one.
- Each box has a number (a box without a number cannot be used).
- The view is right: the animal faces right on an impression and left on a carved seal.

## 5. When you are done (or want to hand over progress)

1. In the project, click **Export**, choose **JSON**, and download the file.
2. Rename it to `indus_labels_<your name>_<date>.json` and send that one file back. Do not send the images; we already have them.

You can export at any point; a partial export is useful too. Every export must come from the same Label Studio project and contain all your work so far: the import on our side replaces the previous export, it does not add to it.

## 6. Licences: please read

- `wiki_*` photos carry open licences (public domain, CC0, CC BY or CC BY-SA), listed in `sources.csv`.
- `harappa_*` photos have **no licence**. We may use them for this student research project only. Do not post them online, put them in a public repository or share them outside the team.
- Your labels (boxes and numbers) are our own work and can be shared.

## What happens after you send the export (for the project team)

1. In the project repository, with the environment active, run:

   ```bash
   python grapheme_extraction/import_labels.py indus_labels_<name>_<date>.json --package path/to/indus_hand_labelling
   ```

   This writes impression-view images (carved-seal photos are mirrored automatically) to `data/real/images/`. It also writes one row per sign box to `data/real/annotations.csv`, and one row per photo with its split to `data/real/manifest.csv`. It prints a summary, lists problems such as boxes without numbers in `data/real/import_problems.txt`, and draws every box with its number into `data/real/check/` for spot-checking.
2. Splits are assigned once and then frozen. 40% of photos become the **test** set, which is never used for training or tuning, 15% become validation and the rest training. Photos of the same object (marked in `sources.csv` under `object_group`, plus any near-identical pair the script detects) share a split, and photos with several inscriptions go to training. Re-running the import with a later export keeps every split already assigned.
3. Then, as Milestone 1 work:
   - retrain the detector on the real training photos together with the synthetic seals;
   - add the real training crops to the classifier's reference bank (no retraining needed);
   - score the whole pipeline on the frozen real test set.

   These steps are not written yet; they are the next piece of code after the labels arrive.
4. Commit `data/real/annotations.csv` and `data/real/manifest.csv`. Never commit the images: `data/real/images/` is gitignored.
