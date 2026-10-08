# pad_v0: first person-split PAD test set

## Setup (no config changes beyond the manifest path)

1. Unzip so you get `common-pipeline/datasets/manifests/pad_v0/`.
2. In the config:
   ```yaml
   data:
     manifest_path: datasets/manifests/pad_v0/pad_v0_single_finger.csv
     calibration_split_name: calibration
     test_split_name: test
     verify_hashes: true
   ```
   Image paths are relative (`images/SF_0001.jpg`). The harness resolves them from the manifest's folder, so `base_image_dir` is not needed.

This was checked with the harness's own `BenchmarkValidator`: 0 errors on files, duplicate hashes and calibration/test leakage.

## Contents

| File | Rows | Live | Screen replay | Split |
|---|---|---|---|---|
| `pad_v0_single_finger.csv` | 374 | 142 | 232 | calibration (18 persons) and test (26 persons) |
| `pad_v0_slap.csv` | 21 | 4 | 17 | test only: too little to calibrate |

- `subject_id` is an anonymous real-person ID (P001–P044).
- For a replay, `source_person_id` is the person whose finger is shown on the screen. Every replay goes to the same split as that person.
- 174 replays could not be traced to a source person (`UNKNOWN`), so they are test-only.
- Images are renamed to their `sample_id`. No names appear in files or paths.

## Read before trusting any number

1. **This data was used to train DualScalePatchNet.** It is the same `live` + `livesness` data behind `screen_paper_pad_manifest.csv`. Its scores here will look high and prove nothing. This run proves the tool works end to end with the real model, and that its scores match the original eval script on the same images.
2. **Live and fake are captured differently.** Live images are 224×224 pre-cropped fingertips. Replays are full WhatsApp-compressed photos. A model can separate them without detecting liveness at all.
3. **Duplicates across person labels.** The original data had the same image under different person labels (for example, three labels for one person). These were merged into 44 real persons, and 46 duplicate images were removed. This also means the earlier 5-fold CV was leaking.
4. **Screen replay is the only attack type.** There are no paper prints and no 3D fakes.
5. **Slap:**
   - Rows SL_0001–SL_0004 are marked live. They were in the screen-replay folder but look like real hands; this is pending confirmation.
   - 6 slap fakes were HEIC and have been converted to JPG.

## Wanted from the run (per model, test split)

- APCER for screen replay
- BPCER
- AUC
- calibration AUC
- the threshold chosen on the calibration split

The next version (pad_v1) adds in-house live captures and vendor attack data, using the same columns.
