# Benchmark Manifest Specification

Every benchmark run requires an explicit CSV manifest defining test and calibration samples.

## Manifest Columns

| Column | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `sample_id` | String | **Yes** | Unique identifier for the sample. Duplicate IDs fail validation. |
| `ground_truth` | String | **Yes** | Label: `live` / `bonafide` / `genuine` OR `spoof` / `attack` / `impostor`. |
| `image_path` | String | **Yes** | Absolute path or relative path to the image file. |
| `subject_id` | String | Recommended | Subject/User identity. Used for subject-level leakage prevention across splits. |
| `split` | String | Optional | Partition: `calibration`, `test`, or `train` (defaults to `test`). |
| `session_id` | String | Optional | Capture session identifier. |
| `finger_id` | String | Optional | Anatomical finger: `right_thumb`, `right_index`, `left_thumb`, etc. |
| `scenario` | String | Optional | Capture condition: `indoor`, `direct_sunlight`, `shadow`, `angled`. |
| `device` | String | Optional | Capture hardware: `iPhone 15 Pro`, `Samsung A53`, `OPPO A18`, `OEM_Sensor`. |
| `lighting` | String | Optional | Lighting condition: `ambient`, `flash`, `low_light`. |
| `spoof_type` | String | Optional | Attack type: `printed_photo`, `silicone_overlay`, `gelatin_pad`, `replay_screen`. |
| `material` | String | Optional | Physical spoof material: `latex`, `silicone`, `paper`, `conductive_film`. |
| `dataset_version` | String | Optional | Dataset version tag: `single_finger_test_v1`. |
| `source_dataset` | String | Optional | Origin collection name. |
| `notes` | String | Optional | Free-form diagnostic notes. |

## Example CSV Format

```csv
sample_id,subject_id,session_id,finger_id,scenario,split,ground_truth,image_path,device,spoof_type
S001,SUBJ_01,SES_01,right_index,indoor,calibration,live,datasets/synthetic_data/S001.jpg,Pixel_7,none
S002,SUBJ_01,SES_01,right_index,indoor,calibration,spoof,datasets/synthetic_data/S002.jpg,Pixel_7,silicone
S003,SUBJ_02,SES_01,right_index,sunlight,test,live,datasets/synthetic_data/S003.jpg,Samsung_A53,none
S004,SUBJ_02,SES_01,right_index,sunlight,test,spoof,datasets/synthetic_data/S004.jpg,Samsung_A53,print
```

## Scientific Partitioning Rules
1. **Subject Isolation:** Never allow the same `subject_id` to appear in both `calibration` and `test` splits.
2. **Held-Out Test Integrity:** Decision thresholds are tuned strictly on `calibration` and frozen before evaluating `test`.
3. **No Duplicate Hashes:** Identical image contents across splits will trigger a validation error.
