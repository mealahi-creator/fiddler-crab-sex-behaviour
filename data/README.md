# Data files

Labels: `sex` = male / female; `activity` = feeding / waving. Probabilities: `p_waving` = predicted probability of waving, `p_male` = predicted probability of male. A sequence is assigned to the positive class when the probability is greater than 0.5.

## predictions/ — cross-validation (n = 278 sequences)

| File | Model | Columns |
|---|---|---|
| `cv_r3d18.csv` | R3D-18 (Kinetics-400 pretrained), two seeds averaged | `seq`, `video` (source video = cross-validation group), `sex`, `activity`, `fold` (0–4, test fold), `p_waving`, `p_male` |
| `cv_effb0_bilstm.csv` | EfficientNet-B0 + BiLSTM (ImageNet pretrained), two seeds averaged | as above |
| `raw_twostream/noaug_pred_fold{k}_seed{s}.csv` | Two-stream 3D CNN, one file per test fold and seed | as above + `seed` |
| `raw_twostream/resaug_pred_fold{k}_seed{s}.csv` | Two-stream 3D CNN with resolution augmentation | as above + `seed` |
| `cv_sex_image_encoders.csv` | Sex classifiers on frozen image features: 2 encoders (`backbone` = dinov2_s, convnext_t) × 2 crops (`crop` = full, tight = central 70%) × 2 training sets (`train_on` = all, feeding_only) | `seq`, `video`, `crab_track` (video + tracker ID), `sex`, `activity`, `fold`, `p_male_seq` (sequence probability = mean over 8 frames), `p_male_crab` (mean over sequences of the same track; not used in the paper), `backbone`, `crop`, `train_on` |

The model reported as **DINOv2 + LR** is `backbone = dinov2_s`, `crop = full`, `train_on = feeding_only`.

## predictions/ — external tests (locked models)

| File | Test | Notes |
|---|---|---|
| `external_E1_primary_71clips.csv` | Pre-declared test on session E1 (Table 8, Figure 8) | `seq_name`, `sex`, `activity`, `video`, `n_frames`, `blind_exclude` (excluded by the blind quality-control scorer), `p_waving_aug`, `p_waving_noaug` (two-stream with / without resolution augmentation), `p_male` (DINOv2 + LR) |
| `external_secondary_90clips.csv` | Secondary activity analysis, sessions E1 + E2 (Tables 8, S1; Figures S2, S4) | as above + `source` (E1 = Feb2026_session, E2 = Aug2025_phone), `low_fps` (real frame rate below 25 fps), `p_waving_r3d18` (locked R3D-18) |
| `external_E1_mismatched_64clips.csv` | First run on E1 with clips extracted differently from the training data (Table S2, Figure S3) | `qc_id`, `strict_exclude` additionally |

## tracking/

`tracking_log_training.csv` (all tracked sequences, including some removed before the final dataset) and `tracking_log_external.csv` (90 external clips): one row per frame of each sequence.
`state` = `tracked` (detected directly), `interpolated` (box extrapolated from the last position and velocity, gaps of up to 5 frames), `recovered` (last box of the most confident track reused); `confidence` is reduced by 0.1 per extrapolated frame and by 0.2 for reused boxes. `xmin`–`ymax` are box coordinates in pixels of the original video frame.

## features/

`image_encoder_features.npz`: `seq` (278 sequence names, same order as the arrays) and four float16 arrays of shape (278 sequences, 8 frames, D): `dinov2_s__full`, `dinov2_s__tight` (D = 384), `convnext_t__full`, `convnext_t__tight` (D = 768).

## detector/

`detector_comparison.csv`: validation mAP@50, precision and recall of YOLOv11, Roboflow 3.0 Object Detection and RF-DETR trained on the same 1,670 annotated images (640 × 640 grayscale; 65/25/10 split), from the detector training output.

## plans/

Pre-declared analysis plans written before predictions were made (`plan_external_*.json`, `EXTERNAL_V2_PLAN.md`) and model lock files with the SHA-256 hash of every locked model file (`LOCK_twostream_dinov2.json`, `LOCK_r3d18.json`).
