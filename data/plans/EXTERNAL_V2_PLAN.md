# External validation v2 — pre-declared plan
Written 2026-10-05, before any re-annotation, re-processing or prediction on the v2 external clips.

## Why v2
The first external clips (v1, 64 sequences) were extracted with a different pipeline from the training data:
colour instead of grayscale, about 2x sharper, a different cropping script, and only the highest-confidence crab
saved in multi-crab frames. The motion input was therefore out of the training range
(frame-difference energy for feeding: 30.6 external vs 11.9 training). v1 results are kept and reported, not replaced.

## Data
- Source: raw external videos MVI_0801 – MVI_0890 (independent recording session, no overlap with training videos).
- Annotation: every usable bout in every video, labelled from the video only, without seeing any model output,
  using External_bout_annotation_template.xlsx (definitions fixed in its Instructions sheet).
- Second annotator labels a random 20% of bouts; agreement (Cohen's kappa) reported.
- Cutting: cut_external_clips_colab.py (time trim + single-crab spatial box, libx264 crf 18, no audio).
- Processing: Video_Preprocessing_Script.ipynb UNCHANGED (YOLOv11 weights-yolov11-model-53.pt, grayscale 640 input,
  conf 0.5, BoT-SORT, 10% padding, INTER_AREA 224, grayscale output). Only input/output folders and zip name changed.

## QC
Blind contact sheets with the same criteria C1–C4 as v1, scored before any prediction exists.
Exclusion list locked (with scorer initials and date) before inference.

## Models
Exactly the files in LOCK.json (locked 2026-10-05T08:19:44 UTC). No retraining, no threshold change.
Inference preprocessing identical to training: 24 evenly spaced frames, resize 224, no blur or resolution matching.

## Analyses (all reported)
- Sets: all v2 clips; clips passing blind QC.
- Activity: aug (primary), no-aug (secondary), waving if mean p > 0.5.
- Sex: DINOv2 + logistic regression, male if p > 0.5; also on feeding clips only.
- Metrics: accuracy, balanced accuracy, per-class recall, majority baseline, AUC,
  95% CI by bootstrapping videos (2000 resamples, seed 0).
- Run once. Any later analysis is labelled post hoc.
