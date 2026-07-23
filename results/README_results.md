# Mudra MANO Fitting Results - 20 Iteration Baseline

## Dataset
- Source: Bharatanatyam Mudra Dataset
- Total images processed: 28,431
- Mudra classes: 50

## Stage 1: Landmark Extraction
- Tool: MediaPipe Hand Landmarker
- Output: `landmark_manifest.csv`
- Valid landmark samples: 15,562

## Stage 2: MANO Fitting
- Tool: SMPL-X MANO model
- Optimization: 20 iterations per sample
- Output folder: `data/mano_fits_20iter_baseline/`
- Total MANO fit JSON files: 15,562
- Canonical pose table: `canonical_poses_20iter.json`

## Final Result Files
- `landmark_manifest.csv`: per-image landmark extraction status
- `canonical_poses_20iter.json`: canonical mudra pose lookup table
- `mano_fit_summary_20iter_clean.csv`: per-class MANO fit summary

## Engineering Notes
- Patched MANO fitting to handle SMPL-X MANO's 16-joint output against MediaPipe's 21 hand landmarks.
- Filtered invalid MANO joint indices during reprojection loss calculation.
- Two-hand mudras may have more hand instances than fit files because each image can contain both left and right hand fits.
