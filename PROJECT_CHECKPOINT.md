# Mudra Pose Processing Pipeline

Built a Bharatanatyam mudra pose-processing pipeline using MediaPipe and MANO for structured hand-pose representation.

## Dataset
- Source: Bharatanatyam Mudra Dataset
- Total images processed: 28,431
- Mudra classes: 50

## Completed
- Downloaded and organized the Bharatanatyam Mudra Dataset.
- Created a Python 3.10 Conda environment for MediaPipe, OpenCV, PyTorch, SMPL-X, and MANO.
- Fixed legacy dependency issues with Chumpy by pinning NumPy to 1.23.5.
- Downloaded and configured the official MANO model files.
- Extracted MediaPipe hand landmarks from all dataset images.
- Patched MANO fitting to handle SMPL-X MANO's 16-joint output against MediaPipe's 21 hand landmarks.
- Generated MANO fit JSON files and canonical mudra pose embeddings.

## Path A: Strict Baseline
- Rule: accepted only images where detected hand count matched expected hand count.
- Valid MANO fit files: 15,562
- Output folder: `data/mano_fits_20iter_baseline/`
- Canonical pose file: `results/canonical_poses_20iter.json`
- Summary file: `results/mano_fit_summary_20iter_clean.csv`

## Path B: Improved Coverage
- Rule: accepted any image with at least one detected hand.
- Added `hand_count_quality` to preserve quality metadata:
  - `expected_match`
  - `partial_detection`
  - `extra_detection`
  - `not_applicable`
- Valid MANO fit files: 23,054
- Coverage improvement: +7,492 samples over Path A
- Output folder: `data/mano_fits_pathB_20iter/`
- Canonical pose file: `results/canonical_poses_pathB_20iter.json`
- Summary file: `results/mano_fit_summary_pathB_20iter.csv`

## Key Results
- Total images processed: 28,431
- Path A valid samples: 15,562
- Path B valid samples: 23,054
- Mudra classes represented: 50
- MANO fitting iterations: 20
- Path B improved usable sample coverage from 54.7% to 81.1%.

## Key Output Files
- `results/landmark_manifest.csv`
- `results/landmark_manifest_pathB.csv`
- `results/canonical_poses_20iter.json`
- `results/canonical_poses_pathB_20iter.json`
- `results/mano_fit_summary_20iter_clean.csv`
- `results/mano_fit_summary_pathB_20iter.csv`
- `results/README_results.md`
- `requirements_mudra.txt`

## Baseline Configuration
- Python 3.10
- NumPy 1.23.5
- OpenCV 4.8.1
- MediaPipe
- SMPL-X / MANO
- PyTorch CPU
- Chumpy 0.70
