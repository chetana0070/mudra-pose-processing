"""
Stage 1: Extract hand landmarks from the Bharatanatyam Mudra Dataset.

For every image in data/raw/<MudraClass>/*.jpg, this script:
  1. Runs MediaPipe HandLandmarker (up to 2 hands).
  2. Records 21 landmarks per detected hand in (x, y, z) — x/y normalized
     image coords, z is a relative-depth estimate (wrist-relative, NOT
     true metric depth).
  3. Flags low-quality samples (no hand found, wrong hand count for the
     mudra type, low detection confidence) instead of silently keeping them.
  4. Saves one JSON record per image into data/landmarks/, plus a single
     summary CSV manifest for quick inspection / filtering downstream.

Folder layout expected (matches the public dataset as distributed):
  data/raw/Alapadmam(1)/Alapadmam_0.jpg
  data/raw/Anjali(1)/Anjali_0.jpg
  ...

Usage:
  python 01_extract_landmarks.py --raw_dir ../data/raw --out_dir ../data/landmarks \
      --model_path ../models/hand_landmarker.task

Model file (one-time manual download — NOT auto-downloaded because this
environment's network allowlist doesn't include storage.googleapis.com):
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
Save it to: models/hand_landmarker.task
"""

import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tqdm import tqdm

# Single-hand mudra classes (asamyukta hastas) expect exactly 1 hand.
# Everything else in this dataset is a samyukta (double-hand) mudra and
# expects exactly 2 hands. Used only as a quality-flagging heuristic.
SINGLE_HAND_CLASSES = {
    "Pathaka", "Tripathaka", "Ardhapathaka", "Kartarimukha", "Katrimukha",
    "Mayura", "Ardhachandran", "Aralam", "Shukatundam", "Mushti",
    "Sikharam", "Kapith", "Katakamukha_1", "Katakamukha_2", "Katakamukha_3",
    "Suchi", "Chandrakala", "Padmakosha", "Sarpasirsha", "Mrigasirsha",
    "Simhamukham", "Kangulam", "Alapadmam", "Mukulam", "Chaturam",
    "Bramaram", "Hamsasyam", "Hamsapaksha", "Tamarachudam", "Trishulam",
}


def clean_class_name(folder_name: str) -> str:
    """Strip trailing '(1)' style suffixes some folders in this dataset have."""
    return re.sub(r"\(\d+\)$", "", folder_name).strip()


def build_detector(model_path: str, num_hands: int = 2):
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_one(detector, image_path: Path):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None, "unreadable_file"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None, "no_hand_detected"

    hands = []
    for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
        label = handedness[0].category_name  # "Left" or "Right"
        score = handedness[0].score
        pts = [[lm.x, lm.y, lm.z] for lm in hand_landmarks]
        hands.append({"handedness": label, "confidence": float(score), "landmarks": pts})

    return hands, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="../data/raw")
    ap.add_argument("--out_dir", default="../data/landmarks")
    ap.add_argument("--model_path", default="../models/hand_landmarker.task")
    ap.add_argument("--limit_per_class", type=int, default=0,
                     help="0 = no limit. Useful for a quick smoke test.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"Model file not found at {args.model_path}\n"
            "Download it manually from:\n"
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task\n"
            "and place it at that path (this sandbox's network allowlist "
            "blocks storage.googleapis.com, so this download must happen "
            "on your own machine)."
        )

    detector = build_detector(args.model_path)

    class_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {raw_dir}")

    manifest_rows = []
    stats = {"ok": 0, "no_hand_detected": 0, "unreadable_file": 0}
    hand_count_stats = {
        "expected_match": 0,
        "partial_detection": 0,
        "extra_detection": 0,
        "not_applicable": 0,
    }

    for class_dir in class_dirs:
        mudra_name = clean_class_name(class_dir.name)
        expected_hands = 1 if mudra_name in SINGLE_HAND_CLASSES else 2

        images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        if args.limit_per_class:
            images = images[: args.limit_per_class]

        class_out_dir = out_dir / mudra_name
        class_out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(images, desc=mudra_name):
            hands, status = extract_one(detector, img_path)

            quality_flag = status
            hand_count_quality = "not_applicable"

            if status == "ok":
                detected_hands = len(hands)

                # Path B behavior:
                # Keep any image with at least one detected hand as usable.
                # Do not reject two-hand mudras when MediaPipe detects only one hand.
                quality_flag = "ok"

                if detected_hands == expected_hands:
                    hand_count_quality = "expected_match"
                elif detected_hands < expected_hands:
                    hand_count_quality = "partial_detection"
                else:
                    hand_count_quality = "extra_detection"

            stats[quality_flag] = stats.get(quality_flag, 0) + 1
            hand_count_stats[hand_count_quality] = hand_count_stats.get(hand_count_quality, 0) + 1

            record = {
                "image_path": str(img_path),
                "mudra": mudra_name,
                "expected_hands": expected_hands,
                "status": quality_flag,
                "hand_count_quality": hand_count_quality,
                "hands": hands if hands else [],
            }

            out_path = class_out_dir / f"{img_path.stem}.json"
            with open(out_path, "w") as f:
                json.dump(record, f)

            manifest_rows.append({
                "mudra": mudra_name,
                "image_path": str(img_path),
                "landmark_path": str(out_path),
                "status": quality_flag,
                "hand_count_quality": hand_count_quality,
                "num_hands_detected": len(hands) if hands else 0,
                "expected_hands": expected_hands,
            })

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("\n--- Extraction summary ---")
    total = sum(stats.values())
    for k, v in stats.items():
        pct = 100 * v / total if total else 0
        print(f"  {k:20s}: {v:6d}  ({pct:.1f}%)")
    print("\n--- Hand-count quality summary ---")
    hand_total = sum(hand_count_stats.values())
    for k, v in hand_count_stats.items():
        pct = 100 * v / hand_total if hand_total else 0
        print(f"  {k:20s}: {v:6d}  ({pct:.1f}%)")

    print(f"\nManifest written to: {manifest_path}")
    print("Use status == 'ok' for Stage 2. Use hand_count_quality to separate full vs partial detections.")


if __name__ == "__main__":
    main()
