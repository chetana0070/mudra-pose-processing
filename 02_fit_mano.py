"""
Stage 2: Fit MANO pose parameters to the landmarks extracted in Stage 1.

For every landmark record with status == "ok":
  1. Load the 21 MediaPipe landmarks for each detected hand.
  2. Optimize MANO parameters (global orientation, per-joint pose, scale,
     translation) so the MANO model's projected 3D joints match the
     MediaPipe 2D landmarks (weak-perspective reprojection loss), with a
     small pose-prior regularizer to keep joints anatomically plausible.
  3. Save the fitted parameters per image.
After all images are fit, aggregate per mudra class (median pose across
all fitted samples of that class) into a single canonical_poses.json —
this is the "mudra name -> MANO pose" lookup table used in Stage 3.

REQUIRED — MANO model files (not auto-downloadable; licensed, requires
registration at https://mano.is.tue.mpg.de):
  models/mano/MANO_RIGHT.pkl
  models/mano/MANO_LEFT.pkl

Usage:
  python 02_fit_mano.py --landmarks_dir ../data/landmarks --mano_dir ../models/mano \
      --out_dir ../data/mano_fits
"""

import argparse
import json
from pathlib import Path

import numpy as np
import smplx
import torch
from tqdm import tqdm

# MediaPipe's 21-landmark hand topology, indices 0..20.
# MANO's joint tree uses a different ordering/count (16 joints + fingertips
# added via the joint regressor). We map the subset that corresponds
# directly to MANO's 16 main joints + wrist for the reprojection loss.
# This mapping follows the commonly used MediaPipe<->MANO correspondence.
MEDIAPIPE_TO_MANO_JOINT_MAP = {
    0: 0,    # wrist
    1: 13, 2: 14, 3: 15, 4: 16,      # thumb: CMC, MCP, IP, tip
    5: 1,  6: 2,  7: 3,  8: 17,      # index: MCP, PIP, DIP, tip
    9: 4,  10: 5, 11: 6, 12: 18,     # middle
    13: 10, 14: 11, 15: 12, 16: 19,  # ring
    17: 7, 18: 8, 19: 9, 20: 20,     # pinky
}


def load_mano_layer(mano_dir: str, is_right: bool, device):
    return smplx.create(
        model_path=mano_dir,
        model_type="mano",
        is_rhand=is_right,
        use_pca=False,       # use full 45-dim axis-angle pose, not the PCA subspace
        flat_hand_mean=True,
        num_pca_comps=45,
    ).to(device)


def fit_single_hand(mano_layer, target_2d, device, n_iters=300, lr=0.05):
    """
    target_2d: (21, 2) MediaPipe normalized image coords for one hand.
    Returns dict of fitted params + final reprojection loss.
    """
    target = torch.tensor(target_2d, dtype=torch.float32, device=device)

    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    hand_pose = torch.zeros(1, 45, device=device, requires_grad=True)
    scale = torch.tensor([1.0], device=device, requires_grad=True)
    translation = torch.zeros(1, 2, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([global_orient, hand_pose, scale, translation], lr=lr)

    mano_idx = list(MEDIAPIPE_TO_MANO_JOINT_MAP.values())
    mp_idx = list(MEDIAPIPE_TO_MANO_JOINT_MAP.keys())

    for _ in range(n_iters):
        optimizer.zero_grad()
        output = mano_layer(global_orient=global_orient, hand_pose=hand_pose)
        joints3d = output.joints[0]                      # (num_mano_joints, 3)

        # smplx MANO returns 16 joints, while MediaPipe has 21 landmarks.
        # Drop mapped MANO indices that are out of range, usually fingertip indices.
        valid_pairs = [(mp, mi) for mp, mi in zip(mp_idx, mano_idx) if mi < joints3d.shape[0]]
        if not valid_pairs:
            raise RuntimeError("No valid MediaPipe-to-MANO joint pairs after filtering.")

        mp_valid = [mp for mp, mi in valid_pairs]
        mano_valid = [mi for mp, mi in valid_pairs]

        proj = joints3d[mano_valid][:, :2] * scale + translation  # weak-perspective
        loss_data = torch.nn.functional.mse_loss(proj, target[mp_valid])
        loss_prior = 0.01 * (hand_pose ** 2).mean()       # keep pose near neutral
        loss = loss_data + loss_prior
        loss.backward()
        optimizer.step()

    return {
        "global_orient": global_orient.detach().cpu().numpy().tolist(),
        "hand_pose": hand_pose.detach().cpu().numpy().tolist(),
        "scale": float(scale.item()),
        "translation": translation.detach().cpu().numpy().tolist(),
        "final_loss": float(loss.item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks_dir", default="../data/landmarks")
    ap.add_argument("--mano_dir", default="../models/mano")
    ap.add_argument("--out_dir", default="../data/mano_fits")
    ap.add_argument("--n_iters", type=int, default=300)
    ap.add_argument("--loss_threshold", type=float, default=0.01,
                     help="Fits above this reprojection loss are excluded "
                          "from the canonical-pose aggregation as likely bad fits.")
    args = ap.parse_args()

    for required in ["MANO_RIGHT.pkl", "MANO_LEFT.pkl"]:
        if not (Path(args.mano_dir) / required).exists():
            raise FileNotFoundError(
                f"Missing {required} in {args.mano_dir}\n"
                "MANO model files require registration at "
                "https://mano.is.tue.mpg.de (free for research use) — "
                "download MANO_RIGHT.pkl and MANO_LEFT.pkl from there and "
                f"place them in {args.mano_dir}"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    landmarks_dir = Path(args.landmarks_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mano_right = load_mano_layer(args.mano_dir, is_right=True, device=device)
    mano_left = load_mano_layer(args.mano_dir, is_right=False, device=device)

    record_paths = sorted(landmarks_dir.rglob("*.json"))
    per_class_fits = {}  # mudra -> {"Right": [...poses...], "Left": [...poses...]}

    for rec_path in tqdm(record_paths, desc="Fitting MANO"):
        with open(rec_path) as f:
            record = json.load(f)

        if record["status"] != "ok":
            continue

        mudra = record["mudra"]
        per_class_fits.setdefault(mudra, {"Right": [], "Left": []})

        image_fit = {"image_path": record["image_path"], "hands": {}}

        for hand in record["hands"]:
            side = hand["handedness"]  # "Left" or "Right"
            layer = mano_right if side == "Right" else mano_left
            landmarks_2d = np.array([[p[0], p[1]] for p in hand["landmarks"]])

            fit = fit_single_hand(layer, landmarks_2d, device, n_iters=args.n_iters)
            image_fit["hands"][side] = fit

            if fit["final_loss"] <= args.loss_threshold:
                per_class_fits[mudra][side].append(fit["hand_pose"])

        out_path = out_dir / f"{Path(record['image_path']).stem}_fit.json"
        with open(out_path, "w") as f:
            json.dump(image_fit, f)

    # Aggregate per-mudra canonical pose = median across accepted fits.
    canonical = {}
    for mudra, sides in per_class_fits.items():
        canonical[mudra] = {}
        for side, poses in sides.items():
            if not poses:
                continue
            arr = np.array(poses)              # (N, 45)
            canonical[mudra][side] = {
                "hand_pose_median": np.median(arr, axis=0).tolist(),
                "hand_pose_std": np.std(arr, axis=0).tolist(),
                "n_samples": len(poses),
            }

    canonical_path = out_dir / "canonical_poses.json"
    with open(canonical_path, "w") as f:
        json.dump(canonical, f, indent=2)

    print(f"\nPer-image fits written to: {out_dir}")
    print(f"Canonical mudra->pose lookup table written to: {canonical_path}")
    for mudra, sides in canonical.items():
        n = {s: v["n_samples"] for s, v in sides.items()}
        print(f"  {mudra:20s}: {n}")


if __name__ == "__main__":
    main()
