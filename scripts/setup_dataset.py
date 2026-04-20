"""
One-shot dataset setup script.

What this does:
  1. Reads JHMDB-GT.pkl  → extracts class labels + train/test splits
  2. Walks Frames/        → finds all video clips
  3. Runs MediaPipe Pose on every frame → extracts 15 skeleton joints
  4. Saves joints as .npy files  (same (T,15,2) format as .mat files)
  5. Writes split .txt files  (same format as official JHMDB splits)

After running this, the full training pipeline works as-is.

Usage:
    # First install mediapipe:
    pip install mediapipe opencv-python

    python scripts/setup_dataset.py \
        --jhmdb_root "C:/Users/likhi/Downloads/JHMDB.tar/JHMDB" \
        --out_root   "data/JHMDB"
"""

import argparse
import pickle
import sys
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── MediaPipe → JHMDB 15-joint mapping ─────────────────────────────────────
# JHMDB joints: neck=0,belly=1,face=2,r_sho=3,l_sho=4,r_hip=5,l_hip=6,
#               r_elb=7,l_elb=8,r_kn=9,l_kn=10,r_wrist=11,l_wrist=12,
#               r_ank=13,l_ank=14
# MediaPipe 33-landmark indices used below
MP_TO_JHMDB = {
    0:  [9, 10],   # neck      → avg(left_shoulder=11, right_shoulder=12) ... use ears avg
    1:  [23, 24],  # belly     → avg(left_hip=23, right_hip=24)
    2:  [0],       # face      → nose=0
    3:  [12],      # r_shoulder
    4:  [11],      # l_shoulder
    5:  [24],      # r_hip
    6:  [23],      # l_hip
    7:  [14],      # r_elbow
    8:  [13],      # l_elbow
    9:  [26],      # r_knee
    10: [25],      # l_knee
    11: [16],      # r_wrist
    12: [15],      # l_wrist
    13: [28],      # r_ankle
    14: [27],      # l_ankle
}
# Override neck: average of left_shoulder(11) and right_shoulder(12)
MP_TO_JHMDB[0] = [11, 12]


def landmarks_to_jhmdb(landmarks, img_w: int, img_h: int) -> np.ndarray:
    """Convert MediaPipe NormalizedLandmarkList → (15, 2) pixel coords."""
    lm = landmarks.landmark
    joints = np.zeros((15, 2), dtype=np.float32)
    for j_idx, mp_indices in MP_TO_JHMDB.items():
        xs = [lm[i].x * img_w for i in mp_indices]
        ys = [lm[i].y * img_h for i in mp_indices]
        joints[j_idx, 0] = float(np.mean(xs))
        joints[j_idx, 1] = float(np.mean(ys))
    return joints


def extract_joints_from_dir(frame_dir: Path, pose_estimator) -> np.ndarray | None:
    """Run MediaPipe on all frames in a directory. Returns (T, 15, 2) or None."""
    import cv2

    frames = sorted(frame_dir.glob("*.png")) + sorted(frame_dir.glob("*.jpg"))
    if not frames:
        return None

    joint_seq = []
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose_estimator.process(img_rgb)
        if result.pose_landmarks:
            h, w = img.shape[:2]
            joints = landmarks_to_jhmdb(result.pose_landmarks, w, h)
            joint_seq.append(joints)

    if len(joint_seq) < 2:
        return None

    return np.stack(joint_seq, axis=0).astype(np.float32)  # (T, 15, 2)


def parse_gt_pkl(pkl_path: Path):
    """
    Parse JHMDB-GT.pkl.
    Returns: labels list, train_videos list-of-sets, test_videos list-of-sets
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    log.info("PKL keys: %s", list(data.keys()))

    labels = data.get("labels", [])
    log.info("Labels (%d): %s ...", len(labels), labels[:5])

    # train_videos / test_videos are lists (one per split)
    train_vids = data.get("train_videos", [])
    test_vids  = data.get("test_videos",  [])

    log.info("Splits found: %d", len(train_vids))
    for i, (tr, te) in enumerate(zip(train_vids, test_vids)):
        log.info("  split %d: %d train, %d test", i + 1, len(tr), len(te))

    return labels, train_vids, test_vids


def write_split_files(train_vids, test_vids, out_splits_dir: Path):
    """Generate split .txt files from PKL data."""
    out_splits_dir.mkdir(parents=True, exist_ok=True)

    for split_idx, (train_set, test_set) in enumerate(zip(train_vids, test_vids)):
        split_num = split_idx + 1

        class_train = defaultdict(list)
        class_test  = defaultdict(list)

        for vpath in train_set:
            parts = str(vpath).replace("\\", "/").split("/")
            cls   = parts[0] if len(parts) >= 2 else "unknown"
            vname = parts[1] if len(parts) >= 2 else parts[0]
            # strip .avi / .mp4 if present
            vname = vname.replace(".avi", "").replace(".mp4", "")
            class_train[cls].append(vname)

        for vpath in test_set:
            parts = str(vpath).replace("\\", "/").split("/")
            cls   = parts[0] if len(parts) >= 2 else "unknown"
            vname = parts[1] if len(parts) >= 2 else parts[0]
            vname = vname.replace(".avi", "").replace(".mp4", "")
            class_test[cls].append(vname)

        all_classes = sorted(set(list(class_train) + list(class_test)))
        for cls in all_classes:
            fname = out_splits_dir / f"{cls}_test_split{split_num}.txt"
            lines = [f"{v} 1" for v in class_train[cls]] + \
                    [f"{v} 2" for v in class_test[cls]]
            fname.write_text("\n".join(lines) + "\n")

        log.info("Split %d: wrote %d class files", split_num, len(all_classes))

    log.info("Split files written to %s", out_splits_dir)


def extract_all_joints(frames_root: Path, joints_out_root: Path, pose_estimator):
    """
    Walk Frames/<class>/<video>/ and extract joints.
    Saves as joints_out_root/<class>/<video>/joint_positions.npy
    """
    joints_out_root.mkdir(parents=True, exist_ok=True)

    action_dirs = sorted([d for d in frames_root.iterdir() if d.is_dir()])
    total = sum(len(list(ad.iterdir())) for ad in action_dirs if ad.is_dir())
    done = 0

    for action_dir in action_dirs:
        cls = action_dir.name
        for video_dir in sorted(action_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            out_dir  = joints_out_root / cls / video_dir.name
            out_file = out_dir / "joint_positions.npy"

            if out_file.exists():
                done += 1
                continue

            joints = extract_joints_from_dir(video_dir, pose_estimator)
            if joints is None:
                log.warning("  SKIP %s/%s — no poses detected", cls, video_dir.name)
                done += 1
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_file), joints)
            done += 1

            if done % 50 == 0:
                log.info("  Progress: %d/%d clips", done, total)

    log.info("Joint extraction complete. Saved to %s", joints_out_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jhmdb_root",
        required=True,
        help="Path to extracted JHMDB folder containing Frames/ and JHMDB-GT.pkl",
    )
    parser.add_argument(
        "--out_root",
        default="data/JHMDB",
        help="Output directory (will become data_root for training)",
    )
    parser.add_argument(
        "--splits_only",
        action="store_true",
        help="Only generate split .txt files (skip MediaPipe extraction)",
    )
    parser.add_argument(
        "--mediapipe_complexity",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="MediaPipe model complexity (0=fast, 1=balanced, 2=accurate)",
    )
    args = parser.parse_args()

    jhmdb_root = Path(args.jhmdb_root)
    out_root   = Path(args.out_root)

    pkl_path    = jhmdb_root / "JHMDB-GT.pkl"
    frames_root = jhmdb_root / "Frames"

    if not pkl_path.exists():
        log.error("JHMDB-GT.pkl not found at %s", pkl_path)
        sys.exit(1)

    # ── Step 1: parse PKL ──────────────────────────────────────────────────
    log.info("=== Step 1: Parsing JHMDB-GT.pkl ===")
    labels, train_vids, test_vids = parse_gt_pkl(pkl_path)

    # ── Step 2: write split files ──────────────────────────────────────────
    log.info("=== Step 2: Writing split .txt files ===")
    write_split_files(train_vids, test_vids, out_root / "splits")

    if args.splits_only:
        log.info("--splits_only flag set, skipping pose extraction.")
        return

    # ── Step 3: MediaPipe extraction ───────────────────────────────────────
    log.info("=== Step 3: Extracting poses with MediaPipe ===")

    try:
        import mediapipe as mp
        import cv2  # noqa: F401
    except ImportError:
        log.error(
            "mediapipe or opencv-python not installed.\n"
            "Run:  pip install mediapipe opencv-python\n"
            "Or use --splits_only if you only need split files."
        )
        sys.exit(1)

    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=args.mediapipe_complexity,
        min_detection_confidence=0.3,
    ) as pose:
        extract_all_joints(
            frames_root=frames_root,
            joints_out_root=out_root / "joint_positions",
            pose_estimator=pose,
        )

    log.info("\n=== Setup Complete ===")
    log.info("Dataset ready at: %s", out_root)
    log.info("Now run:")
    log.info("  python -m psrn.train_ml --data_root %s --split 1 --output_dir outputs/exp1", out_root)


if __name__ == "__main__":
    main()
