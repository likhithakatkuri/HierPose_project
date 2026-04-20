"""
Run this FIRST after extracting annotations.tar.gz.
It reads JHMDB-GT.pkl and shows what's inside,
then generates the split .txt files our loader expects.

Usage:
    python scripts/check_pkl.py --pkl annotations/JHMDB-GT.pkl --out data/JHMDB/splits/
"""
import pickle
import argparse
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to JHMDB-GT.pkl")
    parser.add_argument("--out", default="data/JHMDB/splits/", help="Output splits dir")
    parser.add_argument("--inspect", action="store_true", help="Just inspect, don't write")
    args = parser.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print(f"ERROR: {pkl_path} not found")
        return

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    print(f"\n=== JHMDB-GT.pkl contents ===")
    print(f"Type: {type(data)}")

    if isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for k, v in data.items():
            if hasattr(v, '__len__'):
                print(f"  {k}: {type(v).__name__} len={len(v)}")
            else:
                print(f"  {k}: {v}")

        # Try to find video names and labels
        # Common keys: 'gttubes', 'nframes', 'labels', 'train_videos', 'test_videos'
        if "labels" in data:
            print(f"\nLabels ({len(data['labels'])}): {data['labels'][:5]}...")

        if "train_videos" in data:
            tv = data["train_videos"]
            print(f"\ntrain_videos type: {type(tv)}")
            if isinstance(tv, (list, tuple)):
                print(f"  splits: {len(tv)}")
                for i, split in enumerate(tv):
                    print(f"  split {i+1}: {len(split)} videos, first={list(split)[:2]}")

        if "test_videos" in data:
            tv = data["test_videos"]
            if isinstance(tv, (list, tuple)):
                for i, split in enumerate(tv):
                    print(f"  test split {i+1}: {len(split)} videos")

        if args.inspect:
            return

        # Generate split txt files
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        train_vids = data.get("train_videos", [])
        test_vids = data.get("test_videos", [])
        labels = data.get("labels", [])

        if not train_vids:
            print("\nCould not find train_videos key — showing all keys for manual inspection:")
            for k in data:
                print(f"  {k}: {type(data[k])}")
            return

        # Build per-class split files
        # video names are typically like "brush_hair/April_09_brush_hair_..."
        for split_idx in range(len(train_vids)):
            split_num = split_idx + 1
            # collect all videos for this split
            class_train = defaultdict(list)
            class_test = defaultdict(list)

            for vpath in train_vids[split_idx]:
                parts = str(vpath).replace("\\", "/").split("/")
                if len(parts) >= 2:
                    cls, vname = parts[0], parts[1]
                else:
                    cls, vname = "unknown", parts[0]
                class_train[cls].append(vname)

            for vpath in test_vids[split_idx]:
                parts = str(vpath).replace("\\", "/").split("/")
                if len(parts) >= 2:
                    cls, vname = parts[0], parts[1]
                else:
                    cls, vname = "unknown", parts[0]
                class_test[cls].append(vname)

            # Write one file per class
            all_classes = set(list(class_train.keys()) + list(class_test.keys()))
            for cls in sorted(all_classes):
                fname = out_dir / f"{cls}_test_split{split_num}.txt"
                lines = []
                for vname in class_train[cls]:
                    lines.append(f"{vname} 1")
                for vname in class_test[cls]:
                    lines.append(f"{vname} 2")
                fname.write_text("\n".join(lines) + "\n")
                print(f"  Written: {fname.name}  ({len(class_train[cls])} train, {len(class_test[cls])} test)")

        print(f"\nDone! Split files written to: {out_dir}")

    else:
        print(f"Unexpected type {type(data)} — first item: {str(data)[:200]}")


if __name__ == "__main__":
    main()
