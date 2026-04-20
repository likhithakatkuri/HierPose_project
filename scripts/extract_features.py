#!/usr/bin/env python
"""Pre-extract and cache features for faster training."""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--cache_dir", default="cache/features")
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    from psrn.data.jhmdb_loader import JHMDBSplitLoader
    from psrn.features.extractor import HierarchicalFeatureExtractor, FeatureConfig
    from psrn.data.augmentation import KeypointAugmenter, AugConfig

    loader = JHMDBSplitLoader(args.data_root, args.split)
    train_samples = loader.get_train_samples()
    test_samples = loader.get_test_samples()

    extractor = HierarchicalFeatureExtractor(FeatureConfig())
    augmenter = KeypointAugmenter(AugConfig()) if args.augment else None

    print(
        f"Extracting features for {len(train_samples)} train, "
        f"{len(test_samples)} test samples..."
    )

    X_train, y_train, names_train = extractor.extract_batch(
        train_samples, args.cache_dir, augmenter
    )
    X_test, y_test, names_test = extractor.extract_batch(
        test_samples, args.cache_dir, None
    )

    extractor.save_features_npz(
        args.cache_dir + "/train.npz", X_train, y_train, names_train
    )
    extractor.save_features_npz(
        args.cache_dir + "/test.npz", X_test, y_test, names_test
    )

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Saved to {args.cache_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
