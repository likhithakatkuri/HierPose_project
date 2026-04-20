"""HierPose inference package.

Provides end-to-end inference utilities:
    - HierPosePredictor  : load a trained model and run predictions with
                           optional SHAP explanation and counterfactual guidance.
    - MediaPipeExtractor : extract JHMDB-compatible (T, 15, 2) keypoint arrays
                           from video files or a live webcam feed.

Example usage::

    from psrn.inference import HierPosePredictor, MediaPipeExtractor

    extractor = MediaPipeExtractor()
    frames = extractor.extract_from_video("clip.mp4")   # (T, 15, 2)

    predictor = HierPosePredictor(
        model_path="models/lgbm_model.joblib",
        scaler_path="models/scaler.joblib",
    )
    predictor.load()

    result = predictor.predict_frames(frames)
    print(result.predicted_class, result.confidence)
"""

from psrn.inference.predictor import HierPosePredictor
from psrn.inference.mediapipe_extractor import MediaPipeExtractor

__all__ = [
    "HierPosePredictor",
    "MediaPipeExtractor",
]
