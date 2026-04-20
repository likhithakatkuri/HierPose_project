"""MediaPipe landmark extractor with JHMDB joint mapping.

Extracts (T, 15, 2) normalised keypoint arrays from:
    - Video files  (extract_from_video)
    - Live webcam  (extract_from_webcam)

MediaPipe's 33-landmark BlazePose model is mapped to the 15 JHMDB joints.
Joints that correspond to averages of two MediaPipe landmarks (e.g. NECK)
are computed as the mean of the two landmark coordinates.

MediaPipe is an optional dependency.  An ImportError is raised at
instantiation time with a helpful install message if it is absent.

Example::

    from psrn.inference import MediaPipeExtractor

    extractor = MediaPipeExtractor(model_complexity=1)
    frames = extractor.extract_from_video("clip.mp4")   # (T, 15, 2)
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# MediaPipe is optional — raise at instantiation, not import time
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ─────────────────────────────────────────────────────────────
# MediaPipe → JHMDB joint mapping
# ─────────────────────────────────────────────────────────────

# JHMDB joint index constants
JHMDB_NECK      = 0
JHMDB_BELLY     = 1
JHMDB_FACE      = 2
JHMDB_R_SHOULDER = 3
JHMDB_L_SHOULDER = 4
JHMDB_R_HIP     = 5
JHMDB_L_HIP     = 6
JHMDB_R_ELBOW   = 7
JHMDB_L_ELBOW   = 8
JHMDB_R_KNEE    = 9
JHMDB_L_KNEE    = 10
JHMDB_R_WRIST   = 11
JHMDB_L_WRIST   = 12
JHMDB_R_ANKLE   = 13
JHMDB_L_ANKLE   = 14
N_JHMDB_JOINTS  = 15

# MediaPipe BlazePose landmark indices (33 total)
# Ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
_MP_NOSE          = 0
_MP_L_SHOULDER    = 11
_MP_R_SHOULDER    = 12
_MP_L_ELBOW       = 13
_MP_R_ELBOW       = 14
_MP_L_WRIST       = 15
_MP_R_WRIST       = 16
_MP_L_HIP         = 23
_MP_R_HIP         = 24
_MP_L_KNEE        = 25
_MP_R_KNEE        = 26
_MP_L_ANKLE       = 27
_MP_R_ANKLE       = 28

# Mapping: JHMDB joint index → MediaPipe landmark index or tuple of indices to average
# Tuple means the JHMDB joint is estimated as the mean of the listed MP landmarks.
MEDIAPIPE_TO_JHMDB: Dict[int, Union[int, Tuple[int, ...]]] = {
    JHMDB_NECK:       (_MP_L_SHOULDER, _MP_R_SHOULDER),   # mid-shoulder = neck proxy
    JHMDB_BELLY:      (_MP_L_HIP,      _MP_R_HIP),        # mid-hip = belly proxy
    JHMDB_FACE:       _MP_NOSE,                            # nose ≈ face centre
    JHMDB_R_SHOULDER: _MP_R_SHOULDER,
    JHMDB_L_SHOULDER: _MP_L_SHOULDER,
    JHMDB_R_HIP:      _MP_R_HIP,
    JHMDB_L_HIP:      _MP_L_HIP,
    JHMDB_R_ELBOW:    _MP_R_ELBOW,
    JHMDB_L_ELBOW:    _MP_L_ELBOW,
    JHMDB_R_KNEE:     _MP_R_KNEE,
    JHMDB_L_KNEE:     _MP_L_KNEE,
    JHMDB_R_WRIST:    _MP_R_WRIST,
    JHMDB_L_WRIST:    _MP_L_WRIST,
    JHMDB_R_ANKLE:    _MP_R_ANKLE,
    JHMDB_L_ANKLE:    _MP_L_ANKLE,
}


# ─────────────────────────────────────────────────────────────
# MediaPipeExtractor
# ─────────────────────────────────────────────────────────────

class MediaPipeExtractor:
    """Extract JHMDB-compatible (T, 15, 2) keypoints via MediaPipe BlazePose.

    Args:
        model_complexity:          MediaPipe model complexity (0, 1, or 2).
                                   Higher = more accurate but slower.
        min_detection_confidence:  Minimum confidence for pose detection (0–1).
        min_tracking_confidence:   Minimum confidence for landmark tracking (0–1).
        static_image_mode:         If True, treat every frame independently
                                   (slower but more robust for sparse clips).
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        if not HAS_MEDIAPIPE:
            raise ImportError(
                "mediapipe is not installed. Install it with:\n"
                "    pip install mediapipe\n"
                "See https://developers.google.com/mediapipe for details."
            )
        if not HAS_CV2:
            raise ImportError(
                "opencv-python is not installed. Install it with:\n"
                "    pip install opencv-python"
            )

        self.model_complexity         = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence  = min_tracking_confidence
        self.static_image_mode        = static_image_mode

        self._pose: Optional[Any] = None   # mediapipe Pose instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pose(self) -> Any:
        """Lazily initialise the MediaPipe Pose solution."""
        if self._pose is None:
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
        return self._pose

    def _landmarks_to_jhmdb(
        self,
        landmarks: Any,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Convert MediaPipe NormalizedLandmarkList to (15, 2) pixel coords.

        Args:
            landmarks:    NormalizedLandmarkList from MediaPipe
            image_width:  frame width in pixels
            image_height: frame height in pixels

        Returns:
            (15, 2) array of (x, y) pixel coordinates in JHMDB joint order.
            Joints with detection failure are set to (0.0, 0.0).
        """
        lm = landmarks.landmark  # list of 33 NormalizedLandmark

        keypoints = np.zeros((N_JHMDB_JOINTS, 2), dtype=np.float32)

        for jhmdb_idx, mp_spec in MEDIAPIPE_TO_JHMDB.items():
            if isinstance(mp_spec, (list, tuple)):
                # Average multiple landmarks
                xs = [lm[i].x * image_width  for i in mp_spec]
                ys = [lm[i].y * image_height for i in mp_spec]
                keypoints[jhmdb_idx] = (np.mean(xs), np.mean(ys))
            else:
                keypoints[jhmdb_idx] = (
                    lm[mp_spec].x * image_width,
                    lm[mp_spec].y * image_height,
                )

        return keypoints

    def _process_frame(
        self,
        frame_bgr: np.ndarray,
        pose: Any,
    ) -> Optional[np.ndarray]:
        """Run MediaPipe on a single BGR frame.

        Returns:
            (15, 2) keypoint array, or None if no pose detected.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        result = pose.process(frame_rgb)

        if result.pose_landmarks is None:
            return None

        return self._landmarks_to_jhmdb(result.pose_landmarks, w, h)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_from_video(self, video_path: str) -> np.ndarray:
        """Extract JHMDB keypoints from every frame of a video file.

        Args:
            video_path: path to the video file (mp4, avi, etc.)

        Returns:
            (T, 15, 2) numpy array.
            T = number of frames where a pose was successfully detected.
            Frames without a detection are dropped.

        Raises:
            FileNotFoundError: if the video path does not exist.
            RuntimeError:      if the video cannot be opened by OpenCV.
        """
        import os
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        pose      = self._get_pose()
        all_kps: List[np.ndarray] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                kps = self._process_frame(frame, pose)
                if kps is not None:
                    all_kps.append(kps)
        finally:
            cap.release()

        if not all_kps:
            warnings.warn(
                f"No poses detected in '{video_path}'. "
                "Returning empty array of shape (0, 15, 2).",
                UserWarning,
                stacklevel=2,
            )
            return np.zeros((0, N_JHMDB_JOINTS, 2), dtype=np.float32)

        return np.stack(all_kps, axis=0)   # (T, 15, 2)

    def extract_from_webcam(
        self,
        callback: Optional[Callable[[np.ndarray], None]] = None,
        camera_index: int = 0,
        max_frames: Optional[int] = None,
        display: bool = True,
    ) -> np.ndarray:
        """Live keypoint extraction from webcam feed.

        Runs until the user presses 'q' or ESC, or until max_frames is reached.

        Args:
            callback:     Optional function called with the current accumulated
                          frames array (T, 15, 2) after each detected frame.
                          Useful for streaming prediction during capture.
            camera_index: OpenCV camera device index (0 = default webcam).
            max_frames:   Stop after this many detected frames (None = unlimited).
            display:      If True, show live OpenCV window with pose overlay.

        Returns:
            (T, 15, 2) array of all detected frames.
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {camera_index}.")

        pose      = self._get_pose()
        mp_draw   = mp.solutions.drawing_utils
        mp_pose   = mp.solutions.pose

        all_kps: List[np.ndarray] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                kps = self._process_frame(frame, pose)

                if kps is not None:
                    all_kps.append(kps)
                    accumulated = np.stack(all_kps, axis=0)
                    if callback is not None:
                        try:
                            callback(accumulated)
                        except Exception as exc:
                            warnings.warn(
                                f"Callback raised an exception: {exc}",
                                UserWarning,
                                stacklevel=2,
                            )

                    if max_frames is not None and len(all_kps) >= max_frames:
                        break

                if display:
                    # Draw landmarks on frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb.flags.writeable = False
                    result = pose.process(frame_rgb)
                    frame_rgb.flags.writeable = True
                    frame_disp = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    if result.pose_landmarks:
                        mp_draw.draw_landmarks(
                            frame_disp,
                            result.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS,
                        )

                    n = len(all_kps)
                    cv2.putText(
                        frame_disp,
                        f"Frames: {n}  |  Press Q/ESC to stop",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("HierPose — Webcam Extractor", frame_disp)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):   # 27 = ESC
                        break

        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        if not all_kps:
            return np.zeros((0, N_JHMDB_JOINTS, 2), dtype=np.float32)

        return np.stack(all_kps, axis=0)

    def close(self) -> None:
        """Release the MediaPipe Pose instance."""
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def __enter__(self) -> "MediaPipeExtractor":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"MediaPipeExtractor("
            f"complexity={self.model_complexity}, "
            f"min_det={self.min_detection_confidence})"
        )
