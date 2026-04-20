"""HierPose prediction engine.

HierPosePredictor wraps a trained joblib-serialised sklearn-compatible
classifier and scaler, providing:
    - Single-clip prediction          (predict_frames)
    - Explanation-augmented prediction (predict_with_explanation)
    - Batch prediction                 (predict_batch)

Results are returned as typed dataclasses for easy downstream use.

Example::

    predictor = HierPosePredictor(
        model_path="models/lgbm.joblib",
        scaler_path="models/scaler.joblib",
    )
    predictor.load()

    result = predictor.predict_frames(frames_array)   # frames: (T, 15, 2)
    print(result.predicted_class, f"{result.confidence:.2%}")

    explained = predictor.predict_with_explanation(frames_array)
    print(explained.domain_feedback)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ─────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Output of a single-clip prediction.

    Attributes:
        predicted_class:    predicted action / pose class name
        confidence:         probability of the predicted class (0–1)
        class_probabilities: dict mapping class name → probability
        feature_vector:     (n_features,) extracted + scaled feature array
    """
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    feature_vector: np.ndarray


@dataclass
class ExplainedPredictionResult:
    """Prediction result augmented with SHAP values and CPG corrections.

    Inherits all fields from PredictionResult and adds XAI outputs.

    Attributes:
        predicted_class:     predicted class name
        confidence:          predicted class probability
        class_probabilities: full probability distribution
        feature_vector:      scaled feature array used for prediction
        shap_values:         (n_features,) SHAP values for predicted class,
                             or None if SHAP unavailable / failed
        corrections:         List[PoseCorrection] from CounterfactualPoseGuide,
                             or empty list if domain/CPG not configured
        pose_score:          PoseScore from domain scoring,
                             or None if no domain configured
        domain_feedback:     formatted natural-language feedback string,
                             or empty string if no domain configured
    """
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    feature_vector: np.ndarray
    shap_values: Optional[np.ndarray] = None
    corrections: List[Any] = field(default_factory=list)     # List[PoseCorrection]
    pose_score: Optional[Any] = None                          # PoseScore | None
    domain_feedback: str = ""


# ─────────────────────────────────────────────────────────────
# HierPosePredictor
# ─────────────────────────────────────────────────────────────

class HierPosePredictor:
    """End-to-end HierPose inference wrapper.

    Args:
        model_path:     Path to joblib-serialised sklearn-compatible classifier.
        scaler_path:    Path to joblib-serialised StandardScaler (or None).
        feature_config: Optional dict with keys:
                            "feature_names" : List[str]
                            "n_features"    : int
                        If None, feature names are auto-generated as
                        ["f0", "f1", ..., "fN"].
        domain:         Optional BaseDomain instance for scoring and feedback.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: Optional[str],
        feature_config: Optional[Dict[str, Any]] = None,
        domain: Optional[Any] = None,   # BaseDomain
    ) -> None:
        if not HAS_JOBLIB:
            raise ImportError("joblib is required: pip install joblib")

        self.model_path    = Path(model_path)
        self.scaler_path   = Path(scaler_path) if scaler_path else None
        self._feature_selector = None
        self.feature_config = feature_config or {}
        self.domain         = domain

        self._model:          Optional[Any] = None
        self._scaler:         Optional[Any] = None
        self._class_names:    List[str]     = []
        self._feature_names:  List[str]     = []
        self._shap_explainer: Optional[Any] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "HierPosePredictor":
        """Load model and scaler from disk.

        Returns:
            self (for method chaining)

        Raises:
            FileNotFoundError: if model_path does not exist.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._model = joblib.load(self.model_path)

        if self.scaler_path is not None:
            if not self.scaler_path.exists():
                warnings.warn(
                    f"Scaler not found at {self.scaler_path} — "
                    "features will not be scaled.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                self._scaler = joblib.load(self.scaler_path)

        # Load feature selector if saved alongside model
        selector_path = self.model_path.parent / "feature_selector.pkl"
        if selector_path.exists():
            self._feature_selector = joblib.load(selector_path)

        # Resolve class names — always use JHMDB names (model stores int indices 0-20)
        from psrn.data.jhmdb_loader import JHMDB_CLASSES
        self._class_names = list(JHMDB_CLASSES)

        # Resolve feature names
        self._feature_names = self.feature_config.get("feature_names", [])

        self._loaded = True
        return self

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, frames: np.ndarray) -> np.ndarray:
        """Extract feature vector from (T, 15, 2) frame array.

        Attempts to use psrn.features pipeline; falls back to a simple
        flattened mean/std representation if unavailable.

        Args:
            frames: (T, 15, 2) numpy array of keypoints

        Returns:
            (n_features,) feature vector (unscaled)
        """
        from psrn.features.extractor import HierarchicalFeatureExtractor, FeatureConfig
        cfg = FeatureConfig(joint_schema="jhmdb")
        extractor = HierarchicalFeatureExtractor(cfg)
        feat_vec, names = extractor.extract_and_pool(frames)
        if not self._feature_names:
            self._feature_names = names
        return feat_vec

    def _scale(self, feature_vec: np.ndarray) -> np.ndarray:
        """Apply scaler then feature selector to feature vector."""
        vec = feature_vec.reshape(1, -1)
        if self._scaler is not None:
            vec = self._scaler.transform(vec)
        if self._feature_selector is not None:
            vec = self._feature_selector.transform(vec)
        return vec[0]

    def _ensure_feature_names(self, n: int) -> List[str]:
        """Return feature names, generating placeholders if needed."""
        if len(self._feature_names) == n:
            return self._feature_names
        return [f"f{i}" for i in range(n)]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_from_vector(self, feat_scaled: np.ndarray) -> PredictionResult:
        """Core prediction from a pre-scaled feature vector."""
        proba = self._model.predict_proba(feat_scaled.reshape(1, -1))[0]
        pred_idx = int(np.argmax(proba))
        pred_cls = self._class_names[pred_idx] if self._class_names else str(pred_idx)
        confidence = float(proba[pred_idx])

        class_probs: Dict[str, float] = {
            (self._class_names[i] if self._class_names else str(i)): float(p)
            for i, p in enumerate(proba)
        }

        return PredictionResult(
            predicted_class=pred_cls,
            confidence=confidence,
            class_probabilities=class_probs,
            feature_vector=feat_scaled,
        )

    def predict_frames(self, frames: np.ndarray) -> PredictionResult:
        """Predict action/pose class for a single clip.

        Args:
            frames: (T, 15, 2) array — T frames, 15 JHMDB joints, (x, y) coords.

        Returns:
            PredictionResult

        Raises:
            RuntimeError: if load() has not been called.
        """
        if not self._loaded:
            raise RuntimeError("Call predictor.load() before predicting.")

        feat_raw    = self._extract_features(frames)
        feat_scaled = self._scale(feat_raw)
        return self._predict_from_vector(feat_scaled)

    def predict_with_explanation(
        self,
        frames: np.ndarray,
        target_class: Optional[str] = None,
        max_shap_background: int = 100,
    ) -> ExplainedPredictionResult:
        """Predict with SHAP explanation and CPG counterfactual guidance.

        Args:
            frames:               (T, 15, 2) keypoint array.
            target_class:         target class for CPG.  If None, uses the
                                  second-highest probability class.
            max_shap_background:  max background samples for SHAP KernelExplainer.

        Returns:
            ExplainedPredictionResult
        """
        if not self._loaded:
            raise RuntimeError("Call predictor.load() before predicting.")

        feat_raw    = self._extract_features(frames)
        feat_scaled = self._scale(feat_raw)
        base_result = self._predict_from_vector(feat_scaled)

        feature_names = self._ensure_feature_names(len(feat_scaled))

        # --- SHAP ---
        shap_values: Optional[np.ndarray] = None
        if HAS_SHAP:
            try:
                shap_values = self._compute_shap(feat_scaled)
            except Exception as exc:
                warnings.warn(f"SHAP computation failed: {exc}", UserWarning, stacklevel=2)

        # --- CPG counterfactual ---
        corrections: List[Any] = []
        if target_class is None:
            probs = base_result.class_probabilities
            sorted_cls = sorted(probs, key=lambda c: -probs[c])
            target_class = (
                sorted_cls[1] if len(sorted_cls) > 1 else sorted_cls[0]
            )

        try:
            from psrn.explainability.counterfactual import CounterfactualPoseGuide
            cpg = CounterfactualPoseGuide(
                model=self._model,
                scaler=self._scaler,
                feature_names=feature_names,
                class_names=self._class_names,
                domain=self.domain.domain_name if self.domain else "general",
            )
            cf_result = cpg.generate(feat_scaled, target_class=target_class)
            corrections = cf_result.corrections
        except Exception as exc:
            warnings.warn(f"CPG generation failed: {exc}", UserWarning, stacklevel=2)

        # --- Domain scoring + feedback ---
        pose_score = None
        domain_feedback = ""
        if self.domain is not None:
            try:
                pose_score = self.domain.compute_pose_score(
                    features=feat_raw,
                    feature_names=feature_names,
                    predicted_class=base_result.predicted_class,
                )
                domain_feedback = self.domain.generate_feedback(
                    corrections=corrections,
                    target_class=target_class,
                )
            except Exception as exc:
                warnings.warn(
                    f"Domain scoring failed: {exc}", UserWarning, stacklevel=2
                )

        return ExplainedPredictionResult(
            predicted_class=base_result.predicted_class,
            confidence=base_result.confidence,
            class_probabilities=base_result.class_probabilities,
            feature_vector=feat_scaled,
            shap_values=shap_values,
            corrections=corrections,
            pose_score=pose_score,
            domain_feedback=domain_feedback,
        )

    def predict_batch(
        self,
        samples_list: List[np.ndarray],
    ) -> List[PredictionResult]:
        """Predict for a list of clips.

        Args:
            samples_list: List of (T, 15, 2) arrays (variable T allowed).

        Returns:
            List[PredictionResult] in the same order as input.
        """
        if not self._loaded:
            raise RuntimeError("Call predictor.load() before predicting.")

        results: List[PredictionResult] = []
        for frames in samples_list:
            try:
                results.append(self.predict_frames(frames))
            except Exception as exc:
                warnings.warn(
                    f"Batch prediction failed for one sample: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                # Append a stub result so list length matches input
                results.append(
                    PredictionResult(
                        predicted_class="unknown",
                        confidence=0.0,
                        class_probabilities={},
                        feature_vector=np.array([]),
                    )
                )
        return results

    # ------------------------------------------------------------------
    # SHAP helpers
    # ------------------------------------------------------------------

    def _compute_shap(self, feat_scaled: np.ndarray) -> Optional[np.ndarray]:
        """Compute SHAP values for the predicted class.

        Uses TreeExplainer for tree models (LightGBM, XGBoost, RF) and
        falls back to KernelExplainer for others.

        Returns:
            (n_features,) SHAP values for the predicted class, or None.
        """
        if not HAS_SHAP:
            return None

        model = self._model
        x = feat_scaled.reshape(1, -1)

        # Try TreeExplainer first (fast, exact)
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(x)
            if isinstance(sv, list):
                # Multi-class: pick predicted class
                pred_idx = int(np.argmax(
                    model.predict_proba(x)[0]
                ))
                sv = sv[pred_idx]
            return np.array(sv).flatten()
        except Exception:
            pass

        # Fallback: KernelExplainer with random background
        try:
            n_bg = min(50, len(feat_scaled))
            background = np.zeros((n_bg, len(feat_scaled)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sv = explainer.shap_values(x, nsamples=100)
            if isinstance(sv, list):
                pred_idx = int(np.argmax(model.predict_proba(x)[0]))
                sv = sv[pred_idx]
            return np.array(sv).flatten()
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"HierPosePredictor("
            f"model='{self.model_path.name}', "
            f"classes={len(self._class_names)}, "
            f"status={status})"
        )
