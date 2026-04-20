"""HierPose: Hybrid Geometric–ML for Interpretable Pose Classification.

Dataset  : JHMDB (21 action classes, 15 skeleton joints, .mat annotations)
           data/JHMDB/joint_positions/<class>/<video>/joint_positions.mat
           data/JHMDB/splits/<class>_test_split<N>.txt

Pipeline : joint .mat → hierarchical features (~380) → LightGBM/XGBoost/RF/SVM/LDA/Ensemble
           → SHAP explainability + Counterfactual Pose Guidance (CPG)
"""

__version__ = "2.0.0"

