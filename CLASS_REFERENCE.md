# HierPose: Class Reference & Integration Guide

## QUICK CLASS REFERENCE

### FEATURES (psrn/features/)

#### Registry & Configuration
- **FeatureConfig** (extractor.py:L40) - Controls enabled feature groups, normalization, temporal window
  - `enabled_groups: Optional[List[str]]` - Which of 16 groups to compute
  - `temporal_agg_stats: List[str]` - Aggregation: ["mean", "std", "q25", "q75"]

- **FEATURE_GROUPS** (registry.py:L160) - Dict mapping group names to metadata
  - 16 groups total: 8 static + 4 temporal_per_frame + 4 temporal_clip

#### Main Extractors
- **HierarchicalFeatureExtractor** (extractor.py:L150)
  - `.extract(frames: (T, 15, 2))` → per-frame features (T, n_features)
  - `.extract_and_pool(frames)` → aggregated clip-level vector (n_features,)
  - `.extract_batch(samples, augmenter, cache_dir)` → (n_samples, n_features) matrix

- **frame_all_static_features** (static.py:L280)
  - Computes all 8 static groups for one frame
  - Output: (n_static_features,), (list of names)

- **sequence_all_temporal_per_frame** (temporal.py:L280)
  - Computes velocity, acceleration, moving-avg, motion energy per frame
  - Output: (T, n_temporal_per_frame), (list of names)

- **sequence_all_clip_level** (temporal.py:L330)
  - Computes ROM, variance, peak timing, direction reversals for entire clip
  - Output: (n_clip_features,), (list of names)

#### Gait Analysis (Clinical)
- **HierarchicalGaitFeatureExtractor** (gait.py:L190)
  - **L1**: `.\_extract_angle_profiles(joints)` → joint angle time-series
  - **L2**: `.\_detect_gait_events(joints)` → GaitEvent list + GaitCycle list
  - **L2**: `.\_compute_phase_params(cycles)` → cadence, stance/swing ratios, Robinson SI
  - **L2**: `.\_compute_event_angles(...)` → knee angles at events, ROM
  - **L3**: `.analyse(joints)` → GaitReport (full hierarchical output)

- **GaitReport** (gait.py:L95)
  - Contains all L1/L2/L3 outputs: angle_profiles, cycles, GDI, bilateral correlation, trunk sway, risk_flags

- **GaitEvent, GaitCycle** (gait.py:L62-L87) - Data containers

---

### DOMAINS (psrn/domains/)

#### Base Classes
- **BaseDomain** (base.py:L30-L150)
  - `.compute_pose_score(features, class_name)` → PoseScore
  - `.generate_feedback(corrections, target_class)` → text feedback
  - `.get_class(name)` → DomainPoseClass

#### Specific Domains
1. **SquatFormDomain** (sports.py:L50)
   - Classes: correct_depth, too_shallow, knee_cave, forward_lean, good_form
   - Score components: knee depth (40%), back angle (35%), alignment (25%)
   - Formula: composite = 0.40·S_knee + 0.35·S_back + 0.25·S_align

2. **WorkplaceErgonomicsDomain** (ergonomics.py:L50)
   - Classes: neutral, low_risk, medium_risk, high_risk
   - Factors: upper_arm, neck, trunk, wrist (sigmoid scoring)
   - Returns: risk_level + workstation adjustment guidance

3. **PredictiveBiomechanicalRiskScorer** (injury_risk.py:L190)
   - Computes 5 injury types: ACL, PFPS, IT_Band, LBP, Shoulder_Impingement
   - L1/L2/L3 risk factors (14 total) with clinical weights
   - Sigmoid risk function per factor
   - Returns: InjuryRiskResult dict {injury_type → probability}

4. **HierarchicalKinematicResidualAnalyzer** (compensation.py)
   - Kinematic chain model (hard-coded tree structure)
   - For each joint: expected_angle = β·parent_angle + α
   - Residual analysis: actual − expected
   - Root-cause propagation (which joint is primary fault?)

5. **CrossModalPainBehaviorAnalyzer** (pain_detection.py)
   - Fuses joint kinematics + facial AU geometry
   - PSPI proxy: w4·AU4 + w6·AU6 + ... (Prkachin-Solomon)
   - Output: PC-ROM (pain-correlated range-of-motion)

6. **AdaptiveCareEngine** (adaptive.py)
   - Longitudinal compliance tracking (EWMA, α=0.3)
   - Mann-Kendall trend test (improving/stable/regressing)
   - Auto-generates exercise protocols from FAULT_EXERCISE_MAP
   - Discharge Readiness Index (DRI)

#### Result Classes
- **PoseScore** (base.py:L20) - score, risk_level, feedback, details dict
- **InjuryRiskResult** (injury_risk.py:L75) - injury_type, risk_percent, contributing_factors
- **RiskReport** (injury_risk.py:L100) - injury_risks dict, overall_risk, action_plan
- **PCROMResult** (pain_detection.py:L45) - pain_free_rom, pain_onset_angle, pain_peak_angle

---

### MACHINE LEARNING (psrn/training/)

#### Training Pipeline
- **HierPoseTrainer** (trainer.py:L100)
  - `.run()` → full experiment pipeline
  - `.\_load_data()` → X_train, y_train, X_test, y_test, feature_names
  - `.\_preprocess(X_train, X_test)` → StandardScaler + SelectKBest(k=200)
  - `.\_train_model(X_train, y_train)` → fitted model + report
  - Outputs: ExperimentResult with accuracy, F1, confusion_matrix

- **ExperimentResult** (trainer.py:L45)
  - accuracy, accuracy_per_class, macro_f1, weighted_f1
  - confusion_matrix, cv_mean, cv_std
  - `.summary()` → printable string

#### Model Selection
- **ModelSelector** (model_selector.py:L100)
  - `.fit_best(X_train, y_train, model_type)` → best model + ModelSelectionReport
  - Trains: LightGBM, XGBoost, Random Forest, SVM, LDA
  - Selects top-3 → builds soft-voting ensemble

- **ModelSelectionReport** (model_selector.py:L30)
  - `.summary_table()` → comparison of all models
  - Shows CV accuracy ± std, training time per model

#### Hyperparameter Tuning
- **tune_lightgbm** (hyperparameter_search.py:L40)
  - Optuna TPE sampler, 50 trials default
  - Search space: n_estimators, num_leaves, learning_rate, regularization, etc.

- **tune_xgboost, tune_random_forest, _tune_svm_grid** - Similar structure

#### Cross-Validation & Testing
- **cross_validate_model** (cross_validation.py:L50)
  - 5-fold Stratified K-Fold
  - Returns: CVResult with cv_scores, cv_mean, cv_std, OOF predictions

- **mcnemar_test** (cross_validation.py:L165)
  - Pairwise statistical significance (corrected McNemar)
  - Returns: McNemarResult with χ², p-value, winner

- **nested_cross_validate** (cross_validation.py:L100)
  - For unbiased performance (outer loop for generalization, inner for tuning)

#### Ablation Study
- **AblationStudy** (ablation.py:L60)
  - `.run_leave_one_out()` → DataFrame of feature importance (baseline - ablated)
  - `.run_incremental()` → DataFrame of cumulative accuracy curve
  - Outputs to "ablation/" directory as CSV

---

### EXPLAINABILITY (psrn/explainability/)

#### SHAP Analysis
- **SHAPAnalyzer** (shap_analysis.py:L50)
  - `.compute(max_samples=100)` → SHAP values (n_samples, n_features, n_classes)
  - `.global_importance` → (n_features,) mean |SHAP| per feature
  - `.plot_bar_summary()` → top-K features bar chart (publication-quality PNG)
  - `.plot_beeswarm_global()` → standard SHAP beeswarm plot
  - `.plot_beeswarm_per_class()` → one beeswarm per class (21 for JHMDB)
  - `.map_to_anatomical_groups()` → aggregate importance by feature group
  - `.feature_importance_table()` → DataFrame for IEEE Table 1

#### Counterfactual Explanation
- **CounterfactualPoseGuide** (counterfactual.py:L100)
  - `.generate(features_scaled, target_class)` → CounterfactualResult
  - Solver: L-BFGS-B with L2 distance + classification loss
  - `.explain_as_text(result, domain)` → human-readable corrections with domain-specific templates

- **CounterfactualResult** (counterfactual.py:L35)
  - original_features, counterfactual_features (scaled)
  - corrections: List[PoseCorrection] (ranked by importance)
  - l2_distance, optimization_success
  - `.summary(domain, max_corrections)` → text output

- **PoseCorrection** (counterfactual.py:L20)
  - feature_name, current_value, target_value, delta
  - body_part, feature_group, importance_rank
  - `.to_text(domain)` → "Increase knees 12° (currently 92°, target 80°)"

#### Skeleton Visualization
- **draw_skeleton_plain** (skeleton_viz.py:L90) - Basic skeleton
- **draw_skeleton_shap_heatmap** (skeleton_viz.py:L130) - Joints colored by SHAP importance
- **draw_correction_skeleton** (skeleton_viz.py:L210) - Current pose + correction arrows

---

## INTEGRATION EXAMPLE: SQUAT ANALYSIS

```python
# 1. Extract features
extractor = HierarchicalFeatureExtractor(FeatureConfig())
features, names = extractor.extract_and_pool(frame_sequence)  # (650,)

# 2. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
selector = SelectKBest(k=200)
X_train_processed = selector.fit_transform(X_train_scaled, y_train)

# 3. Train model
from psrn.training.model_selector import ModelSelector
selector = ModelSelector()
model, report = selector.fit_best(X_train_processed, y_train, "ensemble")

# 4. Get domain-specific feedback
domain = SquatFormDomain()
test_features_scaled = scaler.transform(features.reshape(1, -1))
test_features_processed = selector.transform(test_features_scaled)
pred_class = model.predict(test_features_processed)[0]

pose_score = domain.compute_pose_score(features, names, domain.class_names[pred_class])
print(f"Squat Form Score: {pose_score.score}/100")
print(f"Risk Level: {pose_score.risk_level}")
print(f"Feedback: {pose_score.feedback}")

# 5. SHAP Explainability
from psrn.explainability.shap_analysis import SHAPAnalyzer
analyzer = SHAPAnalyzer(model, X_test_processed, y_test, 
                        selected_names, domain.class_names)
analyzer.compute()
analyzer.plot_bar_summary("outputs/shap_importance.png")
group_importance = analyzer.map_to_anatomical_groups()
print(f"Important groups: {group_importance}")  # {'angles': 0.35, 'temporal_vel': 0.28, ...}

# 6. Counterfactual Guidance
from psrn.explainability.counterfactual import CounterfactualPoseGuide
cpg = CounterfactualPoseGuide(model, scaler, selected_names, 
                              domain.class_names, domain="sports")
result = cpg.generate(test_features_scaled, "correct_depth")
print(result.summary("sports", max_corrections=5))

# Output:
# Current pose: too_shallow
# Target pose:  correct_depth
# 
# Required corrections (3/5 shown):
#   1. Bend knees further: 92° → 78° (decrease 14°)
#   2. Reduce trunk lean: 50° → 38° (decrease 12°)
#   3. Check knee alignment: within acceptable range
```

---

## KEY DATA STRUCTURES

### Feature Naming Convention
All feature names follow pattern: `{group}_{part}_{property}`

Examples:
- `angle_r_knee_deg` → angles group, right knee, degree value
- `vel_l_wrist` → temporal_vel group, left wrist joint
- `rom_trunk_inclination` → range_of_motion group, trunk angle
- `sym_knee_angle_diff` → symmetry group, knee angle difference

### Joint Index Constants (JHMDB 15-joint model)
```python
NECK=0, BELLY=1, FACE=2
R_SHOULDER=3, L_SHOULDER=4
R_HIP=5, L_HIP=6
R_ELBOW=7, L_ELBOW=8
R_KNEE=9, L_KNEE=10
R_WRIST=11, L_WRIST=12
R_ANKLE=13, L_ANKLE=14
```

### Registry Grouping
**Static (8 groups)**:
- angles (28), distances (14), ratios (4), centroids (12), symmetry (12), orientation (12), extent (8), cross_body (10)

**Temporal Per-Frame (4 groups)**:
- temporal_vel (10), temporal_acc (10), temporal_ma (10), motion_energy (9)

**Clip-Level (4 groups)**:
- range_of_motion (14), temporal_variance (14), peak_velocity (10), direction_reversals (10)

---

## EXECUTION FLOW: Training → Evaluation → Explanation

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. TRAINING PHASE                                               │
├─────────────────────────────────────────────────────────────────┤
│ Load JHMDB → Extract Features → Preprocess → Train Models       │
│   ↓              ↓                ↓          ↓                  │
│   672 samples    ~650 dims        200 dims   5 base models      │
│   (600 train)    per video        selected   → ensemble         │
│   (72 test)                                                     │
│                                                                 │
│ Output: ExperimentResult (accuracy, F1, CV scores)             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. EVALUATION PHASE                                             │
├─────────────────────────────────────────────────────────────────┤
│ Per-Class Accuracy → Confusion Matrix → McNemar Tests           │
│   ↓                  ↓                  ↓                       │
│   Report by class    Which classes      Pairwise model         │
│   (21 rows)          confused?          significance           │
│                                                                 │
│ Output: ModelSelectionReport (summary table, best model)       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. EXPLANATION PHASE (if model is tree-based or ensemble)      │
├─────────────────────────────────────────────────────────────────┤
│ SHAP Analysis:                  Counterfactual Guidance:        │
│  - TreeExplainer on LightGBM     - Minimize L2 distance in     │
│  - Compute global importance       feature space              │
│  - Per-class importance          - Map back to anatomy        │
│  - Group aggregation             - domain-specific feedback   │
│  → Bar chart + Beeswarm plots    → Correction list + text    │
│                                                                 │
│ Output: SHAP figures + Counterfactual corrections             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. ABLATION STUDY (optional, post-hoc)                         │
├─────────────────────────────────────────────────────────────────┤
│ Leave-One-Out: Train 16 times (all features except one)        │
│   → Measure accuracy drop per group                            │
│   → Rank groups by importance                                  │
│                                                                 │
│ Output: Ablation CSV (which feature groups matter most?)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## DOMAIN APPLICATION MATRIX

| Domain | Key Class | Input Features | Output | Clinical Use |
|--------|-----------|-----------------|--------|--------------|
| **Sports** | SquatFormDomain | angles, distances, trunk | 0–100 score + corrections | Real-time form feedback |
| **Medical (Gait)** | HierarchicalGaitFeatureExtractor | temporal angles + velocity | GDI, SI, ROM, z-scores | Post-stroke rehab assessment |
| **Ergonomics** | WorkplaceErgonomicsDomain | upper_arm, neck, trunk, wrist | Risk level + workstation adjustment | Office ergonomics audit |
| **Injury Risk** | PredictiveBiomechanicalRiskScorer | biomechanical deviations | Per-injury probability + top factors | ACL/PFPS/LBP risk screening |
| **Compensation** | HierarchicalKinematicResidualAnalyzer | joint angles | Residual analysis → root cause | PT diagnosis (which joint is primary?) |
| **Pain** | CrossModalPainBehaviorAnalyzer | joint kinematics + facial AUs | PC-ROM + pain-free angle ranges | Therapeutic ROM boundaries |
| **Longitudinal** | AdaptiveCareEngine | session compliance series | Trends + auto-generated protocols | Outcome tracking + discharge criteria |

---

## CONFIGURATION EXAMPLE

```python
# config.yaml for research paper experiments
HierPoseConfig(
    data_root=Path("data/JHMDB"),
    split_num=1,
    enabled_feature_groups=None,          # All 16 groups
    normalize_by_torso=True,
    temporal_window=5,
    augment_train=True,
    n_cv_folds=5,
    model_type="ensemble",                # LightGBM + RF + XGBoost
    tune_hyperparams=True,
    n_optuna_trials=50,
    save_shap=True,
    save_confusion_matrix=True,
    experiment_name="hierpose_jhmdb_split1_full"
)
```

---

Document Version: 1.0  
Last Updated: 2026-04-18  
For Research Paper: IEEE Table 1 (features), Table 2 (models), Table 3 (ablation), Figures 4–5 (SHAP)
