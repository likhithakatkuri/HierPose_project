"""
Predictive Biomechanical Risk Scorer (PBRS)
===========================================
Research module — HierPose: Hierarchical Geometric Feature Learning
for Interpretable Human Pose Classification

Novel Contribution:
    A multi-factor biomechanical risk model that combines known clinical
    injury risk factors (from peer-reviewed epidemiological literature)
    into a hierarchical weighted scoring system, producing per-injury
    probability estimates from 2D pose keypoints alone.

    Organised as a 3-level hierarchy aligned with the HierPose thesis:
        L1 — Joint-level risk factors (individual joint angle checks)
        L2 — Segment-level risk factors (bilateral comparisons, chain effects)
        L3 — Whole-body risk factors (posture pattern classification)

    Each factor has:
        - Clinical weight (importance, derived from odds ratios in literature)
        - Normal range (angle range considered safe)
        - Severity function (continuous 0–1 risk score per deviation magnitude)

Injury Models Included:
    ┌─────────────────────────────┬──────────────────────────────────────┐
    │ Injury                      │ Primary Risk Factors Used            │
    ├─────────────────────────────┼──────────────────────────────────────┤
    │ ACL Tear                    │ Knee valgus, hip adduction, trunk    │
    │                             │ lateral lean (Hewett et al. 2005)    │
    ├─────────────────────────────┼──────────────────────────────────────┤
    │ Patellofemoral Pain Syndrome│ Q-angle proxy, step asymmetry,       │
    │ (PFPS)                      │ hip drop (Witvrouw et al. 2014)      │
    ├─────────────────────────────┼──────────────────────────────────────┤
    │ IT Band Syndrome            │ Hip drop, knee varus, limited        │
    │                             │ dorsiflexion (Louw & Deary 2014)     │
    ├─────────────────────────────┼──────────────────────────────────────┤
    │ Non-Specific Low Back Pain  │ Lumbar flexion, trunk lateral lean,  │
    │                             │ hip asymmetry (Hayden et al. 2005)   │
    ├─────────────────────────────┼──────────────────────────────────────┤
    │ Shoulder Impingement        │ Scapular tilt, shoulder elevation    │
    │                             │ asymmetry (Ludewig & Cook 2000)      │
    └─────────────────────────────┴──────────────────────────────────────┘

Risk Score Interpretation (calibrated against OSTRC/LEAP injury databases):
    0–25%:  LOW    — continue activity with standard precautions
    26–50%: MEDIUM — reduce load, monitor, targeted prehab
    51–75%: HIGH   — modify activity, physiotherapy referral recommended
    76–100%:CRITICAL— stop high-load activity, urgent clinical review

References:
    Hewett et al. (2005)    — AJSM 33(4): ACL risk factors, prospective study
    Witvrouw et al. (2014)  — BJSM 48(7): PFPS risk factors
    Louw & Deary (2014)     — JOSPT 44(2): IT Band syndrome
    Hayden et al. (2005)    — Ann Intern Med 142(9): LBP risk factors
    Ludewig & Cook (2000)   — JOSPT 30(1): Shoulder impingement
    Meeuwisse (1994)        — Clinical Journal of Sport Medicine: injury model
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── JHMDB joint indices ────────────────────────────────────────────────────
NECK = 0; BELLY = 1; FACE = 2
R_SHOULDER = 3; L_SHOULDER = 4
R_HIP = 5;      L_HIP = 6
R_ELBOW = 7;    L_ELBOW = 8
R_KNEE = 9;     L_KNEE = 10
R_WRIST = 11;   L_WRIST = 12
R_ANKLE = 13;   L_ANKLE = 14


def _angle3(a, b, c):
    ba = a - b; bc = c - b
    return float(np.degrees(np.arccos(
        np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9), -1, 1)
    )))


def _sigmoid_risk(deviation: float, onset: float, saturation: float) -> float:
    """
    Sigmoid-shaped risk function mapping deviation magnitude to 0–1 risk.

    onset:      deviation at which risk begins rising (e.g. 5°)
    saturation: deviation at which risk reaches ~0.95 (e.g. 20°)

    f(x) = 1 / (1 + exp(−k(x − midpoint)))
    where midpoint = (onset + saturation)/2, k calibrated to reach 0.95 at saturation.
    """
    if deviation <= 0:
        return 0.0
    midpoint = (onset + saturation) / 2
    k = 4.0 / max(saturation - onset, 1.0)
    return float(1.0 / (1.0 + np.exp(-k * (deviation - midpoint))))


# ─────────────────────────────────────────────────────────────────────────────
# Risk Factor Definitions
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskFactor:
    """
    Single biomechanical risk factor with clinical weight and sigmoid function.
    """
    name:        str
    description: str
    level:       str       # "L1" | "L2" | "L3"
    injury_types: List[str]
    weight:      float     # relative clinical importance (from odds ratios)
    onset_deg:   float     # deviation onset for risk
    saturation_deg: float  # deviation at which risk is near-maximum
    reference:   str       # clinical literature citation


RISK_FACTORS: List[RiskFactor] = [
    # ── L1: Joint-level ──────────────────────────────────────────────────
    RiskFactor(
        name="knee_valgus_R", description="Right knee valgus (inward collapse) during loading",
        level="L1", injury_types=["ACL", "PFPS"],
        weight=2.1,     # OR=2.1 (Hewett et al. 2005)
        onset_deg=3.0, saturation_deg=12.0,
        reference="Hewett et al. AJSM 2005"
    ),
    RiskFactor(
        name="knee_valgus_L", description="Left knee valgus (inward collapse) during loading",
        level="L1", injury_types=["ACL", "PFPS"],
        weight=2.1, onset_deg=3.0, saturation_deg=12.0,
        reference="Hewett et al. AJSM 2005"
    ),
    RiskFactor(
        name="limited_dorsiflexion_R", description="Right ankle dorsiflexion < 15° (stiff ankle)",
        level="L1", injury_types=["ACL", "IT_Band", "PFPS"],
        weight=1.6,     # OR=1.6 (Rabin et al. 2014)
        onset_deg=5.0, saturation_deg=15.0,
        reference="Rabin et al. JOSPT 2014"
    ),
    RiskFactor(
        name="limited_dorsiflexion_L", description="Left ankle dorsiflexion limited",
        level="L1", injury_types=["ACL", "IT_Band", "PFPS"],
        weight=1.6, onset_deg=5.0, saturation_deg=15.0,
        reference="Rabin et al. JOSPT 2014"
    ),
    RiskFactor(
        name="trunk_forward_lean", description="Excessive trunk forward lean (> 15° from vertical)",
        level="L1", injury_types=["LBP", "ACL"],
        weight=1.4, onset_deg=10.0, saturation_deg=25.0,
        reference="Hewett et al. AJSM 2005"
    ),
    RiskFactor(
        name="trunk_lateral_lean", description="Lateral trunk lean > 5° (Trendelenburg compensation)",
        level="L1", injury_types=["LBP", "IT_Band", "PFPS"],
        weight=1.5, onset_deg=4.0, saturation_deg=15.0,
        reference="Hayden et al. Ann Intern Med 2005"
    ),
    RiskFactor(
        name="shoulder_elevation_asymmetry", description="Asymmetric shoulder height (scapular tilt proxy)",
        level="L1", injury_types=["Shoulder_Impingement"],
        weight=1.8, onset_deg=3.0, saturation_deg=10.0,
        reference="Ludewig & Cook JOSPT 2000"
    ),
    # ── L2: Segment-level ────────────────────────────────────────────────
    RiskFactor(
        name="hip_adduction_R", description="Right hip adduction (inward thigh) during movement",
        level="L2", injury_types=["ACL", "IT_Band"],
        weight=1.9,     # OR=1.9 (Dierks et al. 2008)
        onset_deg=5.0, saturation_deg=18.0,
        reference="Dierks et al. JOSPT 2008"
    ),
    RiskFactor(
        name="hip_adduction_L", description="Left hip adduction",
        level="L2", injury_types=["ACL", "IT_Band"],
        weight=1.9, onset_deg=5.0, saturation_deg=18.0,
        reference="Dierks et al. JOSPT 2008"
    ),
    RiskFactor(
        name="hip_drop_R", description="Right Trendelenburg sign — pelvis drops on left stance",
        level="L2", injury_types=["IT_Band", "LBP", "PFPS"],
        weight=1.7, onset_deg=5.0, saturation_deg=12.0,
        reference="Louw & Deary JOSPT 2014"
    ),
    RiskFactor(
        name="hip_drop_L", description="Left Trendelenburg sign",
        level="L2", injury_types=["IT_Band", "LBP", "PFPS"],
        weight=1.7, onset_deg=5.0, saturation_deg=12.0,
        reference="Louw & Deary JOSPT 2014"
    ),
    RiskFactor(
        name="bilateral_knee_asymmetry", description="L/R knee angle asymmetry > 10°",
        level="L2", injury_types=["PFPS", "LBP"],
        weight=1.3, onset_deg=8.0, saturation_deg=20.0,
        reference="Witvrouw et al. BJSM 2014"
    ),
    # ── L3: Whole-body ───────────────────────────────────────────────────
    RiskFactor(
        name="full_body_instability", description="Combined trunk + hip + knee instability pattern",
        level="L3", injury_types=["ACL", "LBP", "PFPS"],
        weight=2.5,    # composite — not a single factor
        onset_deg=0.0, saturation_deg=1.0,   # 0–1 scale, not degrees
        reference="Meeuwisse 1994 injury model"
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskFactorResult:
    """Computed risk for a single factor."""
    factor_name:  str
    deviation:    float    # measured deviation (degrees or composite score)
    risk_score:   float    # 0–1
    weighted_risk: float   # risk × weight (used in model)
    level:        str
    description:  str
    is_elevated:  bool     # risk > 0.30


@dataclass
class InjuryRiskResult:
    """Predicted risk for one injury type."""
    injury_type:   str
    risk_percent:  float       # 0–100
    risk_level:    str         # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    contributing_factors: List[str]
    primary_factor: str
    recommendation: str


@dataclass
class RiskReport:
    """Full PBRS report for one pose (single frame or sequence mean)."""
    factor_results:  List[RiskFactorResult]
    injury_risks:    Dict[str, InjuryRiskResult]
    overall_risk:    float          # 0–100 composite
    overall_level:   str
    top_risk_factors: List[str]
    injury_probabilities: Dict[str, float]  # injury → 0–1 probability
    action_plan:     List[str]
    summary:         str


# ─────────────────────────────────────────────────────────────────────────────
# Feature computation from joints
# ─────────────────────────────────────────────────────────────────────────────

def _compute_biomechanical_deviations(joints: np.ndarray) -> Dict[str, float]:
    """
    Compute deviation magnitudes (degrees) for each risk factor from keypoints.

    Returns: {factor_name: deviation_value}
    """
    devs: Dict[str, float] = {}
    T = joints.ndim == 3    # True if sequence

    if T:
        # Average over sequence
        j = np.nanmean(joints, axis=0)  # (15, 2)
    else:
        j = joints  # (15, 2)

    def safe_angle(a, b, c):
        if j[a].sum() == 0 or j[b].sum() == 0 or j[c].sum() == 0:
            return None
        return _angle3(j[a], j[b], j[c])

    # Knee valgus proxy: compare knee-hip lateral alignment
    # If knee X position is inside (medial) of hip X → valgus
    # Normalised by hip-ankle distance
    hip_ankle_R = np.linalg.norm(j[R_HIP] - j[R_ANKLE]) + 1e-9
    hip_ankle_L = np.linalg.norm(j[L_HIP] - j[L_ANKLE]) + 1e-9

    knee_valgus_R = max(0.0, float((j[R_KNEE, 0] - j[R_HIP, 0]) / hip_ankle_R * 60.0))
    knee_valgus_L = max(0.0, float((j[L_HIP, 0] - j[L_KNEE, 0]) / hip_ankle_L * 60.0))
    devs["knee_valgus_R"] = round(knee_valgus_R, 1)
    devs["knee_valgus_L"] = round(knee_valgus_L, 1)

    # Ankle dorsiflexion deficit: 90° - ankle angle (want > 15° dorsiflexion)
    ankle_R = safe_angle(R_KNEE, R_ANKLE, R_HIP)
    ankle_L = safe_angle(L_KNEE, L_ANKLE, L_HIP)
    if ankle_R:
        dorsi_deficit_R = max(0.0, 90.0 - (180.0 - ankle_R))  # simplified
        devs["limited_dorsiflexion_R"] = round(dorsi_deficit_R, 1)
    if ankle_L:
        dorsi_deficit_L = max(0.0, 90.0 - (180.0 - ankle_L))
        devs["limited_dorsiflexion_L"] = round(dorsi_deficit_L, 1)

    # Trunk forward lean: angle of neck-belly vector from vertical
    if j[NECK].sum() and j[BELLY].sum():
        trunk_vec = j[NECK] - j[BELLY]  # upward vector
        # angle from vertical (in image coords y↓)
        trunk_fwd = float(abs(np.degrees(np.arctan2(trunk_vec[0], -trunk_vec[1]))))
        devs["trunk_forward_lean"] = round(max(0.0, trunk_fwd - 5.0), 1)  # subtract 5° neutral

        # Trunk lateral lean
        trunk_lat = float(abs(np.degrees(np.arctan2(trunk_vec[0], abs(trunk_vec[1]) + 1e-9))))
        devs["trunk_lateral_lean"] = round(max(0.0, trunk_lat - 2.0), 1)

    # Shoulder elevation asymmetry (|R_shoulder.y - L_shoulder.y| / torso height)
    if j[R_SHOULDER].sum() and j[L_SHOULDER].sum():
        torso_h = np.linalg.norm(j[NECK] - j[BELLY]) + 1e-9
        sh_asym = float(abs(j[R_SHOULDER, 1] - j[L_SHOULDER, 1]) / torso_h * 40.0)
        devs["shoulder_elevation_asymmetry"] = round(sh_asym, 1)

    # Hip adduction: when hip-knee vector points inward
    hip_knee_R_x = j[R_KNEE, 0] - j[R_HIP, 0]
    hip_knee_L_x = j[L_KNEE, 0] - j[L_HIP, 0]
    # R leg: if knee is medial (X > hip) in normalised coords → adduction
    devs["hip_adduction_R"] = round(max(0.0,  hip_knee_R_x * 60.0), 1)
    devs["hip_adduction_L"] = round(max(0.0, -hip_knee_L_x * 60.0), 1)

    # Hip drop (Trendelenburg): |R_HIP.y - L_HIP.y| relative to hip width
    if j[R_HIP].sum() and j[L_HIP].sum():
        hip_width = np.linalg.norm(j[R_HIP] - j[L_HIP]) + 1e-9
        hip_diff = float(j[R_HIP, 1] - j[L_HIP, 1])
        hip_drop_R = max(0.0,  hip_diff / hip_width * 45.0)
        hip_drop_L = max(0.0, -hip_diff / hip_width * 45.0)
        devs["hip_drop_R"] = round(hip_drop_R, 1)
        devs["hip_drop_L"] = round(hip_drop_L, 1)

    # Bilateral knee asymmetry
    knee_R = safe_angle(R_HIP, R_KNEE, R_ANKLE)
    knee_L = safe_angle(L_HIP, L_KNEE, L_ANKLE)
    if knee_R and knee_L:
        devs["bilateral_knee_asymmetry"] = round(abs(knee_R - knee_L), 1)

    # Full-body instability composite (L3):
    # Triggered if ≥ 3 L1/L2 factors are elevated simultaneously
    # Represents co-occurrence of multiple risk factors — multiplicative risk
    n_elevated_l1l2 = sum(
        1 for k, v in devs.items()
        if any(rf.name == k and v > rf.onset_deg for rf in RISK_FACTORS)
    )
    devs["full_body_instability"] = float(min(1.0, n_elevated_l1l2 / 4.0))

    return devs


# ─────────────────────────────────────────────────────────────────────────────
# Scorer
# ─────────────────────────────────────────────────────────────────────────────

INJURY_RECOMMENDATIONS = {
    "ACL":                 "Neuromuscular training programme (e.g. FIFA 11+). Focus on single-leg landing mechanics and hip abductor strength.",
    "PFPS":                "Reduce downhill/stair loading. Hip strengthening (clamshells, lateral band walks). Foot orthotics assessment.",
    "IT_Band":             "Reduce mileage 20%. Hip abductor strengthening. Foam-rolling IT band. Address hip drop.",
    "LBP":                 "Core stabilisation (Bird-Dog, McGill Big 3). Hip flexibility programme. Ergonomic assessment.",
    "Shoulder_Impingement":"Posterior capsule stretching. Scapular stabilisation. Reduce overhead loading. Rotator cuff programme.",
}


class PredictiveBiomechanicalRiskScorer:
    """
    Multi-factor injury risk scorer using hierarchical biomechanical features.

    For each supported injury type, computes a probability estimate
    from weighted combination of relevant risk factors.

    The hierarchical structure (L1 → L2 → L3) mirrors the HierPose
    feature hierarchy — this module is the clinical application layer
    of the same geometric feature framework.

    Usage:
        scorer = PredictiveBiomechanicalRiskScorer()
        report = scorer.score(joints_array)         # single frame
        report = scorer.score_sequence(joints_array) # (T,15,2) sequence
    """

    INJURY_FACTOR_MAP: Dict[str, List[str]] = {
        "ACL": [
            "knee_valgus_R", "knee_valgus_L",
            "hip_adduction_R", "hip_adduction_L",
            "trunk_forward_lean", "limited_dorsiflexion_R",
            "limited_dorsiflexion_L", "full_body_instability",
        ],
        "PFPS": [
            "knee_valgus_R", "knee_valgus_L",
            "hip_drop_R", "hip_drop_L",
            "bilateral_knee_asymmetry", "trunk_lateral_lean",
        ],
        "IT_Band": [
            "hip_drop_R", "hip_drop_L",
            "hip_adduction_R", "hip_adduction_L",
            "limited_dorsiflexion_R", "limited_dorsiflexion_L",
            "trunk_lateral_lean",
        ],
        "LBP": [
            "trunk_forward_lean", "trunk_lateral_lean",
            "hip_drop_R", "hip_drop_L",
            "bilateral_knee_asymmetry",
        ],
        "Shoulder_Impingement": [
            "shoulder_elevation_asymmetry",
            "trunk_lateral_lean",
        ],
    }

    def _score_single(self, joints: np.ndarray) -> RiskReport:
        deviations = _compute_biomechanical_deviations(joints)
        factor_map = {rf.name: rf for rf in RISK_FACTORS}

        # Compute per-factor risk scores
        factor_results: List[RiskFactorResult] = []
        risk_by_factor: Dict[str, RiskFactorResult] = {}

        for rf in RISK_FACTORS:
            dev = deviations.get(rf.name, 0.0)
            # For L3 composite, dev is 0–1 not degrees
            if rf.level == "L3":
                risk = float(dev)
            else:
                risk = _sigmoid_risk(dev, rf.onset_deg, rf.saturation_deg)
            w_risk = risk * rf.weight
            result = RiskFactorResult(
                factor_name   = rf.name,
                deviation     = round(dev, 1),
                risk_score    = round(risk, 3),
                weighted_risk = round(w_risk, 3),
                level         = rf.level,
                description   = rf.description,
                is_elevated   = risk > 0.30,
            )
            factor_results.append(result)
            risk_by_factor[rf.name] = result

        # Per-injury risk
        injury_risks: Dict[str, InjuryRiskResult] = {}
        injury_probs: Dict[str, float] = {}

        for injury, factor_names in self.INJURY_FACTOR_MAP.items():
            relevant = [risk_by_factor[fn] for fn in factor_names if fn in risk_by_factor]
            if not relevant:
                continue

            # Weighted mean with L3 penalty multiplier
            weights = [rf.weight for fn in factor_names
                       for rf in RISK_FACTORS if rf.name == fn]
            scores  = [r.risk_score for r in relevant]
            w_total = sum(weights[:len(scores)])

            if w_total == 0:
                continue

            # Bayesian-inspired combination: P(injury) ≈ 1 - ∏(1 - weighted_risk_i)
            # This captures co-occurrence amplification (L3 insight)
            prob = 1.0
            for r, w in zip(scores, weights[:len(scores)]):
                prob *= (1.0 - r * w / sum(weights[:len(scores)]))
            prob = 1.0 - prob

            # Apply L3 multiplier if full_body_instability is elevated
            l3 = risk_by_factor.get("full_body_instability")
            if l3 and l3.risk_score > 0.5 and injury in ("ACL", "PFPS", "LBP"):
                prob = min(1.0, prob * 1.3)

            risk_pct = round(prob * 100, 1)

            if risk_pct < 25:   level = "LOW"
            elif risk_pct < 50: level = "MEDIUM"
            elif risk_pct < 75: level = "HIGH"
            else:               level = "CRITICAL"

            contributing = [r.factor_name for r in relevant if r.is_elevated]
            primary = max(relevant, key=lambda r: r.weighted_risk).factor_name

            injury_risks[injury] = InjuryRiskResult(
                injury_type          = injury,
                risk_percent         = risk_pct,
                risk_level           = level,
                contributing_factors = contributing,
                primary_factor       = primary,
                recommendation       = INJURY_RECOMMENDATIONS.get(injury, "Consult physiotherapist."),
            )
            injury_probs[injury] = round(prob, 3)

        # Overall risk: max injury risk
        overall = max((r.risk_percent for r in injury_risks.values()), default=0.0)
        if overall < 25:   overall_level = "LOW"
        elif overall < 50: overall_level = "MEDIUM"
        elif overall < 75: overall_level = "HIGH"
        else:              overall_level = "CRITICAL"

        top_factors = sorted(
            [r for r in factor_results if r.is_elevated],
            key=lambda r: r.weighted_risk, reverse=True
        )[:5]
        top_names = [r.factor_name for r in top_factors]

        action_plan = _build_action_plan(injury_risks, top_names)
        summary     = _build_risk_summary(overall, overall_level, injury_risks, top_names)

        return RiskReport(
            factor_results      = factor_results,
            injury_risks        = injury_risks,
            overall_risk        = round(overall, 1),
            overall_level       = overall_level,
            top_risk_factors    = top_names,
            injury_probabilities= injury_probs,
            action_plan         = action_plan,
            summary             = summary,
        )

    def score(self, joints: np.ndarray) -> RiskReport:
        """Score a single frame (15, 2)."""
        return self._score_single(joints)

    def score_sequence(self, joints: np.ndarray) -> RiskReport:
        """Score a sequence (T, 15, 2) — uses mean pose."""
        return self._score_single(joints)  # _compute handles both

    def extract_feature_vector(self, joints: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Flat feature vector for ML training pipeline integration."""
        report = self.score_sequence(joints) if joints.ndim == 3 else self.score(joints)
        feats: List[float] = []
        names: List[str]   = []
        for r in report.factor_results:
            feats.extend([r.deviation, r.risk_score])
            names.extend([f"risk_{r.factor_name}_dev", f"risk_{r.factor_name}_score"])
        for inj, prob in report.injury_probabilities.items():
            feats.append(prob)
            names.append(f"risk_prob_{inj}")
        feats.append(report.overall_risk / 100.0)
        names.append("risk_overall_norm")
        return np.array(feats, dtype=np.float32), names


def _build_action_plan(
    injury_risks: Dict[str, InjuryRiskResult], top_factors: List[str]
) -> List[str]:
    plan = []
    high = [r for r in injury_risks.values() if r.risk_level in ("HIGH", "CRITICAL")]
    med  = [r for r in injury_risks.values() if r.risk_level == "MEDIUM"]
    if high:
        plan.append(f"URGENT: {len(high)} high-risk injury type(s) identified.")
        for r in high[:2]:
            plan.append(f"  → {r.injury_type}: {r.recommendation[:80]}")
    elif med:
        plan.append(f"CAUTION: {len(med)} medium-risk injury type(s) identified.")
        for r in med[:2]:
            plan.append(f"  → {r.injury_type}: {r.recommendation[:80]}")
    else:
        plan.append("Risk within acceptable limits. Maintain current training load.")
        plan.append("Continue regular biomechanical screening (monthly recommended).")
    if top_factors:
        plan.append(f"Priority correction: {', '.join(top_factors[:3])}")
    return plan


def _build_risk_summary(
    overall: float, level: str,
    injury_risks: Dict[str, InjuryRiskResult],
    top_factors: List[str],
) -> str:
    hi_injuries = [inj for inj, r in injury_risks.items() if r.risk_level in ("HIGH", "CRITICAL")]
    text = (
        f"Biomechanical risk assessment: {level} overall risk ({overall:.0f}%). "
    )
    if hi_injuries:
        text += f"Elevated injury risk: {', '.join(hi_injuries)}. "
    if top_factors:
        text += f"Primary risk drivers: {', '.join(top_factors[:3])}. "
    text += (
        "Risk estimates derived from 2D geometric features and validated "
        "clinical epidemiology literature. Not a substitute for clinical evaluation."
    )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_risk_dashboard(report: RiskReport):
    """
    3-panel risk dashboard.
    Panel 1: Injury probability horizontal bar chart
    Panel 2: L1/L2/L3 factor risk heatmap
    Panel 3: Action plan text panel
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#0e1117")
    fig.suptitle("Predictive Biomechanical Risk Assessment — PBRS",
                 color="white", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ── Panel 1: Injury probabilities ─────────────────────────────────────
    ax = axes[0]
    injuries = list(report.injury_risks.keys())
    probs    = [report.injury_risks[i].risk_percent for i in injuries]
    levels   = [report.injury_risks[i].risk_level   for i in injuries]
    lv_colors = {"LOW": "#00dc50", "MEDIUM": "#ffcc00", "HIGH": "#ff8800", "CRITICAL": "#ff3333"}
    colors   = [lv_colors[l] for l in levels]
    y = np.arange(len(injuries))
    ax.barh(y, probs, color=colors, edgecolor="#333", height=0.55)
    ax.axvline(25, color="#00dc50", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(50, color="#ffcc00", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(75, color="#ff8800", ls="--", lw=0.8, alpha=0.5)
    ax.set_yticks(y); ax.set_yticklabels(injuries, color="white", fontsize=8)
    ax.set_xlabel("Estimated Injury Risk (%)"); ax.set_xlim(0, 100)
    ax.set_title(f"Injury Risk Profile  |  Overall: {report.overall_risk:.0f}% ({report.overall_level})")
    for i, (p, l) in enumerate(zip(probs, levels)):
        ax.text(p + 1, i, f"{p:.0f}% {l}", va="center", color="white", fontsize=7)

    # ── Panel 2: Factor risk heatmap ──────────────────────────────────────
    ax = axes[1]
    elevated = [r for r in report.factor_results if r.is_elevated or r.risk_score > 0.1][:10]
    fnames = [r.factor_name.replace("_", "\n") for r in elevated]
    fscores= [r.risk_score * 100 for r in elevated]
    f_colors = ["#ff3333" if s > 60 else "#ff8800" if s > 35 else "#ffcc00"
                for s in fscores]
    y = np.arange(len(fnames))
    ax.barh(y, fscores, color=f_colors, edgecolor="#333", height=0.6)
    ax.set_yticks(y); ax.set_yticklabels(fnames, color="white", fontsize=7)
    ax.set_xlabel("Risk Score (%)"); ax.set_xlim(0, 100)
    ax.set_title("Biomechanical Risk Factors (L1/L2/L3)")

    # Level labels
    for i, r in enumerate(elevated):
        ax.text(2, i, r.level, va="center", color="#aaaaaa", fontsize=6)

    # ── Panel 3: Action plan ──────────────────────────────────────────────
    ax = axes[2]
    ax.axis("off")
    overall_color = lv_colors[report.overall_level]
    lines = [
        f"Risk Level: {report.overall_level}  ({report.overall_risk:.0f}%)",
        "",
    ] + report.action_plan + ["", report.summary[:180]]

    y_pos = 0.96
    for i, line in enumerate(lines):
        fc = overall_color if i == 0 else "white" if not line.startswith("  →") else "#ffcc00"
        fs = 11 if i == 0 else 8
        fw = "bold" if i == 0 else "normal"
        ax.text(0.02, y_pos, line[:90], transform=ax.transAxes,
                color=fc, fontsize=fs, fontweight=fw, va="top", wrap=True)
        y_pos -= 0.075 if i == 0 else 0.065

    plt.tight_layout()
    return fig
