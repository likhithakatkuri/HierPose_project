"""
Gait Lab — Hierarchical Gait Decomposition (HGD) + Kinematic Compensation + Injury Risk
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st

st.set_page_config(page_title="Gait Lab", page_icon="🦶", layout="wide")

# ── Auth gate ──────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.warning("Please log in from the Home page.")
    st.stop()

# ── MediaPipe → JHMDB ─────────────────────────────────────────────────────
# JHMDB: NECK=0 BELLY=1 FACE=2 R_SHOULDER=3 L_SHOULDER=4
#        R_HIP=5 L_HIP=6 R_ELBOW=7 L_ELBOW=8 R_KNEE=9 L_KNEE=10
#        R_WRIST=11 L_WRIST=12 R_ANKLE=13 L_ANKLE=14
# MediaPipe: 0=nose 11=L_shoulder 12=R_shoulder 13=L_elbow 14=R_elbow
#            15=L_wrist 16=R_wrist 23=L_hip 24=R_hip 25=L_knee 26=R_knee
#            27=L_ankle 28=R_ankle
MP2JHMDB = {
    0:2,   # nose   → FACE
    12:3,  # R_shoulder → R_SHOULDER
    11:4,  # L_shoulder → L_SHOULDER
    24:5,  # R_hip  → R_HIP
    23:6,  # L_hip  → L_HIP
    14:7,  # R_elbow → R_ELBOW
    13:8,  # L_elbow → L_ELBOW
    26:9,  # R_knee → R_KNEE
    25:10, # L_knee → L_KNEE
    16:11, # R_wrist → R_WRIST
    15:12, # L_wrist → L_WRIST
    28:13, # R_ankle → R_ANKLE
    27:14, # L_ankle → L_ANKLE
}

def mp_to_jhmdb(lm) -> np.ndarray:
    j = np.zeros((15,2), dtype=np.float32)
    for mp_i, jh_i in MP2JHMDB.items():
        if mp_i < len(lm):
            j[jh_i] = [lm[mp_i].x, lm[mp_i].y]
    # Derived joints: NECK = midpoint(shoulders), BELLY = midpoint(hips)
    if j[3].sum() and j[4].sum():
        j[0] = (j[3] + j[4]) / 2   # NECK
    if j[5].sum() and j[6].sum():
        j[1] = (j[5] + j[6]) / 2   # BELLY
    return j

# ── Page header ────────────────────────────────────────────────────────────
st.markdown("# 🦶 Gait Lab")
st.markdown(
    "**Research module:** Hierarchical Gait Decomposition (HGD) + "
    "Kinematic Chain Compensation (HKRA) + Predictive Injury Risk (PBRS)"
)
st.markdown("---")

tab_video, tab_live, tab_about = st.tabs(["📹 Video Analysis", "📷 Live Assessment", "📖 Algorithm"])

# ── Helpers ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_mediapipe():
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return pose, mp

def extract_joints_from_video(video_bytes: bytes):
    """Extract (T, 15, 2) joint sequence from video bytes."""
    import tempfile, cv2
    pose_mp, mp = load_mediapipe()
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    frames_joints = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_mp.process(rgb)
        if result.pose_landmarks:
            j = mp_to_jhmdb(result.pose_landmarks.landmark)
            frames_joints.append(j)
    cap.release()
    import os; os.unlink(tmp.name)
    return np.array(frames_joints, dtype=np.float32) if frames_joints else None, fps


def run_full_analysis(joints_seq: np.ndarray, fps: float):
    """Run HGD + HKRA + PBRS on a joint sequence."""
    from psrn.features.gait import HierarchicalGaitFeatureExtractor, plot_gait_dashboard
    from psrn.domains.compensation import KinematicChainCompensationDetector, plot_compensation_map
    from psrn.domains.injury_risk import PredictiveBiomechanicalRiskScorer, plot_risk_dashboard

    gait_extractor = HierarchicalGaitFeatureExtractor(fps=fps)
    comp_detector  = KinematicChainCompensationDetector()
    risk_scorer    = PredictiveBiomechanicalRiskScorer()

    gait_report = gait_extractor.analyse(joints_seq)
    comp_report = comp_detector.analyse_sequence(joints_seq)
    risk_report = risk_scorer.score_sequence(joints_seq)

    gait_fig = plot_gait_dashboard(gait_report)
    comp_fig = plot_compensation_map(comp_report)
    risk_fig = plot_risk_dashboard(risk_report)

    return gait_report, comp_report, risk_report, gait_fig, comp_fig, risk_fig


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — VIDEO ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_video:
    st.markdown("### Upload a walking/movement video for full gait analysis")
    st.info(
        "Best results: 5–30 second clip of patient walking straight toward/away from camera "
        "or in profile view. Full body must be visible."
    )

    video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if video_file:
        with st.spinner("Extracting pose sequences... (MediaPipe)"):
            joints_seq, fps = extract_joints_from_video(video_file.read())

        if joints_seq is None or len(joints_seq) < 15:
            st.error("Could not detect sufficient pose frames. Ensure full body is visible.")
            st.stop()

        st.success(f"Extracted {len(joints_seq)} frames at {fps:.0f} fps.")

        with st.spinner("Running HGD + HKRA + PBRS analysis..."):
            try:
                gait_r, comp_r, risk_r, gait_fig, comp_fig, risk_fig = run_full_analysis(joints_seq, fps)
            except Exception as e:
                st.error(f"Analysis error: {e}")
                st.stop()

        # ── Summary metrics ──────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        gait_color  = "normal" if gait_r.overall_gait_score >= 75 else "inverse"
        comp_color  = "inverse" if comp_r.overall_severity in ("moderate","severe") else "normal"
        risk_level  = risk_r.overall_level
        risk_delta  = f"{risk_r.overall_risk:.0f}%"

        c1.metric("Gait Score", f"{gait_r.overall_gait_score:.0f}/100",
                  delta=f"GDI: {gait_r.gait_deviation_index:.0f}")
        c2.metric("Cadence", f"{gait_r.cadence_spm:.0f} spm",
                  delta=f"SI: {gait_r.step_symmetry_index:.1f}%")
        c3.metric("Compensation", comp_r.overall_severity.upper(),
                  delta=f"Score: {comp_r.compensation_score:.0f}/100")
        c4.metric("Injury Risk", risk_level, delta=risk_delta)

        st.markdown("---")

        # ── HGD Dashboard ────────────────────────────────────────────────
        st.markdown("#### Level 1–3: Hierarchical Gait Analysis")
        st.pyplot(gait_fig)

        # ── Risk flags ────────────────────────────────────────────────────
        if gait_r.risk_flags:
            st.markdown("**Clinical Flags from HGD:**")
            for flag in gait_r.risk_flags:
                level_icon = "🔴" if "Severe" in flag or "Significant" in flag else "🟡"
                st.markdown(f"- {level_icon} {flag}")

        st.markdown("---")

        # ── HKRA Dashboard ───────────────────────────────────────────────
        st.markdown("#### HKRA — Kinematic Chain Compensation Map")
        st.pyplot(comp_fig)

        if comp_r.root_causes:
            st.markdown(f"**Root Cause(s):** `{'`, `'.join(comp_r.root_causes)}`")
        st.caption(comp_r.summary)

        st.markdown("---")

        # ── PBRS Dashboard ───────────────────────────────────────────────
        st.markdown("#### PBRS — Predictive Biomechanical Injury Risk")
        st.pyplot(risk_fig)

        with st.expander("View Action Plan"):
            for line in risk_r.action_plan:
                st.markdown(f"- {line}")

        # ── Detailed tables ───────────────────────────────────────────────
        with st.expander("L2/L3 Gait Parameters (Full Table)"):
            import pandas as pd
            gait_data = {
                "Parameter": [
                    "Cadence (spm)", "Gait Deviation Index", "Step Symmetry Index (%)",
                    "Stance Ratio R", "Stance Ratio L",
                    "HS Knee Angle R (°)", "HS Knee Angle L (°)",
                    "Peak Knee Flexion Swing R (°)", "Peak Knee Flexion Swing L (°)",
                    "Hip Flexion ROM R (°)", "Hip Flexion ROM L (°)",
                    "Trunk Sway Range (°)", "Hip Drop R (°)", "Hip Drop L (°)",
                    "Bilateral Waveform Corr (r)",
                ],
                "Value": [
                    f"{gait_r.cadence_spm:.1f}",
                    f"{gait_r.gait_deviation_index:.1f}",
                    f"{gait_r.step_symmetry_index:.1f}",
                    f"{gait_r.stance_ratio_r:.2f}",
                    f"{gait_r.stance_ratio_l:.2f}",
                    f"{gait_r.heel_strike_knee_angle.get('right', 0):.1f}",
                    f"{gait_r.heel_strike_knee_angle.get('left', 0):.1f}",
                    f"{gait_r.peak_knee_flexion_swing.get('right', 0):.1f}",
                    f"{gait_r.peak_knee_flexion_swing.get('left', 0):.1f}",
                    f"{gait_r.hip_flexion_range.get('right', 0):.1f}",
                    f"{gait_r.hip_flexion_range.get('left', 0):.1f}",
                    f"{gait_r.trunk_sway_range_deg:.1f}",
                    f"{gait_r.hip_drop_r:.1f}",
                    f"{gait_r.hip_drop_l:.1f}",
                    f"{gait_r.bilateral_waveform_corr:.3f}",
                ],
                "Normal Range": [
                    "100–120", "> 88", "< 5%", "0.58–0.62", "0.58–0.62",
                    "0–10°", "0–10°", "60–70°", "60–70°",
                    "35–45°", "35–45°", "< 8°", "< 5°", "< 5°", "> 0.85",
                ],
            }
            st.dataframe(pd.DataFrame(gait_data), use_container_width=True)

        st.caption(f"*{gait_r.interpretation}*")

        # ── LLM Narratives ────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🤖 AI Clinical Narrative")
        st.caption("Streaming explanation powered by Groq llama-3.3-70b")

        llm_col1, llm_col2, llm_col3 = st.columns(3)

        with llm_col1:
            st.markdown("**Gait Analysis**")
            gait_box = st.empty()
            try:
                from app.utils.llm import gait_report_narrative, render_stream
                render_stream(gait_box, gait_report_narrative({
                    "overall_gait_score": gait_r.overall_gait_score,
                    "gait_deviation_index": gait_r.gait_deviation_index,
                    "cadence_spm": gait_r.cadence_spm,
                    "step_symmetry_index": gait_r.step_symmetry_index,
                    "stance_ratio_r": gait_r.stance_ratio_r,
                    "trunk_sway_range_deg": gait_r.trunk_sway_range_deg,
                    "hip_drop_r": gait_r.hip_drop_r,
                    "hip_drop_l": gait_r.hip_drop_l,
                    "bilateral_waveform_corr": gait_r.bilateral_waveform_corr,
                    "risk_flags": gait_r.risk_flags,
                }, stream=True))
            except Exception as _e:
                gait_box.warning(f"LLM unavailable: {_e}")

        with llm_col2:
            st.markdown("**Compensation Pattern**")
            comp_box = st.empty()
            try:
                from app.utils.llm import compensation_narrative, render_stream
                render_stream(comp_box, compensation_narrative({
                    "overall_severity": comp_r.overall_severity,
                    "compensation_score": comp_r.compensation_score,
                    "root_causes": comp_r.root_causes,
                    "compensations": [
                        {"joint_name": c.joint_name, "residual": c.residual,
                         "severity": c.severity, "direction": c.direction}
                        for c in comp_r.compensations
                    ],
                }, stream=True))
            except Exception as _e:
                comp_box.warning(f"LLM unavailable: {_e}")

        with llm_col3:
            st.markdown("**Injury Risk**")
            risk_box = st.empty()
            try:
                from app.utils.llm import injury_risk_narrative, render_stream
                render_stream(risk_box, injury_risk_narrative({
                    "overall_level": risk_r.overall_level,
                    "overall_risk": risk_r.overall_risk,
                    "top_risk_factors": risk_r.top_risk_factors,
                    "injury_risks": {
                        k: {"risk_level": v.risk_level, "risk_percent": v.risk_percent}
                        for k, v in risk_r.injury_risks.items()
                    },
                }, stream=True))
            except Exception as _e:
                risk_box.warning(f"LLM unavailable: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE ASSESSMENT (single frame + 3-module analysis)
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("### Single-frame Biomechanical Snapshot")
    st.info(
        "Stand in profile view (side-on) for gait-relevant analysis, or face camera directly "
        "for upper-body compensation and injury risk. Keep full body visible."
    )

    img_data = st.camera_input("Take pose snapshot")

    if img_data:
        import cv2
        from PIL import Image
        import io

        pose_mp, mp = load_mediapipe()
        img = Image.open(img_data)
        img_array = np.array(img)
        rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        result = pose_mp.process(rgb)

        if not result.pose_landmarks:
            st.error("No pose detected. Ensure full body is visible in good lighting.")
        else:
            joints = mp_to_jhmdb(result.pose_landmarks.landmark)

            # Annotate skeleton on image
            mp_drawing = mp.solutions.drawing_utils
            mp_pose    = mp.solutions.pose
            annotated  = img_array.copy()
            mp_drawing.draw_landmarks(
                annotated, result.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0,255,100), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0,150,255), thickness=2),
            )
            st.image(annotated, caption="Detected Skeleton", use_column_width=True)

            with st.spinner("Running HKRA + PBRS..."):
                from psrn.domains.compensation import KinematicChainCompensationDetector, plot_compensation_map
                from psrn.domains.injury_risk import PredictiveBiomechanicalRiskScorer, plot_risk_dashboard

                comp_r = KinematicChainCompensationDetector().analyse(joints)
                risk_r = PredictiveBiomechanicalRiskScorer().score(joints)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### HKRA — Compensation Map")
                comp_fig = plot_compensation_map(comp_r)
                st.pyplot(comp_fig)
                st.markdown(f"**Severity:** `{comp_r.overall_severity.upper()}`")
                if comp_r.root_causes:
                    st.markdown(f"**Root causes:** {', '.join(comp_r.root_causes)}")
                active = [c for c in comp_r.compensations if c.severity != "none"]
                if active:
                    import pandas as pd
                    comp_df = pd.DataFrame([{
                        "Joint": c.joint_name,
                        "Actual (°)": c.actual_angle,
                        "Expected (°)": c.expected_angle,
                        "Residual (°)": f"{c.residual:+.1f}",
                        "Severity": c.severity,
                        "Root Cause": "★" if c.is_root_cause else "",
                    } for c in active])
                    st.dataframe(comp_df, use_container_width=True)

            with col2:
                st.markdown("#### PBRS — Injury Risk")
                risk_fig = plot_risk_dashboard(risk_r)
                st.pyplot(risk_fig)
                for inj, res in risk_r.injury_risks.items():
                    icon = "🔴" if res.risk_level in ("HIGH","CRITICAL") else "🟡" if res.risk_level == "MEDIUM" else "🟢"
                    st.markdown(f"{icon} **{inj}**: {res.risk_percent:.0f}% ({res.risk_level})")

            with st.expander("Clinical Interpretation"):
                st.write(comp_r.summary)
                st.write(risk_r.summary)
                st.write("\n**Action Plan:**")
                for line in risk_r.action_plan:
                    st.markdown(f"- {line}")

            # ── LLM Snapshot Commentary ────────────────────────────────
            st.markdown("---")
            st.markdown("### 🤖 AI Commentary (Snapshot)")
            snap_col1, snap_col2 = st.columns(2)
            with snap_col1:
                st.markdown("**Compensation Analysis**")
                snap_comp_box = st.empty()
                try:
                    from app.utils.llm import compensation_narrative, render_stream
                    render_stream(snap_comp_box, compensation_narrative({
                        "overall_severity": comp_r.overall_severity,
                        "compensation_score": comp_r.compensation_score,
                        "root_causes": comp_r.root_causes,
                        "compensations": [
                            {"joint_name": c.joint_name, "residual": c.residual,
                             "severity": c.severity, "direction": c.direction}
                            for c in comp_r.compensations
                        ],
                    }, stream=True))
                except Exception as _e:
                    snap_comp_box.warning(f"LLM unavailable: {_e}")

            with snap_col2:
                st.markdown("**Injury Risk**")
                snap_risk_box = st.empty()
                try:
                    from app.utils.llm import injury_risk_narrative, render_stream
                    render_stream(snap_risk_box, injury_risk_narrative({
                        "overall_level": risk_r.overall_level,
                        "overall_risk": risk_r.overall_risk,
                        "top_risk_factors": risk_r.top_risk_factors,
                        "injury_risks": {
                            k: {"risk_level": v.risk_level, "risk_percent": v.risk_percent}
                            for k, v in risk_r.injury_risks.items()
                        },
                    }, stream=True))
                except Exception as _e:
                    snap_risk_box.warning(f"LLM unavailable: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ALGORITHM DESCRIPTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## Research Algorithms in This Module

### 1. Hierarchical Gait Decomposition (HGD)
A 3-level decomposition of gait kinematics from 2D body keypoints:

| Level | Name | Features | Clinical Output |
|-------|------|----------|----------------|
| L1 | Joint | Per-joint angle time-series (7 bilateral pairs × T frames) | Bilateral angle profiles |
| L2 | Segment | Gait-cycle events, phase durations, cadence, Robinson Symmetry Index | Cadence, stance/swing ratios, SI |
| L3 | Whole-body | GDI proxy, bilateral waveform correlation, trunk sway, Trendelenburg | Overall gait quality score |

**Gait event detection** (Pijnappels et al. 2001): ankle Y-coordinate local maxima
→ heel-strikes. Gaussian smoothed to suppress noise.

**Symmetry Index** (Robinson et al. 1987):
> SI = 2|R − L| / (R + L) × 100%  (0 = perfect symmetry)

**Gait Deviation Index** (Schwartz & Rozumalski 2008):
> GDI = 100 − 10 × ||z-scored deviation from normative angles||₂

---

### 2. Hierarchical Kinematic Residual Analysis (HKRA)
Model the body as a directed kinematic chain G = (V, E):
- For each edge (i→j): `expected_angle_j = β_ij × angle_i + α_ij`
- `residual_j = actual_angle_j − expected_angle_j`
- If |residual_j| > threshold → **compensation detected**
- Root-cause propagation: most proximal joint with unexplained deviation = root cause

Coefficients β, α derived from Winter (2009) normative biomechanics tables.

---

### 3. Predictive Biomechanical Risk Scoring (PBRS)
Multi-factor injury risk model with **3-level hierarchy**:

| Level | Factors |
|-------|---------|
| L1 (Joint) | Knee valgus, dorsiflexion deficit, trunk lean |
| L2 (Segment) | Hip adduction, Trendelenburg sign, bilateral asymmetry |
| L3 (Whole-body) | Co-occurrence composite instability pattern |

Per-injury probability via Bayesian combination:
> P(injury) = 1 − ∏ᵢ (1 − wᵢ × rᵢ / Σwᵢ)

L3 applies a 1.3× multiplier when ≥ 3 simultaneous L1/L2 factors are elevated.

**Clinical weights** derived from published odds ratios (Hewett et al. 2005,
Witvrouw et al. 2014, Hayden et al. 2005, Ludewig & Cook 2000).

---

### Thesis Alignment
All three algorithms share the same hierarchical geometric feature structure
as the core HierPose framework (L1 joint → L2 segment → L3 whole-body),
making them jointly publishable as a unified contribution.
""")
