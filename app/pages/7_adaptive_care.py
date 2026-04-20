"""
Adaptive Care Engine — Longitudinal Patient Trajectory & Protocol Generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Adaptive Care Engine", page_icon="🧠", layout="wide")

if "user" not in st.session_state:
    st.warning("Please log in from the Home page.")
    st.stop()

st.markdown("# 🧠 Adaptive Care Engine")
st.markdown(
    "**Research module:** EWMA Fault Memory + Mann-Kendall Trend Test + "
    "Evidence-Based Protocol Generation"
)
st.markdown("---")

# ── Patient selector ───────────────────────────────────────────────────────
try:
    from utils.database import get_patients, get_sessions, dashboard_stats
    patients = get_patients()
except Exception:
    patients = []

if not patients:
    st.info("No patients registered yet. Register patients in the Medical Assistant page first.")
    st.stop()

patient_names = [f"{p['name']} (ID {p['id']})" for p in patients]
sel = st.selectbox("Select Patient", patient_names)
patient = patients[patient_names.index(sel)]
patient_id = patient["id"]

# ── Load and run ACE ───────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def run_ace(patient_id: int):
    from psrn.domains.adaptive import AdaptiveCareEngine, plot_ace_dashboard
    engine = AdaptiveCareEngine()
    engine.load_from_db(patient_id)
    report = engine.analyse(patient_id)
    predictions = engine.session_prediction(patient_id, n_ahead=4)
    fig = plot_ace_dashboard(report)
    return report, predictions, fig

with st.spinner("Running Adaptive Care Engine..."):
    try:
        report, predictions, ace_fig = run_ace(patient_id)
    except Exception as e:
        st.error(f"ACE error: {e}")
        st.stop()

if report.n_sessions == 0:
    st.warning(
        f"No session history found for {patient['name']}. "
        "Complete at least one assessment in the Medical Assistant page to build history."
    )
    st.stop()

# ── Top metrics ────────────────────────────────────────────────────────────
dri = report.discharge_readiness_index
dri_color = "normal" if dri >= 65 else "inverse"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sessions Analysed",   report.n_sessions)
c2.metric("Discharge Readiness", f"{dri:.0f}%",
          delta="Ready" if dri >= 85 else "In Progress" if dri >= 65 else "Escalate")
c3.metric("Regression Alerts",   len(report.regression_alerts),
          delta="Urgent" if report.regression_alerts else "None")
chronic_count = sum(1 for t in report.trajectories.values() if t.is_chronic_fault)
c4.metric("Chronic Faults",      chronic_count)

st.markdown("---")

tab_traj, tab_protocol, tab_predict, tab_about = st.tabs([
    "📈 Trajectories", "💊 Protocol", "🔮 Prediction", "📖 Algorithm"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAJECTORIES
# ══════════════════════════════════════════════════════════════════════════════
with tab_traj:
    st.markdown("### ACE Dashboard — EWMA Trajectories & Mann-Kendall Analysis")
    st.pyplot(ace_fig)

    # ── Trajectory summary table ──────────────────────────────────────────
    st.markdown("#### Per-Joint Trajectory Summary")
    traj_rows = []
    for jn, t in report.trajectories.items():
        if jn == "_overall": continue
        sig = "★" if t.kendall_p < 0.10 else ""
        traj_rows.append({
            "Joint": jn,
            "Sessions": len(t.raw_series),
            "Latest EWMA (%)": t.latest_ewma,
            "Kendall τ": f"{t.kendall_tau:+.3f}",
            "p-value": f"{t.kendall_p:.3f}{sig}",
            "Trend": t.trend,
            "Chronic Fault": "⚠️" if t.is_chronic_fault else "✅",
        })

    if traj_rows:
        df = pd.DataFrame(traj_rows)
        # Colour-code Trend column
        def highlight_trend(row):
            colors = {"IMPROVING": "color: #00dc50",
                      "STABLE":    "color: #ffcc00",
                      "REGRESSING":"color: #ff4444"}
            return [colors.get(row["Trend"], "") if col == "Trend" else ""
                    for col in row.index]
        st.dataframe(df.style.apply(highlight_trend, axis=1), use_container_width=True)
        st.caption("★ = statistically significant trend (p < 0.10, Mann-Kendall)")

    # ── Regression alerts ─────────────────────────────────────────────────
    if report.regression_alerts:
        st.markdown("#### 🔴 Regression Alerts")
        for alert in report.regression_alerts:
            st.error(f"REGRESSING: {alert}")
        st.warning(
            "One or more joints are showing statistically significant compliance regression. "
            "Consider modifying the exercise protocol or scheduling an in-person review."
        )

    # ── Overall interpretation ────────────────────────────────────────────
    with st.expander("Clinical Interpretation"):
        st.write(report.interpretation)
        st.markdown(f"**Discharge Recommendation:** {report.discharge_recommendation}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADAPTIVE PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
with tab_protocol:
    st.markdown("### Auto-Generated Rehabilitation Protocol")
    st.caption("Based on AAOS 2024 clinical practice guidelines + detected chronic faults")

    if report.protocol is None:
        st.success(
            "No chronic faults detected. Patient is achieving compliance targets. "
            "Continue current programme."
        )
    else:
        proto = report.protocol

        # Header
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Generated:** {proto.generated_at}")
            st.markdown(f"**Schedule:** {proto.weekly_schedule}")
            st.markdown(f"**Priority joints:** `{'`, `'.join(proto.priority_order)}`")
        with col2:
            st.metric("Chronic Faults", len(proto.chronic_faults))

        if proto.notes:
            st.warning(proto.notes)

        st.markdown("---")

        # Exercises
        st.markdown("#### 🏋️ Exercise Programme")
        for i, ex in enumerate(proto.exercises, 1):
            with st.expander(f"Exercise {i}: {ex['name']}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    sets = ex.get("sets", "-")
                    reps = ex.get("reps", ex.get("duration", "-"))
                    st.metric("Sets", sets)
                    st.metric("Reps/Duration", reps)
                with col2:
                    st.markdown(ex.get("description", ""))

        # Stretches
        if proto.stretches:
            st.markdown("#### 🧘 Stretching Protocol")
            for s in proto.stretches:
                st.markdown(f"- {s}")

        # Avoid
        if proto.avoid:
            st.markdown("#### ⛔ Contraindications / Avoid")
            for a in proto.avoid:
                st.warning(a)

        # Download protocol as text
        protocol_text = f"""ADAPTIVE CARE ENGINE — REHABILITATION PROTOCOL
Patient ID: {proto.patient_id}
Generated: {proto.generated_at}
Schedule: {proto.weekly_schedule}
Chronic Faults: {', '.join(proto.chronic_faults)}

EXERCISES:
"""
        for i, ex in enumerate(proto.exercises, 1):
            protocol_text += f"\n{i}. {ex['name']} — {ex.get('sets','?')} sets × {ex.get('reps', ex.get('duration','?'))}\n"
            protocol_text += f"   {ex.get('description','')}\n"

        protocol_text += "\nSTRETCHES:\n" + "\n".join(f"- {s}" for s in proto.stretches)
        protocol_text += "\n\nAVOID:\n" + "\n".join(f"- {a}" for a in proto.avoid)
        protocol_text += f"\n\nNOTES: {proto.notes}"
        protocol_text += "\n\nAll recommendations require physiotherapist review before implementation."

        st.download_button(
            "📥 Download Protocol (TXT)",
            data=protocol_text,
            file_name=f"rehab_protocol_patient_{patient_id}.txt",
            mime="text/plain",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Session Outcome Prediction")
    st.caption(
        "EWMA extrapolation with Mann-Kendall trend slope. "
        "Predicts likely compliance for next 4 sessions."
    )

    if not predictions:
        st.info("Insufficient session history for prediction (need ≥ 2 sessions).")
    else:
        import matplotlib.pyplot as plt

        # Filter to meaningful joints
        show_joints = [jn for jn in predictions if jn != "_overall"
                       and jn in report.trajectories
                       and len(report.trajectories[jn].ewma_series) >= 2][:6]

        if show_joints:
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0e1117")
            ax.set_facecolor("#1a1a2e")
            for sp in ax.spines.values(): sp.set_edgecolor("#333")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

            trend_colors = {"IMPROVING": "#00dc50", "STABLE": "#ffcc00", "REGRESSING": "#ff4444"}

            for jn in show_joints:
                traj = report.trajectories[jn]
                hist = traj.ewma_series
                pred = predictions.get(jn, [])
                color = trend_colors[traj.trend]

                x_hist = list(range(len(hist)))
                x_pred = list(range(len(hist) - 1, len(hist) + len(pred)))
                y_pred = [hist[-1]] + pred

                ax.plot(x_hist, hist, color=color, lw=2, label=jn)
                ax.plot(x_pred, y_pred, color=color, lw=1.5, ls="--", alpha=0.7)
                ax.scatter([x_pred[-1]], [y_pred[-1]], color=color, s=40, zorder=5)

            # Shade prediction zone
            if show_joints:
                traj_0 = report.trajectories[show_joints[0]]
                ax.axvspan(len(traj_0.ewma_series) - 1,
                           len(traj_0.ewma_series) + 3,
                           alpha=0.07, color="white", label="Prediction zone")

            ax.axhline(85, color="#00dc50", ls=":", lw=1, alpha=0.5, label="Discharge threshold")
            ax.axhline(65, color="#ff8800", ls=":", lw=1, alpha=0.5, label="Fault threshold")
            ax.set_xlabel("Session Number"); ax.set_ylabel("Compliance (%)")
            ax.set_title("EWMA Trajectory + 4-Session Prediction")
            ax.set_ylim(0, 105)
            ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8, ncol=3)
            st.pyplot(fig)

        # Table
        pred_rows = []
        for jn in show_joints:
            row = {"Joint": jn, "Current": f"{report.trajectories[jn].latest_ewma:.1f}%",
                   "Trend": report.trajectories[jn].trend}
            for i, p in enumerate(predictions.get(jn, [])[:4], 1):
                row[f"Session +{i}"] = f"{p:.1f}%"
            pred_rows.append(row)

        if pred_rows:
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True)

        st.caption(
            "Predictions use EWMA extrapolation with Mann-Kendall trend slope. "
            "Not a clinical guarantee — for planning purposes only."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ALGORITHM
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
## Adaptive Care Engine — Research Algorithms

### 1. EWMA Fault Memory
Per-joint compliance tracked across sessions with Exponentially Weighted Moving Average:

> **EWMA_t = α · compliance_t + (1 − α) · EWMA_{t−1}**

- α = 0.40 gives moderate recency weighting
- Smooths single-session outliers (patient didn't sleep well, etc.)
- A joint is **chronically faulted** if EWMA < 65% for ≥ 3 consecutive sessions

*Reference: Gardner (1985) — Management Science 31(5)*

---

### 2. Mann-Kendall Trend Test
Non-parametric monotonic trend detection — no normality assumption (critical for n < 10):

> **S = Σ_{i<j} sign(x_j − x_i)**
> **Z = (S ± 1) / √Var(S)**  (continuity correction)

| τ | p < 0.10 | Classification |
|---|---------|----------------|
| > +0.10 | Yes | **IMPROVING** ↑ |
| ±0.10 | No | **STABLE** → |
| < −0.10 | Yes | **REGRESSING** ↓ 🔴 |

Advantage over linear regression: robust to outliers, no distribution assumption,
valid for 3–20 data points (typical clinical series).

*Reference: Mann (1945) Econometrica; Kendall (1975) Rank Correlation Methods*

---

### 3. Discharge Readiness Index (DRI)
> **DRI = mean(EWMA_compliance) × (1 − regression_ratio × 0.5)**

| DRI | Recommendation |
|-----|---------------|
| ≥ 85% | Discharge / maintenance phase |
| 65–85% | Continue — progressing |
| 45–65% | Increase frequency |
| < 45% | Escalate — clinical review |

---

### 4. Session Prediction
EWMA extrapolation using trend slope from last 3 sessions:
> **predicted_t = EWMA_last + t × slope**

Slope adjusted by trend direction (regressing = negative cap, stable = zero slope).

---

### Evidence-Based Protocol
Exercise prescriptions from:
- AAOS (2024) Clinical Practice Guidelines
- Kisner & Colby (2017) Therapeutic Exercise
- Mapped to detected chronic fault joints — no generic protocol
""")
