"""
AI Pose Coach — Exercise Library, Search, LLM-generated animations & instructions.
"""
from __future__ import annotations
import sys, os, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import streamlit as st

st.set_page_config(page_title="AI Pose Coach", page_icon="🤸", layout="wide")

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.warning("Please sign in from the Home page.")
    st.stop()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from app.utils.skeleton_animator import (
        EXERCISE_ANIMATIONS, EXERCISE_LABELS, get_label,
        create_exercise_gif, create_comparison_gif,
        create_dual_exercise_gif, create_dual_custom_gif,
        render_static_skeleton, joints_from_mediapipe,
    )
    from app.utils.llm import (
        exercise_guide, live_pose_commentary, render_stream,
        generate_exercise_pose,
    )
    UTILS_OK = True
except ImportError as e:
    UTILS_OK = False
    _ERR = str(e)

if not UTILS_OK:
    st.error(f"Import error: {_ERR}")
    st.stop()

# ── Exercise catalogue ────────────────────────────────────────────────────────
EXERCISE_CATALOGUE = {
    # key: (display name, joint_focus, category)
    # ── Built-in (has pre-defined animation) ──────────────────────────────
    "squat":                   ("Squat",                    "knee, hip, ankle",          "Strength — Lower Body"),
    "lunge":                   ("Lunge",                    "knee, hip, ankle",          "Strength — Lower Body"),
    "deadlift":                ("Deadlift",                 "hip, spine, knee",          "Strength — Lower Body"),
    "romanian deadlift":       ("Romanian Deadlift",        "hip, hamstring",            "Strength — Lower Body"),
    "step up":                 ("Step-Up",                  "knee, hip",                 "Strength — Lower Body"),
    "calf raise":              ("Calf Raise",               "ankle, calf",               "Strength — Lower Body"),
    "glute kickback":          ("Glute Kickback",           "hip, glute",                "Strength — Lower Body"),
    "push-up":                 ("Push-Up",                  "elbow, shoulder, core",     "Strength — Upper Body"),
    "overhead press":          ("Overhead Press",           "shoulder, elbow",           "Strength — Upper Body"),
    "lateral raise":           ("Lateral Raise",            "shoulder",                  "Strength — Upper Body"),
    "bicep curl":              ("Bicep Curl",               "elbow, bicep",              "Strength — Upper Body"),
    "tricep dip":              ("Tricep Dip",               "elbow, tricep",             "Strength — Upper Body"),
    "plank":                   ("Plank",                    "core, shoulder, spine",     "Core"),
    "side plank":              ("Side Plank",               "core, hip, shoulder",       "Core"),
    "hip bridge":              ("Hip Bridge / Glute Bridge","hip, glute, core",          "Core"),
    "mountain climber":        ("Mountain Climber",         "core, hip, shoulder",       "Core"),
    "bird dog":                ("Bird Dog",                 "core, hip, shoulder",       "Core"),
    "shoulder abduction":      ("Shoulder Abduction",       "shoulder",                  "Rehabilitation"),
    "knee flexion":            ("Knee Flexion",             "knee",                      "Rehabilitation"),
    "terminal knee extension": ("Terminal Knee Extension",  "knee, quad",                "Rehabilitation"),
    "heel slide":              ("Heel Slide",               "knee, hip",                 "Rehabilitation"),
    "seated row":              ("Seated Row",               "shoulder, elbow, back",     "Rehabilitation"),
    "hamstring stretch":       ("Hamstring Stretch",        "hamstring, hip",            "Flexibility"),
    "hip flexor stretch":      ("Hip Flexor Stretch",       "hip flexor, quad",          "Flexibility"),
    "standing":                ("Standing Posture",         "spine, neck, knee",         "Posture"),
    "walking gait":            ("Walking Gait",             "hip, knee, ankle",          "Gait"),
    "xray pa chest":           ("PA Chest X-Ray Position",  "shoulder, spine",           "Medical Positioning"),
}

CATEGORIES = sorted(set(v[2] for v in EXERCISE_CATALOGUE.values()))

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🤸 AI Pose Coach")
st.markdown(
    "Search any exercise → get an **animated skeleton demo** and "
    "**AI-powered step-by-step instructions**. "
    "Choose from the built-in library or search any exercise to generate a custom demo."
)
st.markdown("---")

# ── Search box ────────────────────────────────────────────────────────────────
search_col, cat_col = st.columns([3, 1])
with search_col:
    query = st.text_input(
        "Search exercise",
        placeholder="e.g.  squat, bicep curl, nordic hamstring, turkish get-up …",
        label_visibility="collapsed",
    )
with cat_col:
    cat_filter = st.selectbox("Category", ["All"] + CATEGORIES, label_visibility="collapsed")

# ── Filter / match logic ──────────────────────────────────────────────────────
def _matches(key: str, name: str, cat: str) -> bool:
    q = query.strip().lower()
    cat_ok = (cat_filter == "All" or cat == cat_filter)
    if not q:
        return cat_ok
    return cat_ok and (q in key or q in name.lower())

matched_keys = [k for k, (n, _, c) in EXERCISE_CATALOGUE.items() if _matches(k, n, c)]
is_custom_search = (
    query.strip() and
    not any(query.strip().lower() in k or query.strip().lower() in v[0].lower()
            for k, v in EXERCISE_CATALOGUE.items())
)

# ── Show library grid ────────────────────────────────────────────────────────
if matched_keys and not is_custom_search:
    st.markdown(f"**{len(matched_keys)} exercise{'s' if len(matched_keys)!=1 else ''} found**")

    # Display as card grid (3 per row)
    cols_per_row = 3
    for row_start in range(0, len(matched_keys), cols_per_row):
        row_keys = matched_keys[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, key in zip(cols, row_keys):
            name, jfocus, cat = EXERCISE_CATALOGUE[key]
            has_anim = key in EXERCISE_ANIMATIONS
            with col:
                st.markdown(f"""
                <div style='background:#1a2535;border-radius:10px;padding:0.8rem;
                            margin-bottom:0.5rem;border:1px solid #2a3a50;'>
                  <b style='color:#7ec8e3;'>{name}</b><br>
                  <small style='color:#aaa;'>📍 {jfocus}</small><br>
                  <small style='color:#666;'>{cat}</small>
                  {"<br><small style='color:#3a8'>▶ Animated</small>" if has_anim else ""}
                </div>
                """, unsafe_allow_html=True)
                if st.button(f"Open →", key=f"open_{key}"):
                    st.session_state["selected_exercise"] = key
                    st.session_state["custom_exercise"] = None
                    st.rerun()

# ── Custom search — LLM generation ───────────────────────────────────────────
elif is_custom_search:
    ex_name = query.strip()
    st.info(
        f'**"{ex_name}"** is not in the built-in library. '
        "Click below to generate an AI skeleton demo using LLM."
    )
    if st.button(f"Generate AI Demo for: {ex_name}", type="primary"):
        st.session_state["selected_exercise"] = None
        st.session_state["custom_exercise"] = ex_name
        st.rerun()

elif query and not matched_keys:
    st.warning("No exercises matched. Try a different term or remove the category filter.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# DETAIL VIEW — built-in exercise
# ═══════════════════════════════════════════════════════════════════════════════
sel = st.session_state.get("selected_exercise")
custom = st.session_state.get("custom_exercise")

if sel and sel in EXERCISE_CATALOGUE:
    name, jfocus, cat = EXERCISE_CATALOGUE[sel]
    anim_key = sel  # same key in EXERCISE_ANIMATIONS

    st.markdown(f"## {name}")
    st.caption(f"Category: **{cat}** | Joints: {jfocus}")

    col_anim, col_guide = st.columns([1, 1])

    # ── Left: dual animation (skeleton + human body) ───────────────────────
    with col_anim:
        st.markdown("### Animated Demo")
        if anim_key in EXERCISE_ANIMATIONS:
            with st.spinner("Rendering skeleton & human body animation…"):
                try:
                    gif_bytes = create_dual_exercise_gif(anim_key, fps=10, size=(820, 460))
                    st.image(gif_bytes, use_container_width=True)
                    st.caption("Left: AI pose skeleton (classification model)  |  Right: realistic human body view")
                except Exception as e:
                    st.warning(f"Dual animation failed: {e} — falling back to skeleton only")
                    try:
                        gif_bytes = create_exercise_gif(anim_key, fps=10, size=(420, 480))
                        st.image(gif_bytes, use_container_width=True, caption=f"{name}")
                    except Exception as e2:
                        frames = EXERCISE_ANIMATIONS[anim_key]
                        if frames:
                            fig = render_static_skeleton(np.array(frames[0]), title=f"{name}")
                            st.pyplot(fig)
        else:
            st.info("No animation for this exercise.")

        st.markdown("""
        <div style='background:#111;border-radius:6px;padding:0.5rem;font-size:0.78rem;color:#888;'>
        🦴 Left: JHMDB 15-joint pose skeleton used in classification &nbsp;|&nbsp;
        🧍 Right: Human body rendering of the same joints
        </div>""", unsafe_allow_html=True)

    # ── Right: LLM instructions ────────────────────────────────────────────
    with col_guide:
        st.markdown("### AI Instructions")
        patient_cond = st.text_input("Patient condition (optional)",
                                      placeholder="e.g. post-knee surgery, shoulder impingement",
                                      key=f"cond_{sel}")
        if st.button("Get AI Instructions", type="primary", key=f"guide_{sel}"):
            guide_box = st.empty()
            try:
                gen = exercise_guide(
                    exercise_name=name,
                    joint_focus=jfocus,
                    patient_condition=patient_cond,
                    stream=True,
                )
                render_stream(guide_box, gen)
            except Exception as e:
                guide_box.error(f"LLM error: {e}")
        else:
            st.markdown("*Press the button above to get step-by-step AI instructions.*")

    if st.button("← Back to library", key="back_sel"):
        st.session_state["selected_exercise"] = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# DETAIL VIEW — custom / LLM-generated exercise
# ═══════════════════════════════════════════════════════════════════════════════
elif custom:
    st.markdown(f"## 🤖 AI-Generated Demo: {custom}")
    st.caption("Pose coordinates generated by Groq llama-3.3-70b")

    # Generate pose data from LLM (cached in session so we don't regenerate on every click)
    cache_key = f"llm_pose_{custom.lower().replace(' ','_')}"
    if cache_key not in st.session_state:
        with st.spinner(f"Asking AI to design '{custom}' skeleton keyframes…"):
            pose_data = generate_exercise_pose(custom)
            st.session_state[cache_key] = pose_data
    else:
        pose_data = st.session_state[cache_key]

    if "_error" in pose_data:
        st.warning(f"Partial result — {pose_data['_error']}")

    col_anim, col_guide = st.columns([1, 1])

    with col_anim:
        st.markdown("### Animated Skeleton Demo")

        # Build numpy keyframe arrays from LLM output
        llm_frames_raw = pose_data.get("frames", [])
        if llm_frames_raw:
            import matplotlib.pyplot as plt
            kf_arrays  = [np.array(f["joints"], dtype=np.float32) for f in llm_frames_raw]
            frame_names = [f["name"] for f in llm_frames_raw]

            with st.spinner("Rendering skeleton & human body animation…"):
                try:
                    gif_bytes = create_dual_custom_gif(
                        kf_arrays, label=custom.title(), fps=8, size=(820, 460)
                    )
                    st.image(gif_bytes, use_container_width=True)
                    st.caption("Left: AI pose skeleton  |  Right: human body — AI-generated keyframes")
                except Exception as e:
                    st.warning(f"Dual animation failed: {e} — showing static frames")
                    static_cols = st.columns(min(len(kf_arrays), 3))
                    for c, (arr, fname) in zip(static_cols, zip(kf_arrays, frame_names)):
                        fig = render_static_skeleton(arr, title=fname)
                        c.pyplot(fig)
                        plt.close(fig)
        else:
            st.warning("No pose frames returned by LLM.")

        # Show key metadata
        st.markdown(f"**Joint focus:** {pose_data.get('joint_focus','—')}")
        st.markdown(f"**Category:** {pose_data.get('category','—')}")
        cues = pose_data.get("cues", [])
        if cues:
            st.markdown("**Key cues:**")
            for cue in cues:
                st.markdown(f"- {cue}")

    with col_guide:
        st.markdown("### AI Instructions")
        instr = pose_data.get("instructions", "")
        if instr:
            st.markdown(instr)
        st.markdown("---")
        st.markdown("#### Full Step-by-Step Guide")
        if st.button("Get Detailed AI Guide", type="primary", key="llm_full_guide"):
            full_box = st.empty()
            try:
                gen = exercise_guide(
                    exercise_name=custom,
                    joint_focus=pose_data.get("joint_focus", ""),
                    patient_condition="",
                    stream=True,
                )
                render_stream(full_box, gen)
            except Exception as e:
                full_box.error(f"LLM error: {e}")

    if st.button("← Back to library", key="back_custom"):
        st.session_state["custom_exercise"] = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT — show popular exercises when nothing selected/searched
# ═══════════════════════════════════════════════════════════════════════════════
else:
    if not query:
        st.markdown("### Popular Exercises")
        popular = ["squat", "push-up", "plank", "deadlift", "lunge",
                   "hip bridge", "overhead press", "shoulder abduction",
                   "knee flexion", "walking gait", "lateral raise", "bird dog"]

        for row_start in range(0, len(popular), 4):
            row_keys = popular[row_start : row_start + 4]
            cols = st.columns(4)
            for col, key in zip(cols, row_keys):
                if key not in EXERCISE_CATALOGUE:
                    continue
                name, jfocus, cat = EXERCISE_CATALOGUE[key]
                with col:
                    st.markdown(f"""
                    <div style='background:#1a2535;border-radius:10px;padding:0.8rem;
                                margin-bottom:0.5rem;border:1px solid #2a3a50;'>
                      <b style='color:#7ec8e3;'>{name}</b><br>
                      <small style='color:#aaa;'>📍 {jfocus}</small>
                    </div>""", unsafe_allow_html=True)
                    if st.button("Open →", key=f"pop_{key}"):
                        st.session_state["selected_exercise"] = key
                        st.session_state["custom_exercise"] = None
                        st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style='background:#1a1a2e;border-radius:8px;padding:1rem;color:#aaa;font-size:0.88rem;'>
        💡 <b>Tip:</b> Search for <i>any</i> exercise — even if it's not in the library.
        The AI will generate a custom skeleton animation and instructions on demand.<br><br>
        Examples: <code>nordic hamstring curl</code> · <code>turkish get-up</code> ·
        <code>box jump</code> · <code>Copenhagen plank</code> · <code>pallof press</code>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:0.76rem;color:#666;text-align:center;'>"
    "AI Pose Coach · Skeleton demos: JHMDB 15-joint model · "
    "LLM: Groq llama-3.3-70b · Not a substitute for clinical advice."
    "</div>",
    unsafe_allow_html=True,
)
