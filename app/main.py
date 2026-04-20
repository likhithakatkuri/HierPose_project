"""HierPose Multi-Domain AI Pose Platform — Main Entry Point."""
import streamlit as st

st.set_page_config(
    page_title="PoseAI — Multi-Domain Pose Intelligence",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Simple credential store (demo — replace with DB in production) ─────────────
DEMO_USERS = {
    "hospital_admin": {
        "password": "med2025",
        "org": "Apollo Orthopaedic Centre",
        "role": "Medical",
        "access": ["medical", "action", "ergonomics"],
    },
    "physio_user": {
        "password": "rehab123",
        "org": "City Physiotherapy Clinic",
        "role": "Physiotherapist",
        "access": ["medical", "sports"],
    },
    "coach_user": {
        "password": "sport123",
        "org": "Elite Sports Academy",
        "role": "Sports Coach",
        "access": ["sports", "action"],
    },
    "safety_user": {
        "password": "safe2025",
        "org": "SafeWork Industries",
        "role": "Safety Officer",
        "access": ["ergonomics", "action"],
    },
    "demo": {
        "password": "demo",
        "org": "Demo Organisation",
        "role": "Guest",
        "access": ["medical", "sports", "action", "ergonomics"],
    },
}

# ── Session state defaults ────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# ── Login wall ────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    col_left, col_mid, col_right = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("""
        <div style='text-align:center; padding: 2rem 0 1rem 0;'>
            <h1>🦴 PoseAI Platform</h1>
            <p style='color:gray; font-size:1.1rem;'>
                Multi-Domain AI Pose Intelligence<br>
                Medical · Rehabilitation · Sports · Ergonomics · Action Recognition
            </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form"):
            st.subheader("Sign In")
            username = st.text_input("Username", placeholder="e.g. demo")
            password = st.text_input("Password", type="password", placeholder="e.g. demo")
            submitted = st.form_submit_button("Sign In", use_container_width=True, type="primary")

            if submitted:
                user = DEMO_USERS.get(username)
                if user and user["password"] == password:
                    st.session_state.logged_in = True
                    st.session_state.user = {
                        "username": username,
                        "org":      user["org"],
                        "role":     user["role"],
                        "access":   user["access"],
                    }
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        st.markdown("---")
        st.markdown("**Demo accounts** (username / password):")
        cols = st.columns(3)
        demos = [
            ("demo / demo", "Full access – Guest"),
            ("hospital_admin / med2025", "Medical – Apollo Centre"),
            ("physio_user / rehab123", "Physiotherapy Clinic"),
            ("coach_user / sport123", "Sports Academy"),
            ("safety_user / safe2025", "Workplace Safety"),
        ]
        for i, (cred, label) in enumerate(demos):
            cols[i % 3].code(cred)
            cols[i % 3].caption(label)

    st.stop()

# ── Logged-in nav ─────────────────────────────────────────────────────────────
user = st.session_state.user

st.sidebar.markdown(f"### 👤 {user['org']}")
st.sidebar.caption(f"Role: {user['role']}")
st.sidebar.markdown("---")

if st.sidebar.button("🚪 Sign Out"):
    st.session_state.logged_in = False
    st.session_state.user = None
    st.rerun()

# ── Home page ─────────────────────────────────────────────────────────────────
st.title("🦴 PoseAI — Multi-Domain AI Pose Intelligence")
st.markdown(f"Welcome, **{user['org']}** | Role: `{user['role']}`")

DOMAIN_CARDS = [
    {
        "key": "medical",
        "icon": "🏥",
        "title": "Medical & Rehabilitation",
        "desc": "X-ray positioning · Orthopaedic rehab · PT exercise monitoring · Current vs ideal pose overlay",
        "page": "pages/2_medical_assistant.py",
        "color": "#1e3a5f",
    },
    {
        "key": "sports",
        "icon": "🏋️",
        "title": "Sports & Fitness Coach",
        "desc": "Rep counting · Form scoring · Joint angle tracking · Push-ups, squats, and more",
        "page": "pages/3_fitness_coach.py",
        "color": "#1e4a2a",
    },
    {
        "key": "action",
        "icon": "🎬",
        "title": "Action Recognition",
        "desc": "JHMDB 21-class recognition · Video / webcam / .mat file · Confidence + skeleton",
        "page": "pages/1_action_recognition.py",
        "color": "#3a1e4a",
    },
    {
        "key": "ergonomics",
        "icon": "🖥️",
        "title": "Workplace Ergonomics",
        "desc": "RULA-proxy posture risk · Injury prevention · Real-time risk scoring",
        "page": "pages/4_ergonomics.py",
        "color": "#4a3a1e",
    },
    {
        "key": "medical",
        "icon": "🦶",
        "title": "Gait Lab (HGD + HKRA + PBRS)",
        "desc": "Hierarchical Gait Decomposition · Kinematic Chain Compensation · Predictive Injury Risk",
        "page": "pages/6_gait_lab.py",
        "color": "#1a3a4a",
    },
    {
        "key": "medical",
        "icon": "🧠",
        "title": "Adaptive Care Engine",
        "desc": "Cross-session fault memory · EWMA + Mann-Kendall · Auto protocol generation · Discharge index",
        "page": "pages/7_adaptive_care.py",
        "color": "#2a1a4a",
    },
    {
        "key": "sports",
        "icon": "🤸",
        "title": "AI Pose Coach",
        "desc": "Animated skeleton demos · LLM exercise guidance · Real-time form feedback · Current vs ideal comparison",
        "page": "pages/8_pose_coach.py",
        "color": "#1a4a2a",
    },
]

cols = st.columns(2)
for i, card in enumerate(DOMAIN_CARDS):
    access = user["access"]
    locked = card["key"] not in access
    with cols[i % 2]:
        st.markdown(f"""
        <div style='background:{card["color"]}; border-radius:12px; padding:1.5rem; margin-bottom:1rem;
                    opacity:{"0.45" if locked else "1"};'>
            <h3>{card["icon"]} {card["title"]}{"&nbsp;&nbsp;<span style='font-size:0.7rem;background:#555;padding:2px 8px;border-radius:8px;'>LOCKED</span>" if locked else ""}</h3>
            <p style='color:#ccc; font-size:0.9rem;'>{card["desc"]}</p>
        </div>
        """, unsafe_allow_html=True)
        if not locked:
            st.page_link(card["page"], label=f"Open {card['title']} →")
        else:
            st.caption("Your subscription does not include this module.")

st.markdown("---")

# ── Platform stats ────────────────────────────────────────────────────────────
st.subheader("Platform Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Pose Classes", "21", "JHMDB standard")
c2.metric("Body Joints Tracked", "15", "per frame")
c3.metric("Feature Dimensions", "592+", "hierarchical + gait + risk")
c4.metric("Research Modules", "7", "novel algorithms")

st.markdown("""
### Core Research Modules
| Module | Algorithm | Novel Contribution |
|--------|-----------|-------------------|
| 🏥 Medical | Joint angle evaluation + CPG | Domain-specific pose correction |
| 🦶 Gait Lab | HGD (3-level hierarchy) | Gait analysis without force plates |
| 🦶 Gait Lab | HKRA (kinematic residuals) | Passive compensation detection |
| 🦶 Gait Lab | PBRS (multi-factor risk) | Injury prediction from 2D pose |
| 🧠 Adaptive Care | EWMA fault memory | Cross-session learning |
| 🧠 Adaptive Care | Mann-Kendall trend test | Regression detection |
| 🧠 Adaptive Care | Evidence-based protocol | Auto-generated rehab programmes |

### How it works
1. **MediaPipe** extracts 33 body landmarks from video, image, or live webcam
2. Landmarks mapped to **15 anatomical joints** (JHMDB standard)
3. **Hierarchical features** computed — angles, distances, bilateral symmetry, gait
4. **Domain-specific analysis** — CPG corrections, gait scores, injury risk, trajectories
5. **Longitudinal ACE** — tracks compliance across sessions, detects regression, generates protocols
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Navigate using the pages above** ↑")
st.sidebar.markdown("Built with MediaPipe · scikit-learn · Streamlit")
