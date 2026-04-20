"""Action Recognition — .mat file, video upload, or live webcam snapshot."""
import os
import sys
import tempfile
import warnings
from pathlib import Path

# Add project root to path so 'psrn' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import streamlit as st

st.set_page_config(page_title="Action Recognition", layout="wide")
st.title("🎬 JHMDB Action Recognition")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("Configuration")
model_path  = st.sidebar.text_input("Model path (.pkl)",
    "outputs/exp1/hierpose_jhmdb_split1/model.pkl")
scaler_path = st.sidebar.text_input("Scaler path (.pkl)",
    "outputs/exp1/hierpose_jhmdb_split1/scaler.pkl")

# ── Helpers ────────────────────────────────────────────────────────────────

JHMDB_BONES = [
    (0,2),(0,1),(0,3),(0,4),(3,7),(4,8),(7,11),(8,12),
    (1,5),(1,6),(5,9),(6,10),(9,13),(10,14),
]

def draw_skeleton(joints_2d, title="Pose", img=None):
    """Draw skeleton on matplotlib figure. joints_2d: (15,2) array."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 5))
    if img is not None:
        ax.imshow(img)
    else:
        ax.set_facecolor("#1a1a2e")
        fig.patch.set_facecolor("#1a1a2e")
    xs, ys = joints_2d[:, 0], joints_2d[:, 1]
    for i, j in JHMDB_BONES:
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]], "c-", lw=2, alpha=0.8)
    ax.scatter(xs, ys, c="yellow", s=60, zorder=5)
    ax.set_title(title, color="white" if img is None else "black", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    return fig


def load_predictor():
    if not Path(model_path).exists():
        st.error(f"Model not found: `{model_path}`  \nWait for training to finish or update the path.")
        return None
    from psrn.inference.predictor import HierPosePredictor
    p = HierPosePredictor(model_path, scaler_path)
    p.load()
    return p


def predict_and_display(frames, predictor, skeleton_title="Skeleton"):
    """Run prediction and display results + skeleton."""
    result = predictor.predict_frames(frames)

    col_res, col_skel = st.columns([1, 1])
    with col_res:
        st.success(f"**Predicted Action:** `{result.predicted_class}`")
        st.metric("Confidence", f"{result.confidence:.1%}")
        if result.class_probabilities:
            top3 = sorted(result.class_probabilities.items(), key=lambda x: -x[1])[:3]
            st.markdown("**Top-3 Predictions:**")
            for cls, prob in top3:
                st.progress(float(prob), text=f"{cls}: {prob:.1%}")

    with col_skel:
        # Use the middle frame for visualisation
        mid = len(frames) // 2
        fig = draw_skeleton(frames[mid], title=f"{result.predicted_class} — {result.confidence:.1%}")
        st.pyplot(fig)
        import matplotlib.pyplot as plt
        plt.close(fig)


def joints_from_mediapipe_frame(frame_rgb):
    """Extract 15 JHMDB-compatible joints from a single BGR/RGB frame via MediaPipe."""
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
    except Exception as e:
        st.error(f"MediaPipe unavailable: {e}. Try uploading a .mat file instead.")
        return None
    H, W = frame_rgb.shape[:2]
    # MediaPipe → JHMDB index mapping (approximate)
    MP2JHMDB = {
        0:  2,   # nose → face
        11: 4,   # left shoulder → l_shoulder
        12: 3,   # right shoulder → r_shoulder
        13: 8,   # left elbow → l_elbow
        14: 7,   # right elbow → r_elbow
        15: 12,  # left wrist → l_wrist
        16: 11,  # right wrist → r_wrist
        23: 6,   # left hip → l_hip
        24: 5,   # right hip → r_hip
        25: 10,  # left knee → l_knee
        26: 9,   # right knee → r_knee
        27: 14,  # left ankle → l_ankle
        28: 13,  # right ankle → r_ankle
    }
    joints = np.zeros((15, 2), dtype=np.float32)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.4) as pose:
        res = pose.process(frame_rgb)
        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            for mp_idx, jhmdb_idx in MP2JHMDB.items():
                joints[jhmdb_idx] = [lm[mp_idx].x * W, lm[mp_idx].y * H]
            # Neck = midpoint of shoulders
            joints[0] = (joints[3] + joints[4]) / 2
            # Belly = midpoint of hips
            joints[1] = (joints[5] + joints[6]) / 2
    return joints


# ── Tabs ───────────────────────────────────────────────────────────────────
tab_mat, tab_video, tab_cam = st.tabs(["📄 .mat File", "🎥 Video Upload", "📷 Live Camera"])


# ── Tab 1: .mat file ───────────────────────────────────────────────────────
with tab_mat:
    st.markdown("Upload a JHMDB `.mat` joint file directly.")
    uploaded_mat = st.file_uploader("Upload joint .mat file", type=["mat"], key="mat")
    if uploaded_mat and st.button("Classify (.mat)", key="btn_mat"):
        predictor = load_predictor()
        if predictor:
            with st.spinner("Classifying..."):
                try:
                    from psrn.data.jhmdb_loader import load_mat_joints
                    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as f:
                        f.write(uploaded_mat.read())
                        tmp = f.name
                    frames = load_mat_joints(Path(tmp))
                    os.unlink(tmp)
                    predict_and_display(frames, predictor)
                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── Tab 2: Video upload ────────────────────────────────────────────────────
with tab_video:
    st.markdown("Upload a video file. MediaPipe extracts skeleton joints per frame, then the model classifies the clip.")
    uploaded_vid = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"], key="vid")
    sample_frames = st.slider("Sample every N frames", 1, 10, 3, key="stride")

    if uploaded_vid and st.button("Classify Video", key="btn_vid"):
        try:
            import cv2, mediapipe  # noqa: F401
        except Exception as e:
            st.error(f"MediaPipe/OpenCV unavailable ({e}). Use the .mat file tab instead.")
            st.stop()

        predictor = load_predictor()
        if predictor:
            with st.spinner("Extracting joints from video..."):
                try:
                    import cv2
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                        f.write(uploaded_vid.read())
                        tmp_vid = f.name

                    cap = cv2.VideoCapture(tmp_vid)
                    joint_frames, preview_frames = [], []
                    i = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if i % sample_frames == 0:
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            joints = joints_from_mediapipe_frame(rgb)
                            joint_frames.append(joints)
                            if len(preview_frames) < 6:
                                preview_frames.append((rgb, joints))
                        i += 1
                    cap.release()
                    os.unlink(tmp_vid)

                    if len(joint_frames) < 2:
                        st.error("Could not extract joints. Check the video has a visible person.")
                    else:
                        frames_arr = np.stack(joint_frames)   # (T, 15, 2)
                        st.info(f"Extracted {len(joint_frames)} frames with joints.")

                        # Show preview skeletons
                        if preview_frames:
                            cols = st.columns(min(len(preview_frames), 6))
                            for col, (img, jnts) in zip(cols, preview_frames):
                                with col:
                                    import matplotlib.pyplot as plt
                                    fig, ax = plt.subplots(figsize=(2.5, 3))
                                    ax.imshow(img)
                                    xs, ys = jnts[:,0], jnts[:,1]
                                    for a, b in JHMDB_BONES:
                                        ax.plot([xs[a],xs[b]],[ys[a],ys[b]],"c-",lw=1.5)
                                    ax.scatter(xs, ys, c="yellow", s=20, zorder=5)
                                    ax.axis("off")
                                    plt.tight_layout(pad=0)
                                    col.pyplot(fig)
                                    plt.close(fig)

                        predict_and_display(frames_arr, predictor, "Video Clip")

                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())


# ── Tab 3: Live camera ─────────────────────────────────────────────────────
with tab_cam:
    st.markdown("Take a photo with your camera. MediaPipe extracts the pose and the model classifies the action.")
    st.info(
        "**Tips for best results:**\n"
        "- Stand back so your **full body** is visible\n"
        "- Hold the **mid-point** of the action (e.g. arm raised mid-brush, mid-pour)\n"
        "- Good lighting, contrasting background\n"
        "- Since this is a single frame, similar poses (brush_hair / pour / wave) may overlap — the model was trained on video motion"
    )

    img_file = st.camera_input("Take a photo", key="cam")
    if img_file and st.button("Classify Pose", key="btn_cam"):
        try:
            import mediapipe  # noqa: F401
        except ImportError:
            st.error("MediaPipe not installed. Run: `pip install mediapipe` then restart.")
            st.stop()

        predictor = load_predictor()
        if predictor:
            with st.spinner("Detecting pose and classifying..."):
                try:
                    import cv2
                    from PIL import Image
                    import io

                    img = Image.open(img_file).convert("RGB")
                    frame_rgb = np.array(img)
                    joints = joints_from_mediapipe_frame(frame_rgb)

                    if joints.sum() == 0:
                        st.warning("No person detected. Try better lighting or move further from camera.")
                    else:
                        # Create a synthetic clip: 20 frames with small Gaussian jitter
                        # so velocity/acceleration features are non-zero (realistic motion)
                        rng = np.random.default_rng(42)
                        frames_arr = np.stack([
                            joints + rng.normal(0, 0.8, joints.shape).astype(np.float32)
                            for _ in range(20)
                        ])  # (20, 15, 2)

                        col1, col2 = st.columns([1, 1])
                        with col1:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(4, 5))
                            ax.imshow(frame_rgb)
                            xs, ys = joints[:,0], joints[:,1]
                            for a, b in JHMDB_BONES:
                                ax.plot([xs[a],xs[b]],[ys[a],ys[b]],"c-",lw=2)
                            ax.scatter(xs, ys, c="yellow", s=60, zorder=5)
                            ax.set_title("Detected Skeleton", fontsize=11)
                            ax.axis("off")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)

                        with col2:
                            result = predictor.predict_frames(frames_arr)
                            st.success(f"**Predicted Action:** `{result.predicted_class}`")
                            st.metric("Confidence", f"{result.confidence:.1%}")
                            if result.class_probabilities:
                                top3 = sorted(result.class_probabilities.items(),
                                              key=lambda x: -x[1])[:3]
                                st.markdown("**Top-3 Predictions:**")
                                for cls, prob in top3:
                                    st.progress(float(prob), text=f"{cls}: {prob:.1%}")

                except Exception as e:
                    import traceback
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

# ── Info panel ─────────────────────────────────────────────────────────────
with st.expander("ℹ️ About JHMDB (21 action classes)"):
    from psrn.data.jhmdb_loader import JHMDB_CLASSES
    st.write(", ".join(JHMDB_CLASSES))
    st.markdown("""
    - **15 skeleton joints** per frame (JHMDB format)
    - **~960 video clips** across 3 official splits
    - Model: SVM + hierarchical features (~592 → 200 selected)
    """)
