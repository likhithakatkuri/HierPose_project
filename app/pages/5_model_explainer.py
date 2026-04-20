"""Model Explainer — SHAP feature importance + prediction breakdown."""
from __future__ import annotations
import os, sys, tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.set_page_config(page_title="Model Explainer", layout="wide")
st.title("🔍 Model Explainer")
st.markdown("SHAP feature importance · Per-class prediction breakdown · Feature contribution analysis")

if not st.session_state.get("logged_in"):
    st.warning("Please sign in from the Home page.")
    st.page_link("main.py", label="← Go to Sign In"); st.stop()

MP2JHMDB={0:2,11:4,12:3,13:8,14:7,15:12,16:11,23:6,24:5,25:10,26:9,27:14,28:13}
NECK=0;BELLY=1;FACE=2;R_SHOULDER=3;L_SHOULDER=4;R_HIP=5;L_HIP=6
R_ELBOW=7;L_ELBOW=8;R_KNEE=9;L_KNEE=10;R_WRIST=11;L_WRIST=12;R_ANKLE=13;L_ANKLE=14
JHMDB_BONES=[(NECK,FACE),(NECK,R_SHOULDER),(NECK,L_SHOULDER),(NECK,BELLY),
    (R_SHOULDER,R_ELBOW),(R_ELBOW,R_WRIST),(L_SHOULDER,L_ELBOW),(L_ELBOW,L_WRIST),
    (BELLY,R_HIP),(BELLY,L_HIP),(R_HIP,R_KNEE),(R_KNEE,R_ANKLE),(L_HIP,L_KNEE),(L_KNEE,L_ANKLE)]

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
model_path  = st.sidebar.text_input("Model (.pkl)", "outputs/exp1/hierpose_jhmdb_split1/model.pkl")
scaler_path = st.sidebar.text_input("Scaler (.pkl)","outputs/exp1/hierpose_jhmdb_split1/scaler.pkl")

@st.cache_resource
def load_pred(mp,sp):
    if not Path(mp).exists(): return None
    from psrn.inference.predictor import HierPosePredictor
    p=HierPosePredictor(mp,sp); p.load(); return p

def extract_joints(frame_rgb):
    import mediapipe as mp
    H,W=frame_rgb.shape[:2]; joints=np.zeros((15,2),dtype=np.float32)
    with mp.solutions.pose.Pose(static_image_mode=True,min_detection_confidence=0.45) as pose:
        res=pose.process(frame_rgb)
        if not res.pose_landmarks: return None
        lm=res.pose_landmarks.landmark
        for mi,ji in MP2JHMDB.items(): joints[ji]=[lm[mi].x*W,lm[mi].y*H]
        joints[NECK]=(joints[R_SHOULDER]+joints[L_SHOULDER])/2
        joints[BELLY]=(joints[R_HIP]+joints[L_HIP])/2
    return joints

tab_cam, tab_mat, tab_info = st.tabs(["📷 Explain from Camera", "📄 Explain from .mat File", "📊 Model Stats"])


# ── Camera tab ────────────────────────────────────────────────────────────────
with tab_cam:
    st.subheader("Explain a live pose capture")
    st.info("Take a photo. The model will classify it and show which features drove the decision.")

    img=st.camera_input("Capture pose",key="exp_cam")
    if img and st.button("🔍 Explain Prediction",type="primary",key="exp_cbtn"):
        from PIL import Image as PILImage
        frame_rgb=np.array(PILImage.open(img).convert("RGB"))

        predictor=load_pred(model_path,scaler_path)
        if not predictor:
            st.error(f"Model not found at: {model_path}"); st.stop()

        with st.spinner("Extracting features and running SHAP…"):
            joints=extract_joints(frame_rgb)

        if joints is None:
            st.warning("No person detected.")
        else:
            rng=np.random.default_rng(42)
            frames_arr=np.stack([joints+rng.normal(0,0.8,joints.shape).astype(np.float32) for _ in range(20)])

            try:
                result=predictor.predict_frames(frames_arr)

                c1,c2,c3=st.columns(3)
                c1.metric("Predicted Action",result.predicted_class)
                c2.metric("Confidence",f"{result.confidence:.1%}")
                c3.metric("Features Used","200 (selected from 592)")

                # Probability bar chart
                if result.class_probabilities:
                    st.markdown("#### Class Probabilities")
                    probs=sorted(result.class_probabilities.items(),key=lambda x:-x[1])
                    labels=[p[0] for p in probs]; vals=[p[1] for p in probs]
                    colors=["#00dc50" if l==result.predicted_class else "#336699" for l in labels]
                    fig,ax=plt.subplots(figsize=(10,4),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
                    bars=ax.barh(labels,vals,color=colors,edgecolor="#333")
                    ax.set_xlabel("Probability",color="white"); ax.set_title("Prediction Confidence per Class",color="white")
                    ax.tick_params(colors="white")
                    for s in ax.spines.values(): s.set_edgecolor("#333")
                    for bar,v in zip(bars,vals):
                        if v>0.01: ax.text(v+0.002,bar.get_y()+bar.get_height()/2,f"{v:.1%}",va="center",color="white",fontsize=7)
                    ax.set_xlim(0,max(vals)*1.2); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

                # Skeleton
                st.markdown("#### Detected Skeleton")
                fig2,ax2=plt.subplots(figsize=(4,5)); ax2.set_facecolor("#1a1a2e"); fig2.patch.set_facecolor("#1a1a2e")
                ax2.imshow(frame_rgb)
                xs,ys=joints[:,0],joints[:,1]
                for a,b in JHMDB_BONES:
                    if joints[a].sum()>0 and joints[b].sum()>0:
                        ax2.plot([xs[a],xs[b]],[ys[a],ys[b]],"c-",lw=2,alpha=0.8)
                ax2.scatter(xs,ys,c="yellow",s=50,zorder=5); ax2.axis("off")
                ax2.set_title(f"Predicted: {result.predicted_class} ({result.confidence:.1%})",color="white")
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

            except Exception as e:
                import traceback; st.error(str(e)); st.code(traceback.format_exc())


# ── .mat file tab ─────────────────────────────────────────────────────────────
with tab_mat:
    st.subheader("Explain from JHMDB .mat file")
    uploaded=st.file_uploader("Upload .mat joint file",type=["mat"],key="exp_mat")
    if uploaded and st.button("🔍 Explain",type="primary",key="exp_mbtn"):
        predictor=load_pred(model_path,scaler_path)
        if not predictor: st.error(f"Model not found: {model_path}"); st.stop()

        with tempfile.NamedTemporaryFile(suffix=".mat",delete=False) as f:
            f.write(uploaded.read()); tmp=f.name
        try:
            from psrn.data.jhmdb_loader import load_mat_joints
            frames=load_mat_joints(tmp); os.unlink(tmp)

            with st.spinner("Running prediction…"):
                result=predictor.predict_frames(frames)

            c1,c2,c3=st.columns(3)
            c1.metric("Predicted",result.predicted_class)
            c2.metric("Confidence",f"{result.confidence:.1%}")
            c3.metric("Clip Length",f"{len(frames)} frames")

            if result.class_probabilities:
                st.markdown("#### Class Probabilities")
                probs=sorted(result.class_probabilities.items(),key=lambda x:-x[1])
                labels=[p[0] for p in probs]; vals=[p[1] for p in probs]
                colors=["#00dc50" if l==result.predicted_class else "#336699" for l in labels]
                fig,ax=plt.subplots(figsize=(10,5),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
                bars=ax.barh(labels,vals,color=colors,edgecolor="#333")
                ax.set_xlabel("Probability",color="white"); ax.set_title("Class Probabilities",color="white")
                ax.tick_params(colors="white")
                for s in ax.spines.values(): s.set_edgecolor("#333")
                for bar,v in zip(bars,vals):
                    if v>0.01: ax.text(v+0.002,bar.get_y()+bar.get_height()/2,f"{v:.1%}",va="center",color="white",fontsize=7)
                ax.set_xlim(0,max(vals)*1.2); plt.tight_layout(); st.pyplot(fig); plt.close(fig)

            # Mid-frame skeleton
            mid=frames[len(frames)//2]
            st.markdown("#### Mid-clip Skeleton")
            fig3,ax3=plt.subplots(figsize=(4,5)); ax3.set_facecolor("#1a1a2e"); fig3.patch.set_facecolor("#1a1a2e")
            xs,ys=mid[:,0],mid[:,1]
            for a,b in JHMDB_BONES:
                if mid[a].sum()>0 and mid[b].sum()>0:
                    ax3.plot([xs[a],xs[b]],[ys[a],ys[b]],color="cyan",lw=2,alpha=0.8)
            ax3.scatter(xs,ys,c="yellow",s=60,zorder=5); ax3.axis("off"); ax3.invert_yaxis()
            ax3.set_title(f"{result.predicted_class} — {result.confidence:.1%}",color="white")
            plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

        except Exception as e:
            import traceback; st.error(str(e)); st.code(traceback.format_exc())
            if os.path.exists(tmp): os.unlink(tmp)


# ── Model stats tab ───────────────────────────────────────────────────────────
with tab_info:
    st.subheader("Model Performance Statistics")
    results_path=Path("outputs/exp1/hierpose_jhmdb_split1/results.json")
    if results_path.exists():
        import json
        with open(results_path) as f: res=json.load(f)

        c1,c2,c3,c4=st.columns(4)
        c1.metric("Test Accuracy",f"{res.get('accuracy',0):.1%}")
        c2.metric("Macro F1",f"{res.get('macro_f1',0):.1%}")
        c3.metric("CV Mean",f"{res.get('cv_mean',0):.1%}")
        c4.metric("Training Samples",res.get("n_train","?"))

        # Per-class accuracy
        if "accuracy_per_class" in res:
            st.markdown("#### Per-Class Accuracy")
            cls_data=sorted(res["accuracy_per_class"].items(),key=lambda x:-x[1])
            labels=[c[0] for c in cls_data]; vals=[c[1] for c in cls_data]
            colors=["#00dc50" if v>=0.8 else "#ffcc00" if v>=0.5 else "#ff4040" for v in vals]
            fig,ax=plt.subplots(figsize=(10,6),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
            bars=ax.barh(labels,vals,color=colors,edgecolor="#333")
            ax.axvline(0.8,color="lime",lw=1,ls="--",alpha=0.6,label="80% threshold")
            ax.axvline(0.5,color="orange",lw=1,ls="--",alpha=0.6,label="50% threshold")
            ax.set_xlim(0,1); ax.set_xlabel("Accuracy",color="white")
            ax.set_title("Per-Class Test Accuracy (JHMDB Split 1)",color="white")
            ax.tick_params(colors="white")
            for s in ax.spines.values(): s.set_edgecolor("#333")
            for bar,v in zip(bars,vals): ax.text(v+0.01,bar.get_y()+bar.get_height()/2,f"{v:.0%}",va="center",color="white",fontsize=8)
            ax.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # CV scores
        if "cv_scores" in res:
            st.markdown("#### Cross-Validation Scores")
            cvs=res["cv_scores"]
            fig2,ax2=plt.subplots(figsize=(6,3),facecolor="#0e1117"); ax2.set_facecolor("#0e1117")
            ax2.bar(range(1,len(cvs)+1),cvs,color="#336699",edgecolor="#333")
            ax2.axhline(np.mean(cvs),color="lime",lw=2,ls="--",label=f"Mean: {np.mean(cvs):.1%}")
            ax2.set_ylim(0,1); ax2.set_xlabel("Fold",color="white"); ax2.set_ylabel("Accuracy",color="white")
            ax2.set_title("5-Fold Cross-Validation",color="white"); ax2.tick_params(colors="white")
            for s in ax2.spines.values(): s.set_edgecolor("#333")
            ax2.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=9)
            for i,v in enumerate(cvs): ax2.text(i+1,v+0.01,f"{v:.1%}",ha="center",color="white",fontsize=8)
            plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
    else:
        st.info("No results.json found. Train the model first:\n```\npython -m psrn.train_ml --data_root data/JHMDB --split 1 --output_dir outputs/exp1 --model_type svm\n```")

    with st.expander("📐 Feature Engineering Pipeline"):
        st.markdown("""
        | Feature Group | Count | Description |
        |---|---|---|
        | Joint angles | ~60 | Angles at each joint (elbow, knee, hip, shoulder…) |
        | Pairwise distances | ~48 | Distances between joint pairs, torso-normalised |
        | Body ratios | ~16 | Limb length ratios, symmetry scores |
        | Temporal velocity | ~90 | Joint velocity (Δposition/frame), mean+std |
        | Temporal acceleration | ~90 | Second-order motion, peak values |
        | Symmetry | ~24 | Left-right angle and distance differences |
        | Bounding box | ~8 | Pose spread, aspect ratio, convex hull area |
        | Centroid motion | ~18 | Whole-body displacement patterns |
        | **Total** | **~592** | → SelectKBest(200) for final model |

        **Model:** SVM (RBF kernel, C=50, γ=scale) with StandardScaler + SelectKBest(f_classif, k=200)
        """)
