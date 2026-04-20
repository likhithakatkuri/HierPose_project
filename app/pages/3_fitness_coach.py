"""AI Fitness Coach — rep counting, form scoring, real-time webcam."""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Dict, List, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Fitness Coach", layout="wide")
st.title("🏋️ AI Fitness Coach")
st.markdown("Rep counting · Form scoring · Joint angle tracking · Live webcam")

if not st.session_state.get("logged_in"):
    st.warning("Please sign in from the Home page.")
    st.page_link("main.py", label="← Go to Sign In"); st.stop()

NECK=0; BELLY=1; FACE=2
R_SHOULDER=3; L_SHOULDER=4; R_HIP=5; L_HIP=6
R_ELBOW=7; L_ELBOW=8; R_KNEE=9; L_KNEE=10
R_WRIST=11; L_WRIST=12; R_ANKLE=13; L_ANKLE=14

JHMDB_BONES = [
    (NECK,FACE),(NECK,R_SHOULDER),(NECK,L_SHOULDER),(NECK,BELLY),
    (R_SHOULDER,R_ELBOW),(R_ELBOW,R_WRIST),
    (L_SHOULDER,L_ELBOW),(L_ELBOW,L_WRIST),
    (BELLY,R_HIP),(BELLY,L_HIP),
    (R_HIP,R_KNEE),(R_KNEE,R_ANKLE),
    (L_HIP,L_KNEE),(L_KNEE,L_ANKLE),
]
MP2JHMDB = {
    0:FACE, 11:L_SHOULDER, 12:R_SHOULDER,
    13:L_ELBOW, 14:R_ELBOW, 15:L_WRIST, 16:R_WRIST,
    23:L_HIP, 24:R_HIP, 25:L_KNEE, 26:R_KNEE, 27:L_ANKLE, 28:R_ANKLE,
}

EXERCISES = {
    "Push-ups":       {"angle_joint":(R_SHOULDER,R_ELBOW,R_WRIST), "up":155,"down":90,  "good_d":95, "good_e":150,"label":"Elbow angle (°)","muscles":"Chest · Triceps · Deltoid"},
    "Squats":         {"angle_joint":(R_HIP,R_KNEE,R_ANKLE),       "up":160,"down":110, "good_d":115,"good_e":155,"label":"Knee angle (°)", "muscles":"Quads · Glutes · Hamstrings"},
    "Jumping Jacks":  {"angle_joint":(R_HIP,R_SHOULDER,R_ELBOW),   "up":110,"down":45,  "good_d":50, "good_e":105,"label":"Shoulder abduction (°)","muscles":"Deltoids · Hip abductors"},
    "Shoulder Press": {"angle_joint":(R_ELBOW,R_SHOULDER,R_HIP),   "up":155,"down":70,  "good_d":80, "good_e":150,"label":"Shoulder angle (°)","muscles":"Deltoids · Triceps"},
    "Bicep Curls":    {"angle_joint":(R_SHOULDER,R_ELBOW,R_WRIST), "up":150,"down":50,  "good_d":55, "good_e":145,"label":"Elbow angle (°)","muscles":"Biceps · Brachialis"},
    "Lunges":         {"angle_joint":(R_HIP,R_KNEE,R_ANKLE),       "up":165,"down":90,  "good_d":95, "good_e":160,"label":"Front knee angle (°)","muscles":"Quads · Glutes · Hamstrings"},
}


def compute_angle(a,b,c):
    ba,bc = a-b, c-b
    cos_v = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_v,-1,1))))

def extract_joints(frame_rgb):
    import mediapipe as mp
    H,W = frame_rgb.shape[:2]
    joints = np.zeros((15,2),dtype=np.float32)
    with mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.45) as pose:
        res = pose.process(frame_rgb)
        if not res.pose_landmarks: return None
        lm = res.pose_landmarks.landmark
        for mi,ji in MP2JHMDB.items():
            joints[ji] = [lm[mi].x*W, lm[mi].y*H]
        joints[NECK]  = (joints[R_SHOULDER]+joints[L_SHOULDER])/2
        joints[BELLY] = (joints[R_HIP]+joints[L_HIP])/2
    return joints

def count_reps(angles, up, down):
    state,count,rfs = "up",0,[]
    for i,a in enumerate(angles):
        if state=="up" and a<down: state="down"
        elif state=="down" and a>up: state="up"; count+=1; rfs.append(i)
    return count,rfs

def score_form(angles, rfs, good_d, good_e, up, down):
    if not rfs: return 0.0,[]
    per_rep=[]; bounds=[0]+rfs+[len(angles)]
    for i in range(len(rfs)):
        seg=angles[bounds[i]:bounds[i+1]+1]
        if not seg: continue
        mn,mx=min(seg),max(seg)
        d_pct=max(0,min(1,(down-mn)/max(1,down-good_d)))
        e_pct=max(0,min(1,(mx-up)/max(1,good_e-up)))
        per_rep.append({"rep":i+1,"min_angle":round(mn,1),"max_angle":round(mx,1),
                        "depth_ok":mn<=good_d,"ext_ok":mx>=good_e,"score":round((d_pct+e_pct)*50,1)})
    overall=np.mean([r["score"] for r in per_rep]) if per_rep else 0.0
    return round(overall,1),per_rep

def annotate_frame(frame_rgb, joints, angle_val, angle_lbl, rep_count, form_score):
    import cv2
    out=frame_rgb.copy(); H,W=out.shape[:2]
    for a,b in JHMDB_BONES:
        pa,pb=tuple(joints[a].astype(int)),tuple(joints[b].astype(int))
        if pa!=(0,0) and pb!=(0,0): cv2.line(out,pa,pb,(0,220,255),2)
    for idx,(x,y) in enumerate(joints.astype(int)):
        if (x,y)!=(0,0): cv2.circle(out,(x,y),7,(0,220,255),-1); cv2.circle(out,(x,y),7,(0,0,0),1)
    ov=out.copy(); cv2.rectangle(ov,(0,0),(W,55),(0,0,0),-1); cv2.addWeighted(ov,0.55,out,0.45,0,out)
    sc=(0,220,80) if form_score>=75 else (0,180,255) if form_score>=50 else (60,60,255)
    cv2.putText(out,f"REPS: {rep_count}",(10,36),cv2.FONT_HERSHEY_DUPLEX,1.1,(255,255,0),2)
    cv2.putText(out,f"{angle_lbl.split()[0]}: {angle_val:.0f}°",(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)
    cv2.putText(out,f"FORM: {form_score:.0f}%",(W-175,36),cv2.FONT_HERSHEY_DUPLEX,1.0,sc,2)
    return out

def angle_plot(angles, rfs, label, up, down):
    fig,ax=plt.subplots(figsize=(10,3),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
    ax.axhspan(down,up,alpha=0.08,color="cyan")
    ax.axhline(down,color="orange",lw=1,ls="--",alpha=0.6)
    ax.axhline(up,  color="lime",  lw=1,ls="--",alpha=0.6)
    ax.plot(angles,color="deepskyblue",lw=2,label=label)
    for rf in rfs:
        ax.axvline(rf,color="yellow",lw=1,alpha=0.7)
        ax.annotate("Rep",xy=(rf,up),color="yellow",fontsize=7,ha="center",va="bottom")
    ax.set_xlabel("Frame",color="white",fontsize=9); ax.set_ylabel("Angle (°)",color="white",fontsize=9)
    ax.set_title(f"{label}  ({len(rfs)} reps)",color="white",fontsize=11)
    ax.tick_params(colors="white")
    for s in ax.spines.values(): s.set_edgecolor("#333")
    ax.legend(fontsize=7,facecolor="#1a1a2e",labelcolor="white"); plt.tight_layout(); return fig

def rep_quality_plot(per_rep):
    if not per_rep: return None
    reps=[r["rep"] for r in per_rep]; scores=[r["score"] for r in per_rep]; depths=[r["min_angle"] for r in per_rep]
    colors=["#00ff80" if s>=75 else "#ffcc00" if s>=50 else "#ff4040" for s in scores]
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,3),facecolor="#0e1117")
    for ax in(ax1,ax2):
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#333")
    bars=ax1.bar(reps,scores,color=colors,edgecolor="#333")
    ax1.set_xlabel("Rep #",color="white"); ax1.set_ylabel("Form (%)",color="white")
    ax1.set_title("Form per Rep",color="white"); ax1.set_ylim(0,100)
    for bar,s in zip(bars,scores): ax1.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1,f"{s:.0f}",ha="center",va="bottom",color="white",fontsize=8)
    ax2.bar(reps,depths,color="deepskyblue",edgecolor="#333")
    ax2.set_xlabel("Rep #",color="white"); ax2.set_ylabel("Min angle (°)",color="white")
    ax2.set_title("Depth per Rep",color="white")
    for rn,d in zip(reps,depths): ax2.text(rn,d+1,f"{d:.0f}°",ha="center",va="bottom",color="white",fontsize=8)
    plt.tight_layout(); return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
exercise  = st.sidebar.selectbox("Exercise", list(EXERCISES.keys()))
model_path = st.sidebar.text_input("Model (.pkl)", "outputs/exp1/hierpose_jhmdb_split1/model.pkl")
scaler_path= st.sidebar.text_input("Scaler (.pkl)","outputs/exp1/hierpose_jhmdb_split1/scaler.pkl")
stride     = st.sidebar.slider("Sample every N frames",1,5,2)
ex = EXERCISES[exercise]
st.sidebar.markdown(f"**Muscles worked:** {ex['muscles']}")

@st.cache_resource
def load_predictor(mp_path, sp_path):
    from pathlib import Path
    if not Path(mp_path).exists(): return None
    from psrn.inference.predictor import HierPosePredictor
    p = HierPosePredictor(mp_path, sp_path); p.load(); return p

tab_vid, tab_cam, tab_live = st.tabs(["🎥 Video Analysis","📷 Camera Snapshot","📡 Live Webcam"])


# ── Video Analysis ────────────────────────────────────────────────────────────
with tab_vid:
    st.subheader(f"Video Analysis: {exercise}")
    vid_file = st.file_uploader("Upload video (mp4/avi/mov)", type=["mp4","avi","mov","mkv"])
    if vid_file and st.button("🚀 Analyse", type="primary"):
        try: import cv2, mediapipe  # noqa
        except ImportError: st.error("pip install opencv-python mediapipe==0.10.9"); st.stop()

        with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False) as f:
            f.write(vid_file.read()); tmp=f.name
        cap=cv2.VideoCapture(tmp); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pb=st.progress(0,"Extracting joints…")
        joint_seq,frame_seq,angles=[],[],[]
        i=0
        while True:
            ret,frame=cap.read()
            if not ret: break
            if i%stride==0:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                joints=extract_joints(rgb)
                if joints is not None:
                    joint_seq.append(joints); frame_seq.append(rgb)
                    ja,jb,jc=ex["angle_joint"]
                    angles.append(compute_angle(joints[ja],joints[jb],joints[jc]))
                pb.progress(min(int(i/max(total,1)*100),99))
            i+=1
        cap.release(); os.unlink(tmp); pb.progress(100,"Done!")
        if len(joint_seq)<3: st.error("Too few frames with detected pose."); st.stop()

        rep_count,rfs=count_reps(angles,ex["up"],ex["down"])
        form_score,per_rep=score_form(angles,rfs,ex["good_d"],ex["good_e"],ex["up"],ex["down"])

        predictor=load_predictor(model_path,scaler_path)
        pred,conf,top3="N/A",0.0,[]
        if predictor:
            try:
                result=predictor.predict_frames(np.stack(joint_seq))
                pred=result.predicted_class; conf=result.confidence
                if result.class_probabilities:
                    top3=sorted(result.class_probabilities.items(),key=lambda x:-x[1])[:5]
            except: pass

        st.markdown("---"); st.subheader("📊 Results")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("🔁 Reps",rep_count)
        c2.metric("💪 Form","{}%".format(int(form_score)),
                  "🟢 Excellent" if form_score>=75 else "🟡 Good" if form_score>=50 else "🔴 Needs Work")
        c3.metric("🎯 Exercise", exercise)   # use the selected exercise, not JHMDB guess
        c4.metric("📈 JHMDB Guess", f"{pred} ({conf:.0%})", help="JHMDB has no 'squat/push-up' class — treat this as approximate")

        fig_a=angle_plot(angles,rfs,ex["label"],ex["up"],ex["down"])
        st.markdown("#### Angle Over Time"); st.pyplot(fig_a); plt.close(fig_a)

        if per_rep:
            st.markdown("#### Per-Rep Quality")
            fig_r=rep_quality_plot(per_rep)
            if fig_r: st.pyplot(fig_r); plt.close(fig_r)
            df=pd.DataFrame(per_rep)
            df["depth_ok"]=df["depth_ok"].map({True:"✅",False:"❌"})
            df["ext_ok"]=df["ext_ok"].map({True:"✅",False:"❌"})
            df.columns=["Rep","Min Angle (°)","Max Angle (°)","Good Depth","Full Extension","Form (%)"]
            st.dataframe(df,use_container_width=True,hide_index=True)

        st.markdown("#### Key Frames")
        idxs=np.linspace(0,len(frame_seq)-1,min(6,len(frame_seq)),dtype=int)
        cols=st.columns(len(idxs))
        for col,fi in zip(cols,idxs):
            ann=annotate_frame(frame_seq[fi],joint_seq[fi],angles[fi],ex["label"],rep_count,form_score)
            col.image(ann,use_container_width=True,caption=f"Frame {fi} | {angles[fi]:.0f}°")

        if top3:
            st.markdown("#### JHMDB Action Classifier (reference only)")
            st.caption("ℹ️ The JHMDB dataset has 21 specific classes (walk, jump, clap…) — it does not include gym exercises like squats or push-ups. The rep counter above is the accurate exercise result.")
            for cls,prob in top3:
                st.progress(float(prob),text=f"{cls}: {prob:.1%}")

        # AI coaching
        st.markdown("#### 🤖 Coaching Feedback")
        good_r=sum(1 for r in per_rep if r["depth_ok"] and r["ext_ok"])
        d_fail=sum(1 for r in per_rep if not r["depth_ok"])
        e_fail=sum(1 for r in per_rep if not r["ext_ok"])
        lines=[f"You completed **{rep_count} reps** of {exercise} — form score **{form_score:.0f}%**."]
        if rep_count==0: lines.append("⚠️ No reps detected. Try a fuller range of motion.")
        else:
            if good_r==rep_count: lines.append("🏆 **Perfect!** All reps had good depth and full extension.")
            else:
                if d_fail: lines.append(f"📉 **{d_fail}/{rep_count} reps lacked depth.** Go lower on each rep.")
                if e_fail: lines.append(f"📈 **{e_fail}/{rep_count} reps lacked extension.** Fully straighten at the top.")
            lines.append("🌟 Excellent!" if form_score>=85 else "👍 Good — work on flagged reps." if form_score>=60 else "💪 Keep practising — prioritise range of motion.")
        st.info("\n\n".join(lines))


# ── Camera snapshot ────────────────────────────────────────────────────────────
with tab_cam:
    st.subheader(f"Snapshot: {exercise}")
    st.info("Hold the mid-point of the exercise. Full body must be visible.")
    img_file=st.camera_input("Take photo",key="fit_cam")
    if img_file and st.button("🔍 Analyse Pose",type="primary",key="btn_fit_cam"):
        from PIL import Image as PILImage
        frame_rgb=np.array(PILImage.open(img_file).convert("RGB"))
        joints=extract_joints(frame_rgb)
        if joints is None:
            st.warning("No person detected. Try better lighting or move further from camera.")
        else:
            ja,jb,jc=ex["angle_joint"]
            angle_val=compute_angle(joints[ja],joints[jb],joints[jc])
            ann=annotate_frame(frame_rgb,joints,angle_val,ex["label"],0,0)
            col1,col2=st.columns([3,2])
            with col1:
                st.image(ann,caption=f"{ex['label']}: {angle_val:.1f}°",use_container_width=True)
            with col2:
                st.metric(ex["label"],f"{angle_val:.1f}°")
                if angle_val<=ex["down"]: st.success(f"✅ Good depth ({angle_val:.0f}° ≤ {ex['down']}°)")
                elif angle_val>=ex["up"]: st.info(f"🔝 Full extension ({angle_val:.0f}°)")
                else: st.warning(f"↕️ Mid-movement ({angle_val:.0f}°)")
                predictor=load_predictor(model_path,scaler_path)
                if predictor:
                    rng=np.random.default_rng(42)
                    frames_arr=np.stack([joints+rng.normal(0,0.8,joints.shape).astype(np.float32) for _ in range(20)])
                    try:
                        result=predictor.predict_frames(frames_arr)
                        st.success(f"**Action:** `{result.predicted_class}`")
                        st.metric("Confidence",f"{result.confidence:.1%}")
                        if result.class_probabilities:
                            for cls,prob in sorted(result.class_probabilities.items(),key=lambda x:-x[1])[:3]:
                                st.progress(float(prob),text=f"{cls}: {prob:.1%}")
                    except Exception as e: st.warning(str(e))


# ── Live webcam ────────────────────────────────────────────────────────────────
with tab_live:
    st.subheader(f"Live: {exercise}")
    st.info("Real-time rep counting and form scoring via your webcam.")
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
        import av
        RTC_CFG=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
        _ex=ex

        class FitProc(VideoProcessorBase):
            def __init__(self): self._mp=None; self.reps=0; self.state="up"; self.angles=[]
            def _pose(self):
                if self._mp is None:
                    import mediapipe as mp
                    self._mp=mp.solutions.pose.Pose(static_image_mode=False,min_detection_confidence=0.45,min_tracking_confidence=0.45)
                return self._mp
            def recv(self,frame):
                import cv2
                img=frame.to_ndarray(format="bgr24"); rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                H,W=rgb.shape[:2]; joints=np.zeros((15,2),dtype=np.float32)
                res=self._pose().process(rgb)
                if res.pose_landmarks:
                    lm=res.pose_landmarks.landmark
                    for mi,ji in MP2JHMDB.items(): joints[ji]=[lm[mi].x*W,lm[mi].y*H]
                    joints[NECK]=(joints[R_SHOULDER]+joints[L_SHOULDER])/2
                    joints[BELLY]=(joints[R_HIP]+joints[L_HIP])/2
                    ja,jb,jc=_ex["angle_joint"]
                    ang=compute_angle(joints[ja],joints[jb],joints[jc])
                    self.angles.append(ang)
                    if self.state=="up" and ang<_ex["down"]: self.state="down"
                    elif self.state=="down" and ang>_ex["up"]: self.state="up"; self.reps+=1
                    recent=self.angles[-20:] if len(self.angles)>=20 else self.angles
                    _,pr=score_form(recent,[],_ex["good_d"],_ex["good_e"],_ex["up"],_ex["down"])
                    fs=np.mean([r["score"] for r in pr]) if pr else 0.0
                    out_rgb=annotate_frame(rgb,joints,ang,_ex["label"].split()[0],self.reps,fs)
                    out=cv2.cvtColor(out_rgb,cv2.COLOR_RGB2BGR)
                else:
                    out=img.copy()
                    cv2.putText(out,"No person detected",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,80,255),2)
                return av.VideoFrame.from_ndarray(out,format="bgr24")

        ctx=webrtc_streamer(key=f"fit-{exercise}",video_processor_factory=FitProc,
                            rtc_configuration=RTC_CFG,
                            media_stream_constraints={"video":True,"audio":False},
                            async_processing=True)
        if ctx.video_processor: st.metric("Live Rep Count",ctx.video_processor.reps)
    except ImportError:
        st.error("pip install streamlit-webrtc aiortc")
