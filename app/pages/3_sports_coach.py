"""Sports & Performance Coach — video/webcam form analysis per sport."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os, tempfile
from typing import Dict, List, Optional
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Sports Coach", layout="wide")
st.title("⚽ Sports & Performance Coach")
st.markdown("Biomechanical form analysis · Joint angle tracking · Rep counting · Real-time feedback")

if not st.session_state.get("logged_in"):
    st.warning("Please sign in from the Home page.")
    st.page_link("main.py", label="← Go to Sign In"); st.stop()

NECK=0;BELLY=1;FACE=2;R_SHOULDER=3;L_SHOULDER=4;R_HIP=5;L_HIP=6
R_ELBOW=7;L_ELBOW=8;R_KNEE=9;L_KNEE=10;R_WRIST=11;L_WRIST=12;R_ANKLE=13;L_ANKLE=14

JHMDB_BONES=[(NECK,FACE),(NECK,R_SHOULDER),(NECK,L_SHOULDER),(NECK,BELLY),
    (R_SHOULDER,R_ELBOW),(R_ELBOW,R_WRIST),(L_SHOULDER,L_ELBOW),(L_ELBOW,L_WRIST),
    (BELLY,R_HIP),(BELLY,L_HIP),(R_HIP,R_KNEE),(R_KNEE,R_ANKLE),(L_HIP,L_KNEE),(L_KNEE,L_ANKLE)]
MP2JHMDB={0:FACE,11:L_SHOULDER,12:R_SHOULDER,13:L_ELBOW,14:R_ELBOW,15:L_WRIST,16:R_WRIST,
    23:L_HIP,24:R_HIP,25:L_KNEE,26:R_KNEE,27:L_ANKLE,28:R_ANKLE}

# metrics: (label, ja, vertex, jc, target°, tolerance°, clinical_desc)
SPORTS: Dict[str,Dict] = {
    "Squat":{"icon":"🏋️","category":"Strength","tip":"Knees over toes, parallel depth, chest up.",
        "metrics":[("Knee (R)",R_HIP,R_KNEE,R_ANKLE,90,10,"90° at bottom"),
                   ("Knee (L)",L_HIP,L_KNEE,L_ANKLE,90,10,"90° at bottom"),
                   ("Hip hinge",R_SHOULDER,R_HIP,R_KNEE,95,12,"Crease below knee"),
                   ("Spine",FACE,NECK,BELLY,160,12,"Chest up")],
        "rep_joint":(R_HIP,R_KNEE,R_ANKLE),"rep_up":160,"rep_down":110},
    "Deadlift":{"icon":"🏋️","category":"Strength","tip":"Hinge hips, flat back, bar over mid-foot.",
        "metrics":[("Hip hinge",NECK,R_HIP,R_KNEE,90,15,"45-90° forward lean"),
                   ("Knee bend",R_HIP,R_KNEE,R_ANKLE,130,15,"Slight bend at lockout"),
                   ("Spine",FACE,NECK,BELLY,155,12,"Neutral — no rounding")],
        "rep_joint":(NECK,R_HIP,R_KNEE),"rep_up":170,"rep_down":100},
    "Overhead Press":{"icon":"💪","category":"Strength","tip":"Full lockout, vertical bar path, tight core.",
        "metrics":[("Elbow (R)",R_SHOULDER,R_ELBOW,R_WRIST,170,8,"Full lockout at top"),
                   ("Shoulder",NECK,R_SHOULDER,R_ELBOW,160,10,"Arms fully overhead"),
                   ("Spine",FACE,NECK,BELLY,172,10,"No excessive arch")],
        "rep_joint":(R_SHOULDER,R_ELBOW,R_WRIST),"rep_up":160,"rep_down":70},
    "Golf Swing":{"icon":"⛳","category":"Sport","tip":"Hip rotation drives swing, consistent shoulder plane.",
        "metrics":[("Lead arm",NECK,L_SHOULDER,L_ELBOW,160,12,"Straight at impact"),
                   ("Hip rotation",R_SHOULDER,R_HIP,L_HIP,120,15,"Hips open at impact"),
                   ("Spine",FACE,NECK,BELLY,145,15,"Maintain throughout")],
        "rep_joint":(NECK,L_SHOULDER,L_ELBOW),"rep_up":150,"rep_down":80},
    "Sprint":{"icon":"🏃","category":"Sport","tip":"High knee drive, forward lean, powerful arm swing.",
        "metrics":[("Knee drive",R_HIP,R_KNEE,R_ANKLE,60,20,"High knee in drive"),
                   ("Hip ext.",R_SHOULDER,R_HIP,R_KNEE,175,12,"Full push-off"),
                   ("Lean",FACE,NECK,BELLY,160,10,"5-10° forward lean")],
        "rep_joint":(R_HIP,R_KNEE,R_ANKLE),"rep_up":150,"rep_down":70},
    "Yoga – Warrior I":{"icon":"🧘","category":"Flexibility","tip":"Front knee 90°, back leg straight, arms overhead.",
        "metrics":[("Front knee",R_HIP,R_KNEE,R_ANKLE,90,10,"90° front knee"),
                   ("Back leg",L_HIP,L_KNEE,L_ANKLE,170,8,"Straight back leg"),
                   ("Arms",NECK,R_SHOULDER,R_ELBOW,170,10,"Fully raised"),
                   ("Torso",FACE,NECK,BELLY,170,10,"Upright torso")],
        "rep_joint":(R_HIP,R_KNEE,R_ANKLE),"rep_up":155,"rep_down":95},
    "Yoga – Downward Dog":{"icon":"🧘","category":"Flexibility","tip":"Hips high, heels toward floor, long spine.",
        "metrics":[("Hip height",R_SHOULDER,R_HIP,R_KNEE,120,12,"Hips pushed high"),
                   ("Arm straight",R_SHOULDER,R_ELBOW,R_WRIST,170,8,"Arms extended"),
                   ("Leg straight",R_HIP,R_KNEE,R_ANKLE,165,10,"Legs as straight as possible")],
        "rep_joint":(R_SHOULDER,R_HIP,R_KNEE),"rep_up":115,"rep_down":80},
}
CATEGORIES=sorted(set(v["category"] for v in SPORTS.values()))

def compute_angle(a,b,c):
    ba,bc=a-b,c-b
    return float(np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9),-1,1))))

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

def evaluate(joints,sport):
    out=[]
    for label,ja,jb,jc,tgt,tol,desc in sport["metrics"]:
        if joints[ja].sum()==0 or joints[jb].sum()==0 or joints[jc].sum()==0: continue
        cur=compute_angle(joints[ja],joints[jb],joints[jc]); dev=cur-tgt; absd=abs(dev)
        if absd<=tol: st,col="✅ Good","green"
        elif absd<=tol*2: st,col="🟡 Close","orange"
        else: st,col="🔴 Adjust","red"
        instr=("Hold position." if absd<=tol else
               f"Increase {label.lower()} by {absd:.1f}° (need {tgt}°, have {cur:.0f}°)." if dev<0 else
               f"Decrease {label.lower()} by {absd:.1f}° (need {tgt}°, have {cur:.0f}°).")
        out.append({"Joint":label,"Current":round(cur,1),"Target":tgt,"Δ":round(dev,1),
                    "Tolerance":f"±{tol}°","Status":st,"_color":col,"Instruction":instr})
    return out

def fscore(evals): return round(sum(1 for e in evals if e["_color"]=="green")/max(len(evals),1)*100,1)

def count_reps(angles,up,down):
    state,count,rfs="up",0,[]
    for i,a in enumerate(angles):
        if state=="up" and a<down: state="down"
        elif state=="down" and a>up: state="up"; count+=1; rfs.append(i)
    return count,rfs

def annotate(rgb,joints,evals,sport,reps,score):
    import cv2; out=rgb.copy(); H,W=out.shape[:2]
    jcol={}
    for e in evals:
        c={"green":(0,220,80),"orange":(0,165,255),"red":(60,60,255)}.get(e["_color"],(180,180,180))
        for lbl,ja,jb,jc,*_ in sport["metrics"]:
            if lbl==e["Joint"]: jcol[jb]=c
    for a,b in JHMDB_BONES:
        pa,pb=tuple(joints[a].astype(int)),tuple(joints[b].astype(int))
        if pa!=(0,0) and pb!=(0,0): cv2.line(out,pa,pb,jcol.get(a,jcol.get(b,(0,220,255))),3)
    for idx,(x,y) in enumerate(joints.astype(int)):
        if (x,y)!=(0,0):
            cv2.circle(out,(x,y),8,jcol.get(idx,(0,220,255)),-1); cv2.circle(out,(x,y),8,(0,0,0),1)
    for e in evals:
        for lbl,ja,jb,jc,*_ in sport["metrics"]:
            if lbl==e["Joint"] and joints[jb].sum()>0:
                px,py=joints[jb].astype(int)
                c={"green":(0,220,80),"orange":(0,165,255),"red":(100,100,255)}.get(e["_color"])
                cv2.putText(out,f"{e['Current']:.0f}°",(px+10,py-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,2)
    ov=out.copy(); cv2.rectangle(ov,(0,0),(W,55),(0,0,0),-1); cv2.addWeighted(ov,0.55,out,0.45,0,out)
    sc=(0,220,80) if score>=75 else (0,180,255) if score>=50 else (60,60,255)
    cv2.putText(out,f"REPS: {reps}",(10,36),cv2.FONT_HERSHEY_DUPLEX,1.1,(255,255,0),2)
    cv2.putText(out,f"FORM: {score:.0f}%",(W-175,36),cv2.FONT_HERSHEY_DUPLEX,1.0,sc,2)
    bad=[e for e in evals if e["_color"]=="red"]
    if bad: cv2.putText(out,bad[0]["Instruction"][:55],(10,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,255),2)
    return out

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Sport / Exercise")
cat=st.sidebar.selectbox("Category",CATEGORIES)
sport_names=[k for k,v in SPORTS.items() if v["category"]==cat]
sport_name=st.sidebar.selectbox("Activity",sport_names)
sport=SPORTS[sport_name]
st.sidebar.markdown("---")
st.sidebar.info(f"**{sport['icon']} {sport_name}**\n\n{sport['tip']}")
for lbl,*_,t,tol,desc in sport["metrics"]:
    st.sidebar.markdown(f"- **{lbl}**: {t}° ±{tol}° — *{desc}*")
model_path=st.sidebar.text_input("Model (.pkl)","outputs/exp1/hierpose_jhmdb_split1/model.pkl",key="sp_m")
stride=st.sidebar.slider("Video stride",1,5,2,key="sp_str")

tab_vid,tab_cam,tab_live=st.tabs(["🎥 Video Analysis","📷 Photo Snapshot","📡 Live Webcam"])

with tab_vid:
    st.subheader(f"{sport['icon']} {sport_name} — Video Analysis")
    st.info(f"💡 {sport['tip']}")
    vid=st.file_uploader("Upload video",type=["mp4","avi","mov","mkv"],key="sp_vid")
    if vid and st.button("🚀 Analyse",type="primary",key="sp_vbtn"):
        try: import cv2,mediapipe  # noqa
        except ImportError: st.error("pip install opencv-python mediapipe==0.10.9"); st.stop()
        with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False) as f:
            f.write(vid.read()); tmp=f.name
        cap=cv2.VideoCapture(tmp); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pb=st.progress(0,"Analysing…")
        joint_seq,frame_seq,eval_seq,comp_seq,angles=[],[],[],[],[]
        i=0
        while True:
            ret,frame=cap.read()
            if not ret: break
            if i%stride==0:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); joints=extract_joints(rgb)
                if joints is not None:
                    evs=evaluate(joints,sport); comp=fscore(evs)
                    ja,jb,jc=sport["rep_joint"]; ang=compute_angle(joints[ja],joints[jb],joints[jc])
                    joint_seq.append(joints); frame_seq.append(rgb)
                    eval_seq.append(evs); comp_seq.append(comp); angles.append(ang)
                pb.progress(min(int(i/max(total,1)*100),99))
            i+=1
        cap.release(); os.unlink(tmp); pb.progress(100,"Done!")
        if not joint_seq: st.error("No person detected."); st.stop()
        reps,rfs=count_reps(angles,sport["rep_up"],sport["rep_down"])
        avg_form=float(np.mean(comp_seq))

        st.markdown("---"); c1,c2,c3=st.columns(3)
        c1.metric("🔁 Reps",reps); c2.metric("💪 Avg Form",f"{avg_form:.0f}%"); c3.metric("📐 Frames",len(joint_seq))

        # Angle plot
        fig,ax=plt.subplots(figsize=(10,3),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
        ax.axhspan(sport["rep_down"],sport["rep_up"],alpha=0.08,color="cyan")
        ax.axhline(sport["rep_down"],color="orange",lw=1,ls="--",alpha=0.6)
        ax.axhline(sport["rep_up"],color="lime",lw=1,ls="--",alpha=0.6)
        ax.plot(angles,color="deepskyblue",lw=2)
        for rf in rfs: ax.axvline(rf,color="yellow",lw=1,alpha=0.7)
        ax.set_xlabel("Frame",color="white"); ax.set_ylabel("Angle (°)",color="white")
        ax.set_title(f"{sport['metrics'][0][0]}  •  {reps} reps",color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#333")
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Compliance
        fig2,ax2=plt.subplots(figsize=(10,2.5),facecolor="#0e1117"); ax2.set_facecolor("#0e1117")
        ax2.fill_between(range(len(comp_seq)),comp_seq,alpha=0.3,color="deepskyblue")
        ax2.plot(comp_seq,color="deepskyblue",lw=2)
        ax2.axhline(75,color="lime",lw=1,ls="--",alpha=0.6); ax2.axhline(50,color="orange",lw=1,ls="--",alpha=0.6)
        ax2.set_ylim(0,100); ax2.set_xlabel("Frame",color="white"); ax2.set_ylabel("Form %",color="white")
        ax2.set_title("Form Compliance Over Time",color="white"); ax2.tick_params(colors="white")
        for s in ax2.spines.values(): s.set_edgecolor("#333")
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

        # Key frames
        st.markdown("#### Key Frames")
        idxs=np.linspace(0,len(frame_seq)-1,min(6,len(frame_seq)),dtype=int)
        cols=st.columns(len(idxs))
        for col,fi in zip(cols,idxs):
            ann=annotate(frame_seq[fi],joint_seq[fi],eval_seq[fi],sport,reps,comp_seq[fi])
            col.image(ann,use_container_width=True,caption=f"F{fi}|{comp_seq[fi]:.0f}%")

        # Report
        st.markdown("#### 🏅 Coaching Report")
        traces={e["Joint"]:[] for e in eval_seq[0]}
        for evs in eval_seq:
            for e in evs: traces[e["Joint"]].append(e["Current"])
        for jname,trace in traces.items():
            avg_a=float(np.mean(trace))
            tgt=next(t for lbl,_,_,_,t,tol,_ in sport["metrics"] if lbl==jname)
            tol=next(tl for lbl,_,_,_,t,tl,_ in sport["metrics"] if lbl==jname)
            dev=avg_a-tgt; amt=abs(round(dev,1))
            if abs(dev)>tol:
                if dev<0: st.error(f"🔴 **{jname}**: avg {avg_a:.0f}° — need {tgt}°. **Increase by {amt}°.**")
                else: st.error(f"🔴 **{jname}**: avg {avg_a:.0f}° — need {tgt}°. **Decrease by {amt}°.**")
            else: st.success(f"✅ **{jname}**: avg {avg_a:.0f}° — within target range.")

with tab_cam:
    st.subheader(f"{sport['icon']} {sport_name} — Snapshot")
    st.info(f"Hold the key position. Full body visible. 💡 {sport['tip']}")
    img=st.camera_input("Capture",key="sp_cam")
    if img and st.button("🔍 Analyse",type="primary",key="sp_cbtn"):
        from PIL import Image as PILImage
        frame_rgb=np.array(PILImage.open(img).convert("RGB"))
        joints=extract_joints(frame_rgb)
        if joints is None: st.warning("No person detected.")
        else:
            evs=evaluate(joints,sport); score=fscore(evs)
            ann=annotate(frame_rgb,joints,evs,sport,0,score)
            c1,c2=st.columns([3,2])
            with c1: st.image(ann,caption=f"Form: {score:.0f}%",use_container_width=True)
            with c2:
                st.metric("Form Score",f"{score:.0f}%","🟢" if score>=75 else "🟡" if score>=50 else "🔴")
                st.dataframe(pd.DataFrame([{"Joint":e["Joint"],"Current":f"{e['Current']}°",
                    "Target":f"{e['Target']}°","Δ":f"{e['Δ']:+.1f}°","Status":e["Status"]}
                    for e in evs]),use_container_width=True,hide_index=True)
                bad=[e for e in evs if e["_color"]!="green"]
                if not bad: st.success("✅ All positions correct!")
                else:
                    for e in bad: st.error(f"**{e['Joint']}** — {e['Instruction']}")

with tab_live:
    st.subheader(f"{sport['icon']} {sport_name} — Live")
    try:
        from streamlit_webrtc import webrtc_streamer,VideoProcessorBase,RTCConfiguration
        import av
        _sp=sport
        class SProc(VideoProcessorBase):
            def __init__(self): self._mp=None;self.reps=0;self.state="up"
            def _pose(self):
                if self._mp is None:
                    import mediapipe as mp
                    self._mp=mp.solutions.pose.Pose(static_image_mode=False,
                        min_detection_confidence=0.45,min_tracking_confidence=0.45)
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
                    evs=evaluate(joints,_sp); sc=fscore(evs)
                    ja,jb,jc=_sp["rep_joint"]; ang=compute_angle(joints[ja],joints[jb],joints[jc])
                    if self.state=="up" and ang<_sp["rep_down"]: self.state="down"
                    elif self.state=="down" and ang>_sp["rep_up"]: self.state="up"; self.reps+=1
                    out=cv2.cvtColor(annotate(rgb,joints,evs,_sp,self.reps,sc),cv2.COLOR_RGB2BGR)
                else:
                    out=img.copy(); cv2.putText(out,"No person detected",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,80,255),2)
                return av.VideoFrame.from_ndarray(out,format="bgr24")
        ctx=webrtc_streamer(key=f"sp-{sport_name}",video_processor_factory=SProc,
            rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video":True,"audio":False},async_processing=True)
        if ctx.video_processor: st.metric("Live Reps",ctx.video_processor.reps)
    except ImportError: st.error("pip install streamlit-webrtc aiortc")
