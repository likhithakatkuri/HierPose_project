"""Workplace Ergonomics — RULA-proxy real-time risk scoring."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import os, tempfile
from typing import List, Optional
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Ergonomics", layout="wide")
st.title("🖥️ Workplace Ergonomics Analyser")
st.markdown("RULA-proxy posture risk · Real-time joint scoring · Injury prevention guidance")

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

# RULA-proxy scoring rules
# Each check: (label, ja, vertex, jc, ideal°, warn_thresh°, risk_thresh°, description)
ERGO_CHECKS = [
    ("Neck flexion",   FACE,       NECK,       R_SHOULDER,  170, 20, 35, "Keep head upright. >20° flexion increases disc pressure."),
    ("Trunk forward",  FACE,       NECK,       BELLY,       170, 20, 35, "Sit tall. Forward lean compresses lumbar spine."),
    ("Shoulder (R)",   NECK,       R_SHOULDER, R_ELBOW,      20, 45, 70, "Keep upper arms close to body. Elevation causes fatigue."),
    ("Shoulder (L)",   NECK,       L_SHOULDER, L_ELBOW,      20, 45, 70, "Keep upper arms close to body."),
    ("Elbow (R)",      R_SHOULDER, R_ELBOW,    R_WRIST,      90, 30, 50, "Keep elbows at ~90°. Deviation causes repetitive strain."),
    ("Elbow (L)",      L_SHOULDER, L_ELBOW,    L_WRIST,      90, 30, 50, "Keep elbows at ~90°."),
    ("Hip angle",      R_SHOULDER, R_HIP,      R_KNEE,       90, 15, 30, "Hips at 90°. Avoid slouching or perching."),
    ("Knee angle",     R_HIP,      R_KNEE,     R_ANKLE,      90, 20, 40, "Knees at 90°, feet flat. Avoid crossing legs."),
]

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

def rula_score(joints):
    """Compute RULA-proxy scores. Returns list of dicts per check."""
    results=[]
    total_score=0
    for label,ja,jb,jc,ideal,warn,risk,tip in ERGO_CHECKS:
        if joints[ja].sum()==0 or joints[jb].sum()==0 or joints[jc].sum()==0: continue
        cur=compute_angle(joints[ja],joints[jb],joints[jc])
        dev=abs(cur-ideal)
        if dev<=warn:   risk_lvl,col,score_contrib,icon="Low","green",1,"🟢"
        elif dev<=risk: risk_lvl,col,score_contrib,icon="Medium","orange",2,"🟡"
        else:           risk_lvl,col,score_contrib,icon="High","red",3,"🔴"
        total_score+=score_contrib
        instr=("Good position." if dev<=warn else
               f"Adjust {label.lower()} by {dev:.0f}° toward {ideal}° — {tip.split('.')[0]}.")
        results.append({"Check":label,"Current":round(cur,1),"Ideal":ideal,
                        "Deviation":round(dev,1),"Risk":risk_lvl,"_color":col,
                        "Score":score_contrib,"Icon":icon,"Instruction":instr,"Tip":tip})
    return results, total_score

def overall_risk(score,n):
    """Map total RULA-proxy score to risk label."""
    avg=score/max(n,1)
    if avg<=1.4: return "🟢 Low Risk","green","Posture is acceptable. Continue monitoring."
    elif avg<=2.0: return "🟡 Medium Risk","orange","Some adjustments recommended. Review flagged joints."
    else: return "🔴 High Risk","red","Immediate action required. Risk of musculoskeletal injury."

def annotate(frame_rgb,joints,evals):
    import cv2; out=frame_rgb.copy(); H,W=out.shape[:2]
    jcol={}
    for e in evals:
        c={"green":(0,220,80),"orange":(0,165,255),"red":(60,60,255)}.get(e["_color"],(180,180,180))
        for lbl,ja,jb,jc_idx,*_ in ERGO_CHECKS:
            if lbl==e["Check"]: jcol[jb]=c
    for a,b in JHMDB_BONES:
        pa,pb=tuple(joints[a].astype(int)),tuple(joints[b].astype(int))
        if pa!=(0,0) and pb!=(0,0): cv2.line(out,pa,pb,jcol.get(a,jcol.get(b,(0,220,255))),3)
    for idx,(x,y) in enumerate(joints.astype(int)):
        if (x,y)!=(0,0):
            cv2.circle(out,(x,y),8,jcol.get(idx,(0,220,255)),-1); cv2.circle(out,(x,y),8,(0,0,0),1)
    for e in evals:
        for lbl,ja,jb,jc_idx,*_ in ERGO_CHECKS:
            if lbl==e["Check"] and joints[jb].sum()>0:
                px,py=joints[jb].astype(int)
                c={"green":(0,220,80),"orange":(0,165,255),"red":(100,100,255)}.get(e["_color"])
                cv2.putText(out,f"{e['Current']:.0f}°",(px+10,py-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,c,2)
    _,total=rula_score(joints)
    n=sum(1 for e in evals)
    risk_lbl,*_=overall_risk(total,n)
    ov=out.copy(); cv2.rectangle(ov,(0,0),(W,55),(0,0,0),-1); cv2.addWeighted(ov,0.55,out,0.45,0,out)
    rc=(0,220,80) if "Low" in risk_lbl else (0,165,255) if "Medium" in risk_lbl else (60,60,255)
    cv2.putText(out,f"RULA RISK: {risk_lbl.split()[1]}",(10,36),cv2.FONT_HERSHEY_DUPLEX,1.0,rc,2)
    bad=[e for e in evals if e["_color"]=="red"]
    if bad: cv2.putText(out,bad[0]["Instruction"][:55],(10,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,255),2)
    return out

def risk_gauge(score,n):
    avg=score/max(n,1); pct=min(avg/3,1)
    fig,ax=plt.subplots(figsize=(4,2.5),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
    theta=np.linspace(0,np.pi,100)
    ax.plot(np.cos(theta),np.sin(theta),color="#333",lw=12,solid_capstyle="round")
    color="#00dc50" if pct<0.4 else "#ffcc00" if pct<0.7 else "#ff4040"
    ax.plot(np.cos(theta[:int(pct*100)]),np.sin(theta[:int(pct*100)]),color=color,lw=12,solid_capstyle="round")
    lbl,*_=overall_risk(score,n)
    ax.text(0,0.3,lbl,ha="center",color=color,fontsize=11,fontweight="bold")
    ax.text(0,0.0,f"RULA Score: {score}/{n*3}",ha="center",color="gray",fontsize=9)
    ax.set_xlim(-1.3,1.3);ax.set_ylim(-0.4,1.3);ax.axis("off")
    plt.tight_layout(pad=0); return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")
st.sidebar.markdown("**RULA Proxy Thresholds:**")
st.sidebar.markdown("- 🟢 Low: deviation ≤ warn threshold\n- 🟡 Medium: ≤ risk threshold\n- 🔴 High: > risk threshold")
for lbl,*_,ideal,warn,risk,tip in ERGO_CHECKS:
    st.sidebar.markdown(f"- **{lbl}**: ideal {ideal}°, warn >{warn}°, risk >{risk}°")

tab_cam,tab_vid,tab_live=st.tabs(["📷 Photo Assessment","🎥 Video Session","📡 Live Monitor"])

with tab_cam:
    st.subheader("📷 Instant Posture Assessment")
    st.info("Sit in your normal working position. Full body must be visible.")
    img=st.camera_input("Capture posture",key="ergo_cam")
    if img and st.button("🔍 Assess Posture",type="primary",key="ergo_cbtn"):
        from PIL import Image as PILImage
        frame_rgb=np.array(PILImage.open(img).convert("RGB"))
        joints=extract_joints(frame_rgb)
        if joints is None: st.warning("No person detected.")
        else:
            evals,total=rula_score(joints)
            ann=annotate(frame_rgb,joints,evals)
            risk_lbl,risk_col,risk_msg=overall_risk(total,len(evals))

            c1,c2=st.columns([3,2])
            with c1:
                st.image(ann,caption=risk_lbl,use_container_width=True)
                fig_g=risk_gauge(total,len(evals)); st.pyplot(fig_g); plt.close(fig_g)
            with c2:
                if risk_col=="green": st.success(f"**{risk_lbl}** — {risk_msg}")
                elif risk_col=="orange": st.warning(f"**{risk_lbl}** — {risk_msg}")
                else: st.error(f"**{risk_lbl}** — {risk_msg}")

                st.dataframe(pd.DataFrame([{"Check":e["Check"],"Current":f"{e['Current']}°",
                    "Ideal":f"{e['Ideal']}°","Dev":f"{e['Deviation']:.0f}°","Risk":e["Icon"]+" "+e["Risk"]}
                    for e in evals]),use_container_width=True,hide_index=True)

                st.markdown("#### Recommendations")
                for e in sorted(evals,key=lambda x:x["Score"],reverse=True):
                    if e["_color"]!="green":
                        st.error(f"**{e['Check']}** — {e['Instruction']}")
                    else:
                        st.success(f"✅ **{e['Check']}** — Good position.")

with tab_vid:
    st.subheader("🎥 Video Session Analysis")
    st.info("Upload a recording of your work session. The AI tracks posture over time.")
    vid=st.file_uploader("Upload video",type=["mp4","avi","mov","mkv"],key="ergo_vid")
    stride=st.slider("Every N frames",1,5,3,key="ergo_str")
    if vid and st.button("▶ Run Analysis",type="primary",key="ergo_vbtn"):
        try: import cv2,mediapipe  # noqa
        except ImportError: st.error("pip install opencv-python mediapipe==0.10.9"); st.stop()
        with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False) as f:
            f.write(vid.read()); tmp=f.name
        cap=cv2.VideoCapture(tmp); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        pb=st.progress(0,"Analysing…")
        joint_seq,frame_seq,eval_seq,score_seq=[],[],[],[]
        i=0
        while True:
            ret,frame=cap.read()
            if not ret: break
            if i%stride==0:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); joints=extract_joints(rgb)
                if joints is not None:
                    evs,total_s=rula_score(joints)
                    joint_seq.append(joints); frame_seq.append(rgb)
                    eval_seq.append(evs); score_seq.append(total_s/max(len(evs),1))
                pb.progress(min(int(i/max(total,1)*100),99))
            i+=1
        cap.release(); os.unlink(tmp); pb.progress(100,"Done!")
        if not joint_seq: st.error("No person detected."); st.stop()

        avg_s=float(np.mean(score_seq)); peak_s=float(np.max(score_seq))
        n_low=sum(1 for s in score_seq if s<1.4); n_med=sum(1 for s in score_seq if 1.4<=s<2)
        n_hi=sum(1 for s in score_seq if s>=2); pct_hi=n_hi/max(len(score_seq),1)*100

        st.markdown("---"); c1,c2,c3,c4=st.columns(4)
        c1.metric("Avg RULA Score",f"{avg_s:.1f}/3.0"); c2.metric("Peak Score",f"{peak_s:.1f}/3.0")
        c3.metric("⚠️ Time at High Risk",f"{pct_hi:.0f}%")
        c4.metric("Overall",("🟢 Low" if avg_s<1.4 else "🟡 Medium" if avg_s<2 else "🔴 High"))

        # Risk timeline
        fig,ax=plt.subplots(figsize=(10,3),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
        colors=["#00dc50" if s<1.4 else "#ffcc00" if s<2 else "#ff4040" for s in score_seq]
        ax.bar(range(len(score_seq)),score_seq,color=colors,width=1)
        ax.axhline(1.4,color="lime",lw=1,ls="--",alpha=0.7,label="Low/Medium boundary")
        ax.axhline(2.0,color="orange",lw=1,ls="--",alpha=0.7,label="Medium/High boundary")
        ax.set_ylim(0,3); ax.set_xlabel("Frame",color="white"); ax.set_ylabel("RULA Score",color="white")
        ax.set_title("Posture Risk Score Over Session",color="white")
        ax.tick_params(colors="white")
        for s in ax.spines.values(): s.set_edgecolor("#333")
        ax.legend(facecolor="#1a1a2e",labelcolor="white",fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        # Per-check risk breakdown
        st.markdown("#### Risk Breakdown by Body Part")
        check_scores={e["Check"]:[] for e in eval_seq[0]}
        for evs in eval_seq:
            for e in evs: check_scores[e["Check"]].append(e["Score"])
        avg_check={k:float(np.mean(v)) for k,v in check_scores.items()}
        fig2,ax2=plt.subplots(figsize=(10,3),facecolor="#0e1117"); ax2.set_facecolor("#0e1117")
        labels=list(avg_check.keys()); vals=list(avg_check.values())
        bar_colors=["#00dc50" if v<1.4 else "#ffcc00" if v<2 else "#ff4040" for v in vals]
        bars=ax2.barh(labels,vals,color=bar_colors,edgecolor="#333")
        ax2.axvline(1.4,color="lime",lw=1,ls="--",alpha=0.7)
        ax2.axvline(2.0,color="orange",lw=1,ls="--",alpha=0.7)
        ax2.set_xlim(0,3); ax2.set_xlabel("Avg RULA Score",color="white")
        ax2.set_title("Average Risk per Body Part",color="white"); ax2.tick_params(colors="white")
        for s in ax2.spines.values(): s.set_edgecolor("#333")
        for bar,v in zip(bars,vals): ax2.text(v+0.05,bar.get_y()+bar.get_height()/2,f"{v:.1f}",va="center",color="white",fontsize=8)
        plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)

        # Key frames
        st.markdown("#### Key Frames")
        idxs=np.linspace(0,len(frame_seq)-1,min(6,len(frame_seq)),dtype=int)
        cols=st.columns(len(idxs))
        for col,fi in zip(cols,idxs):
            ann=annotate(frame_seq[fi],joint_seq[fi],eval_seq[fi])
            rl,*_=overall_risk(eval_seq[fi][0]["Score"] if eval_seq[fi] else 0,1)
            col.image(ann,use_container_width=True,caption=f"F{fi}|Score:{score_seq[fi]:.1f}")

        # Recommendations
        st.markdown("#### 📋 Session Recommendations")
        worst=[k for k,v in sorted(avg_check.items(),key=lambda x:-x[1]) if v>=1.4]
        if not worst: st.success("✅ Overall posture was acceptable during this session.")
        else:
            for ck in worst[:4]:
                tip=next(t for lbl,*_,t in ERGO_CHECKS if lbl==ck)
                sc=avg_check[ck]
                if sc>=2: st.error(f"🔴 **{ck}** (avg {sc:.1f}/3) — {tip}")
                else: st.warning(f"🟡 **{ck}** (avg {sc:.1f}/3) — {tip}")

with tab_live:
    st.subheader("📡 Live Posture Monitor")
    st.info("Real-time RULA-proxy risk scoring via webcam.")
    try:
        from streamlit_webrtc import webrtc_streamer,VideoProcessorBase,RTCConfiguration
        import av
        class ErgoProc(VideoProcessorBase):
            def __init__(self): self._mp=None; self.risk="Unknown"
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
                    evs,total=rula_score(joints); n=len(evs)
                    self.risk,_,_=overall_risk(total,n)
                    out=cv2.cvtColor(annotate(rgb,joints,evs),cv2.COLOR_RGB2BGR)
                else:
                    out=img.copy(); cv2.putText(out,"No person detected",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,80,255),2)
                return av.VideoFrame.from_ndarray(out,format="bgr24")
        ctx=webrtc_streamer(key="ergo-live",video_processor_factory=ErgoProc,
            rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video":True,"audio":False},async_processing=True)
        if ctx.video_processor: st.metric("Current Risk",ctx.video_processor.risk)
    except ImportError: st.error("pip install streamlit-webrtc aiortc")

with st.expander("ℹ️ About RULA Proxy Scoring"):
    st.markdown("""
    **RULA (Rapid Upper Limb Assessment)** is a standard ergonomic assessment method. This app computes a *proxy* score from keypoint joint angles.
    | Score | Risk Level | Action |
    |-------|-----------|--------|
    | 1.0 – 1.4 | 🟢 Low | Acceptable posture |
    | 1.4 – 2.0 | 🟡 Medium | Investigate and change soon |
    | 2.0 – 3.0 | 🔴 High | Implement change immediately |
    > *This tool provides AI-assisted guidance only and does not replace a formal ergonomic assessment by a qualified professional.*
    """)
