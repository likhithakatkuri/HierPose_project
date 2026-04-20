"""Medical & Rehabilitation Assistant — full feature version.

New features:
  1. Patient profile & SQLite history database
  2. PDF session report (download button)
  3. Voice correction instructions (gTTS → st.audio)
  4. Bilateral symmetry analysis with L/R comparison chart
  5. Range-of-Motion progress tracker across sessions
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Dict, List, Optional
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# Add app/ to path so utils imports work
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

st.set_page_config(page_title="Medical Assistant", layout="wide")
st.title("🏥 Medical & Rehabilitation Assistant")

if not st.session_state.get("logged_in"):
    st.warning("Please sign in from the Home page.")
    st.page_link("main.py", label="← Go to Sign In"); st.stop()

user = st.session_state.user

# ── DB init ────────────────────────────────────────────────────────────────────
from utils.database import (
    init_db, add_patient, get_patients, get_patient, save_session,
    get_sessions, save_rom, get_rom_history, save_symmetry,
    get_symmetry_history, dashboard_stats
)
init_db()

# ── Joint indices ──────────────────────────────────────────────────────────────
NECK=0;BELLY=1;FACE=2;R_SHOULDER=3;L_SHOULDER=4;R_HIP=5;L_HIP=6
R_ELBOW=7;L_ELBOW=8;R_KNEE=9;L_KNEE=10;R_WRIST=11;L_WRIST=12;R_ANKLE=13;L_ANKLE=14

JHMDB_BONES=[(NECK,FACE),(NECK,R_SHOULDER),(NECK,L_SHOULDER),(NECK,BELLY),
    (R_SHOULDER,R_ELBOW),(R_ELBOW,R_WRIST),(L_SHOULDER,L_ELBOW),(L_ELBOW,L_WRIST),
    (BELLY,R_HIP),(BELLY,L_HIP),(R_HIP,R_KNEE),(R_KNEE,R_ANKLE),(L_HIP,L_KNEE),(L_KNEE,L_ANKLE)]
MP2JHMDB={0:FACE,11:L_SHOULDER,12:R_SHOULDER,13:L_ELBOW,14:R_ELBOW,15:L_WRIST,16:R_WRIST,
    23:L_HIP,24:R_HIP,25:L_KNEE,26:R_KNEE,27:L_ANKLE,28:R_ANKLE}

# ── Procedure library ──────────────────────────────────────────────────────────
PROCEDURES: Dict[str,Dict] = {
    "Knee Flexion – 30°":  {"category":"Orthopaedic","icon":"🦵","instruction":"Bend your knee to approximately 30°. Keep your back straight.",
        "description":"Post-operative knee ROM at 30° flexion.",
        "angles":[("Knee (R)",R_HIP,R_KNEE,R_ANKLE,30,8),("Knee (L)",L_HIP,L_KNEE,L_ANKLE,30,8),("Hip (R)",R_SHOULDER,R_HIP,R_KNEE,170,10),("Spine",FACE,NECK,BELLY,170,12)]},
    "Knee Flexion – 90°":  {"category":"Orthopaedic","icon":"🦵","instruction":"Bend your knee to 90°. Thigh should be horizontal.",
        "description":"Post-operative knee ROM at 90° (post-TKR).",
        "angles":[("Knee (R)",R_HIP,R_KNEE,R_ANKLE,90,8),("Knee (L)",L_HIP,L_KNEE,L_ANKLE,90,8),("Hip (R)",R_SHOULDER,R_HIP,R_KNEE,90,10),("Spine",FACE,NECK,BELLY,170,12)]},
    "Full Knee Extension":  {"category":"Orthopaedic","icon":"🦵","instruction":"Straighten your leg fully. Keep a small soft bend — do not hyperextend.",
        "description":"Full extension check — quad activation, hyperextension risk.",
        "angles":[("Knee (R)",R_HIP,R_KNEE,R_ANKLE,170,8),("Knee (L)",L_HIP,L_KNEE,L_ANKLE,170,8),("Hip (R)",R_SHOULDER,R_HIP,R_KNEE,175,8)]},
    "Straight Leg Raise":  {"category":"Orthopaedic","icon":"🦵","instruction":"Raise your straight leg to 45°. Keep the other leg flat.",
        "description":"SLR test — hamstring/neural tension, quad assessment.",
        "angles":[("Hip (R)",NECK,R_HIP,R_KNEE,135,10),("Knee (R)",R_HIP,R_KNEE,R_ANKLE,175,8),("Spine",FACE,NECK,BELLY,170,12)]},
    "Shoulder Abduction – 90°": {"category":"Orthopaedic","icon":"💪","instruction":"Raise arm sideways to shoulder height. Keep elbow straight.",
        "description":"Shoulder ROM 90° — post-RCR/impingement.",
        "angles":[("Shoulder (R)",NECK,R_SHOULDER,R_ELBOW,90,8),("Shoulder (L)",NECK,L_SHOULDER,L_ELBOW,90,8),("Elbow (R)",R_SHOULDER,R_ELBOW,R_WRIST,175,10)]},
    "Shoulder Abduction – 180°":{"category":"Orthopaedic","icon":"💪","instruction":"Raise arm fully overhead. Palm facing inward.",
        "description":"Full shoulder abduction — frozen shoulder/capsulitis.",
        "angles":[("Shoulder (R)",NECK,R_SHOULDER,R_ELBOW,175,8),("Shoulder (L)",NECK,L_SHOULDER,L_ELBOW,175,8),("Elbow (R)",R_SHOULDER,R_ELBOW,R_WRIST,175,10),("Spine",FACE,NECK,BELLY,170,10)]},
    "PA Chest X-ray":      {"category":"Radiology","icon":"🫁","instruction":"Stand against the plate. Shoulders forward, chin up, arms rotated inward.",
        "description":"Standard PA chest radiograph positioning.",
        "angles":[("Spine",FACE,NECK,BELLY,175,8),("Hip align",NECK,BELLY,R_HIP,175,10),("Shoulder (R)",NECK,R_SHOULDER,R_ELBOW,45,12),("Shoulder (L)",NECK,L_SHOULDER,L_ELBOW,45,12)]},
    "Lateral Spine X-ray": {"category":"Radiology","icon":"🫁","instruction":"Stand sideways. Arms raised overhead, feet together.",
        "description":"Lateral lumbar/thoracic spine positioning.",
        "angles":[("Spine",FACE,NECK,BELLY,175,8),("Shoulder (R)",NECK,R_SHOULDER,R_ELBOW,170,12),("Hip (R)",R_SHOULDER,R_HIP,R_KNEE,175,10)]},
    "Neutral Standing":    {"category":"Physiotherapy","icon":"🧍","instruction":"Stand tall. Ears over shoulders, shoulders over hips.",
        "description":"Baseline posture — spinal alignment, pelvic tilt.",
        "angles":[("Head/Neck",FACE,NECK,R_SHOULDER,175,10),("Spine",FACE,NECK,BELLY,172,10),("Pelvis",NECK,BELLY,R_HIP,172,10),("Knee (R)",R_HIP,R_KNEE,R_ANKLE,175,8),("Knee (L)",L_HIP,L_KNEE,L_ANKLE,175,8)]},
    "Hip Hinge – Lifting": {"category":"Physiotherapy","icon":"🏋️","instruction":"Hinge at hips, keep back flat, knees slightly bent.",
        "description":"Safe lifting mechanics — back injury prevention.",
        "angles":[("Hip (R)",NECK,R_HIP,R_KNEE,135,12),("Knee (R)",R_HIP,R_KNEE,R_ANKLE,145,12),("Spine",FACE,NECK,BELLY,150,15)]},
    "Squat – Parallel":    {"category":"Physiotherapy","icon":"🏋️","instruction":"Lower until thighs parallel. Knees over toes, back upright.",
        "description":"Functional squat — knee/hip mobility and stability.",
        "angles":[("Knee (R)",R_HIP,R_KNEE,R_ANKLE,95,10),("Knee (L)",L_HIP,L_KNEE,L_ANKLE,95,10),("Hip (R)",NECK,R_HIP,R_KNEE,95,12),("Spine",FACE,NECK,BELLY,155,15)]},
    "Romberg – Quiet Stance":{"category":"Neurology","icon":"🧠","instruction":"Feet together, arms at sides. Hold still for 30 seconds.",
        "description":"Balance/proprioception — sway detection.",
        "angles":[("Spine",FACE,NECK,BELLY,175,6),("Hip (R)",NECK,R_HIP,R_KNEE,175,6),("Knee (R)",R_HIP,R_KNEE,R_ANKLE,175,6),("Knee (L)",L_HIP,L_KNEE,L_ANKLE,175,6)]},
}
CATEGORIES = sorted(set(v["category"] for v in PROCEDURES.values()))

# ── Core helpers ───────────────────────────────────────────────────────────────
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

def evaluate_pose(joints,procedure):
    results=[]
    for label,ja,jb,jc,target,tol in procedure["angles"]:
        if joints[ja].sum()==0 or joints[jb].sum()==0 or joints[jc].sum()==0: continue
        cur=compute_angle(joints[ja],joints[jb],joints[jc]); dev=cur-target; absd=abs(dev)
        if absd<=tol: st_,col="✅ Good","green"
        elif absd<=tol*2: st_,col="🟡 Close","orange"
        else: st_,col="🔴 Adjust","red"
        body=label.replace("(R)","right").replace("(L)","left").lower()
        if absd<=tol: instr="Hold this position."
        elif dev<0: instr=f"Bend {body} {absd:.1f}° more (currently {cur:.0f}°, need {target}°)."
        else: instr=f"Straighten {body} by {absd:.1f}° (currently {cur:.0f}°, need {target}°)."
        results.append({"Joint":label,"Current (°)":round(cur,1),"Target (°)":target,
                        "Deviation (°)":round(dev,1),"Tolerance":f"±{tol}°","Status":st_,
                        "_color":col,"Instruction":instr})
    return results

def compliance_score(evals):
    return round(sum(1 for e in evals if e["_color"]=="green")/max(len(evals),1)*100,1)

def annotate_frame(frame_rgb,joints,evals,procedure):
    import cv2; out=frame_rgb.copy(); H,W=out.shape[:2]
    jcol={}
    for ev in evals:
        c={"green":(0,220,80),"orange":(0,165,255),"red":(60,60,255)}.get(ev["_color"],(180,180,180))
        for lbl,ja,jb,jc,*_ in procedure["angles"]:
            if lbl==ev["Joint"]: jcol[jb]=c
    for a,b in JHMDB_BONES:
        pa,pb=tuple(joints[a].astype(int)),tuple(joints[b].astype(int))
        if pa!=(0,0) and pb!=(0,0): cv2.line(out,pa,pb,jcol.get(a,jcol.get(b,(0,220,255))),3)
    for idx,(x,y) in enumerate(joints.astype(int)):
        if (x,y)!=(0,0):
            cv2.circle(out,(x,y),8,jcol.get(idx,(0,220,255)),-1); cv2.circle(out,(x,y),8,(0,0,0),1)
    for ev in evals:
        for lbl,ja,jb,jc,*_ in procedure["angles"]:
            if lbl==ev["Joint"] and joints[jb].sum()>0:
                px,py=joints[jb].astype(int)
                c={"green":(0,220,80),"orange":(0,165,255),"red":(100,100,255)}.get(ev["_color"])
                cv2.putText(out,f"{ev['Current (°)']:.0f}°",(px+10,py-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,2)
    ov=out.copy(); cv2.rectangle(ov,(0,0),(W,55),(0,0,0),-1); cv2.addWeighted(ov,0.55,out,0.45,0,out)
    score=compliance_score(evals)
    sc=(0,220,80) if score>=75 else (0,180,255) if score>=50 else (60,60,255)
    cv2.putText(out,f"COMPLIANCE: {score:.0f}%",(10,36),cv2.FONT_HERSHEY_DUPLEX,1.0,sc,2)
    bad=[e for e in evals if e["_color"]=="red"]
    if bad: cv2.putText(out,bad[0]["Instruction"][:55],(10,H-12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(100,100,255),2)
    return out

def gauge_fig(current,target,tol,label):
    fig,ax=plt.subplots(figsize=(3,2.4),facecolor="#0e1117"); ax.set_facecolor("#0e1117")
    dev=abs(current-target); pct=max(0,1-dev/max(tol*3,1))
    color="#00dc50" if dev<=tol else "#ffcc00" if dev<=tol*2 else "#ff4040"
    theta=np.linspace(0,np.pi,100)
    ax.plot(np.cos(theta),np.sin(theta),color="#333",lw=8,solid_capstyle="round")
    ax.plot(np.cos(theta[:max(1,int(pct*100))]),np.sin(theta[:max(1,int(pct*100))]),
            color=color,lw=8,solid_capstyle="round")
    ax.text(0,0.35,f"{current:.0f}°",ha="center",color=color,fontsize=18,fontweight="bold")
    ax.text(0,0.05,f"target {target}°",ha="center",color="gray",fontsize=8)
    ax.text(0,-0.25,label,ha="center",color="white",fontsize=8)
    ax.set_xlim(-1.3,1.3);ax.set_ylim(-0.6,1.3);ax.axis("off")
    plt.tight_layout(pad=0); return fig

# ── Sidebar — patient + procedure ─────────────────────────────────────────────
st.sidebar.header("👤 Patient")
patients = get_patients(hospital=user.get("org"))
patient_names = ["— New Patient —"] + [f"{p['name']} (ID:{p['id']})" for p in patients]
sel = st.sidebar.selectbox("Select patient", patient_names)

if sel == "— New Patient —":
    with st.sidebar.expander("➕ Register New Patient", expanded=True):
        with st.form("new_patient"):
            p_name  = st.text_input("Full name *")
            p_dob   = st.date_input("Date of birth")
            p_gender= st.selectbox("Gender",["Male","Female","Other"])
            p_cond  = st.text_input("Diagnosis / condition")
            p_doc   = st.text_input("Referring doctor")
            p_notes = st.text_area("Notes", height=60)
            submitted = st.form_submit_button("Register", type="primary")
            if submitted and p_name:
                new_id = add_patient(p_name, str(p_dob), p_gender, p_cond,
                                     user.get("org",""), p_doc, p_notes)
                st.success(f"Patient registered (ID: {new_id})")
                st.rerun()
    patient = None
else:
    pid = int(sel.split("ID:")[1].rstrip(")"))
    patient = get_patient(pid)
    st.sidebar.markdown(f"**{patient['name']}**  |  {patient.get('gender','')}  |  DOB: {patient.get('dob','')}")
    st.sidebar.caption(f"Condition: {patient.get('condition','N/A')}")

st.sidebar.markdown("---")
st.sidebar.header("🔬 Procedure")
category  = st.sidebar.selectbox("Department", CATEGORIES)
proc_names= [k for k,v in PROCEDURES.items() if v["category"]==category]
proc_name = st.sidebar.selectbox("Procedure", proc_names)
procedure = PROCEDURES[proc_name]
st.sidebar.info(f"**{procedure['icon']} {proc_name}**\n\n{procedure['description']}")
st.sidebar.markdown(f"**Instruction:** {procedure['instruction']}")

# ── Dashboard (top of page) ───────────────────────────────────────────────────
stats = dashboard_stats(hospital=user.get("org"))
d1,d2,d3 = st.columns(3)
d1.metric("👥 Registered Patients", stats["patients"])
d2.metric("📋 Total Sessions", stats["sessions"])
d3.metric("📈 Avg Compliance", f"{stats['avg_compliance']}%")
st.markdown("---")

# ── Main tabs ──────────────────────────────────────────────────────────────────
tab_assess, tab_history, tab_rom, tab_patients = st.tabs([
    "📷 Assessment", "📊 Patient History", "📈 ROM Progress", "👥 Patient Records"
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — ASSESSMENT (Photo + Video + Live)
# ═══════════════════════════════════════════════════════════════
with tab_assess:
    if not patient:
        st.info("Please select or register a patient in the sidebar first.")
    else:
        st.subheader(f"{procedure['icon']} {proc_name} — {patient['name']}")
        st.info(f"**Patient instruction:** {procedure['instruction']}")

        sub_live, sub_photo, sub_video = st.tabs(["📡 Live Webcam","📷 Photo","🎥 Video"])

        # ── LIVE ──────────────────────────────────────────────────────────────
        with sub_live:
            try:
                from streamlit_webrtc import webrtc_streamer,VideoProcessorBase,RTCConfiguration
                import av
                _proc=procedure
                class MedProc(VideoProcessorBase):
                    def __init__(self): self._mp=None; self.eval=[]; self.comp=0.0
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
                            self.eval=evaluate_pose(joints,_proc); self.comp=compliance_score(self.eval)
                            out=cv2.cvtColor(annotate_frame(rgb,joints,self.eval,_proc),cv2.COLOR_RGB2BGR)
                        else:
                            out=img.copy(); cv2.putText(out,"No person detected",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,80,255),2)
                        return av.VideoFrame.from_ndarray(out,format="bgr24")

                col_v,col_p=st.columns([3,2])
                with col_v:
                    ctx=webrtc_streamer(key=f"med-{proc_name}",video_processor_factory=MedProc,
                        rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
                        media_stream_constraints={"video":True,"audio":False},async_processing=True)
                with col_p:
                    st.markdown("#### Live Metrics")
                    m_ph=st.empty(); tbl_ph=st.empty(); instr_ph=st.empty()
                    if ctx.video_processor:
                        vp=ctx.video_processor
                        icon="🟢" if vp.comp>=75 else "🟡" if vp.comp>=50 else "🔴"
                        m_ph.metric("Compliance",f"{vp.comp:.0f}%",icon)
                        if vp.eval:
                            tbl_ph.dataframe(pd.DataFrame([{"Joint":e["Joint"],
                                "Current":f"{e['Current (°)']}°","Target":f"{e['Target (°)']}°",
                                "Δ":f"{e['Deviation (°)']:+.1f}°","Status":e["Status"]}
                                for e in vp.eval]),use_container_width=True,hide_index=True)
                            bad=[e for e in vp.eval if e["_color"]!="green"]
                            if bad:
                                instr_ph.warning("**Corrections:**\n\n"+"\n\n".join(
                                    f"**{e['Joint']}** — {e['Instruction']}" for e in bad[:3]))
                                # Voice
                                from utils.voice import speak_correction,build_correction_speech
                                speech_text=build_correction_speech(vp.eval,patient["name"])
                                if st.button("🔊 Speak Instructions",key="speak_live"):
                                    mp3=speak_correction(speech_text)
                                    if mp3: st.audio(mp3,format="audio/mp3")
                                    else: st.info(speech_text)
                            else:
                                instr_ph.success("✅ All joints within tolerance.")
            except ImportError:
                st.error("pip install streamlit-webrtc aiortc")

        # ── PHOTO ─────────────────────────────────────────────────────────────
        with sub_photo:
            st.caption("Full body must be visible. Good lighting, plain background.")
            img_file=st.camera_input("Capture pose",key="med_cam")
            session_notes=st.text_area("Session notes (optional)",key="med_notes",height=60)

            if img_file and st.button("🔍 Assess & Save",type="primary",key="med_assess"):
                from PIL import Image as PILImage
                frame_rgb=np.array(PILImage.open(img_file).convert("RGB"))
                with st.spinner("Detecting pose…"):
                    joints=extract_joints(frame_rgb)
                if joints is None:
                    st.warning("No person detected.")
                else:
                    evals=evaluate_pose(joints,procedure); comp=compliance_score(evals)
                    ann=annotate_frame(frame_rgb,joints,evals,procedure)

                    # Symmetry analysis
                    from utils.symmetry import analyse_symmetry,symmetry_score,plot_symmetry
                    sym_data=analyse_symmetry(joints); sym_score=symmetry_score(sym_data)

                    # Save session
                    sess_id=save_session(patient["id"],proc_name,category,comp,
                        "Pass" if comp>=75 else "Borderline" if comp>=50 else "Fail",
                        evals,1,session_notes)

                    # Save ROM
                    for ev in evals:
                        save_rom(patient["id"],proc_name,ev["Joint"],
                                 ev["Current (°)"],ev["Current (°)"],ev["Target (°)"],sess_id)

                    # Save symmetry
                    for s in sym_data:
                        save_symmetry(patient["id"],sess_id,s["joint_pair"],
                                      s["right_angle"],s["left_angle"],s["asymmetry"])

                    col_img,col_rep=st.columns([3,2])
                    with col_img:
                        st.image(ann,caption=f"Compliance: {comp:.0f}%",use_container_width=True)
                        gcols=st.columns(min(len(evals),4))
                        for gc,ev in zip(gcols,evals):
                            tol=next(tl for lbl,_,_,_,t,tl in procedure["angles"] if lbl==ev["Joint"])
                            fg=gauge_fig(ev["Current (°)"],ev["Target (°)"],tol,ev["Joint"])
                            gc.pyplot(fg); plt.close(fg)

                    with col_rep:
                        icon="🟢" if comp>=75 else "🟡" if comp>=50 else "🔴"
                        st.metric("Compliance",f"{comp:.0f}%",icon)
                        st.metric("Symmetry Score",f"{sym_score:.0f}%")
                        st.dataframe(pd.DataFrame([{"Joint":e["Joint"],
                            "Current":f"{e['Current (°)']}°","Target":f"{e['Target (°)']}°",
                            "Δ":f"{e['Deviation (°)']:+.1f}°","Status":e["Status"]}
                            for e in evals]),use_container_width=True,hide_index=True)

                        # Voice corrections
                        from utils.voice import speak_correction,build_correction_speech
                        speech_text=build_correction_speech(evals,patient["name"])
                        st.markdown("#### 🔊 Voice Instructions")
                        if st.button("▶ Play Correction Audio",key="speak_photo"):
                            mp3=speak_correction(speech_text)
                            if mp3: st.audio(mp3,format="audio/mp3")
                            else: st.info(speech_text)
                        with st.expander("Read instructions"):
                            st.write(speech_text)

                        bad=[e for e in evals if e["_color"]!="green"]
                        if not bad: st.success("✅ All joints within tolerance.")
                        else:
                            for e in bad: st.error(f"**{e['Joint']}** — {e['Instruction']}")

                    # Symmetry section
                    st.markdown("#### ↔️ Bilateral Symmetry")
                    fig_sym=plot_symmetry(sym_data)
                    if fig_sym: st.pyplot(fig_sym); plt.close(fig_sym)

                    # PDF download
                    st.markdown("#### 📄 Download Report")
                    from utils.pdf_report import generate_pdf
                    rom_hist=get_rom_history(patient["id"])
                    pdf_bytes=generate_pdf(
                        patient=patient, procedure=proc_name, department=category,
                        evaluations=evals, compliance=comp, annotated_frame=ann,
                        symmetry_data=sym_data, rom_data=rom_hist,
                        session_notes=session_notes, clinician=user.get("role","")
                    )
                    st.download_button("⬇️ Download PDF Report",data=pdf_bytes,
                        file_name=f"{patient['name'].replace(' ','_')}_{proc_name.replace(' ','_')}.pdf",
                        mime="application/pdf",type="primary")
                    st.success(f"✅ Session saved (ID: {sess_id})")

                    # ── LLM Commentary ─────────────────────────────────────
                    st.markdown("---")
                    st.markdown("### 🤖 AI Clinical Commentary")
                    st.caption("Powered by Groq llama-3.3-70b — clinical language, streaming response")
                    llm_box = st.empty()
                    try:
                        from utils.llm import live_pose_commentary, render_stream
                        gen = live_pose_commentary(
                            evaluations=evals,
                            procedure=proc_name,
                            patient_name=patient["name"],
                            stream=True,
                        )
                        render_stream(llm_box, gen)
                    except Exception as _llm_e:
                        llm_box.warning(f"LLM commentary unavailable: {_llm_e}")

        # ── VIDEO ─────────────────────────────────────────────────────────────
        with sub_video:
            vid_file=st.file_uploader("Upload video",type=["mp4","avi","mov","mkv"],key="med_vid")
            stride_v=st.slider("Every N frames",1,5,2,key="med_vstride")
            vid_notes=st.text_area("Session notes",key="med_vnotes",height=60)

            if vid_file and st.button("▶ Run Assessment",type="primary",key="med_vbtn"):
                try: import cv2,mediapipe  # noqa
                except ImportError: st.error("pip install opencv-python mediapipe==0.10.9"); st.stop()
                with tempfile.NamedTemporaryFile(suffix=".mp4",delete=False) as f:
                    f.write(vid_file.read()); tmp=f.name
                cap=cv2.VideoCapture(tmp); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                pb=st.progress(0,"Analysing…")
                joint_seq,frame_seq,eval_seq,comp_seq=[],[],[],[]
                i=0
                while True:
                    ret,frame=cap.read()
                    if not ret: break
                    if i%stride_v==0:
                        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); joints_v=extract_joints(rgb)
                        if joints_v is not None:
                            evs=evaluate_pose(joints_v,procedure); comp_v=compliance_score(evs)
                            joint_seq.append(joints_v); frame_seq.append(rgb)
                            eval_seq.append(evs); comp_seq.append(comp_v)
                        pb.progress(min(int(i/max(total,1)*100),99))
                    i+=1
                cap.release(); os.unlink(tmp); pb.progress(100,"Done!")
                if not joint_seq: st.error("No person detected."); st.stop()

                avg_comp=float(np.mean(comp_seq))
                # Save session
                avg_evals=eval_seq[len(eval_seq)//2]
                sess_id=save_session(patient["id"],proc_name,category,avg_comp,
                    "Pass" if avg_comp>=75 else "Borderline" if avg_comp>=50 else "Fail",
                    avg_evals,len(joint_seq),vid_notes)
                for ev in avg_evals:
                    save_rom(patient["id"],proc_name,ev["Joint"],
                             ev["Current (°)"],ev["Current (°)"],ev["Target (°)"],sess_id)

                # Symmetry on middle frame
                from utils.symmetry import analyse_symmetry,symmetry_score,plot_symmetry
                mid_joints=joint_seq[len(joint_seq)//2]
                sym_data=analyse_symmetry(mid_joints)
                for s in sym_data:
                    save_symmetry(patient["id"],sess_id,s["joint_pair"],
                                  s["right_angle"],s["left_angle"],s["asymmetry"])

                st.markdown("---"); st.subheader("📋 Assessment Report")
                c1,c2,c3=st.columns(3)
                c1.metric("Avg Compliance",f"{avg_comp:.0f}%")
                c2.metric("Peak Compliance",f"{max(comp_seq):.0f}%")
                c3.metric("Frames Analysed",len(joint_seq))

                # Timeline
                fig_t,ax_t=plt.subplots(figsize=(10,3),facecolor="#0e1117"); ax_t.set_facecolor("#0e1117")
                ax_t.fill_between(range(len(comp_seq)),comp_seq,alpha=0.3,color="deepskyblue")
                ax_t.plot(comp_seq,color="deepskyblue",lw=2)
                ax_t.axhline(75,color="lime",lw=1,ls="--",alpha=0.6)
                ax_t.axhline(50,color="orange",lw=1,ls="--",alpha=0.6)
                ax_t.set_ylim(0,100); ax_t.set_xlabel("Frame",color="white"); ax_t.set_ylabel("Compliance %",color="white")
                ax_t.set_title("Compliance Over Time",color="white"); ax_t.tick_params(colors="white")
                for s in ax_t.spines.values(): s.set_edgecolor("#333")
                plt.tight_layout(); st.pyplot(fig_t); plt.close(fig_t)

                # Key frames
                st.markdown("#### Key Frames")
                idxs=np.linspace(0,len(frame_seq)-1,min(6,len(frame_seq)),dtype=int)
                cols=st.columns(len(idxs))
                for col,fi in zip(cols,idxs):
                    ann=annotate_frame(frame_seq[fi],joint_seq[fi],eval_seq[fi],procedure)
                    col.image(ann,use_container_width=True,caption=f"F{fi}|{comp_seq[fi]:.0f}%")

                # Symmetry
                st.markdown("#### ↔️ Bilateral Symmetry")
                fig_s=plot_symmetry(sym_data)
                if fig_s: st.pyplot(fig_s); plt.close(fig_s)

                # Voice
                from utils.voice import speak_correction,build_correction_speech
                speech_text=build_correction_speech(avg_evals,patient["name"])
                if st.button("🔊 Play Correction Audio",key="speak_vid"):
                    mp3=speak_correction(speech_text)
                    if mp3: st.audio(mp3,format="audio/mp3")
                    else: st.info(speech_text)

                # PDF
                ann_mid=annotate_frame(frame_seq[len(frame_seq)//2],mid_joints,eval_seq[len(eval_seq)//2],procedure)
                from utils.pdf_report import generate_pdf
                rom_hist=get_rom_history(patient["id"])
                pdf_bytes=generate_pdf(patient=patient,procedure=proc_name,department=category,
                    evaluations=avg_evals,compliance=avg_comp,annotated_frame=ann_mid,
                    symmetry_data=sym_data,rom_data=rom_hist,session_notes=vid_notes,clinician=user.get("role",""))
                st.download_button("⬇️ Download PDF Report",data=pdf_bytes,
                    file_name=f"{patient['name'].replace(' ','_')}_{proc_name.replace(' ','_')}_video.pdf",
                    mime="application/pdf",type="primary")
                st.success(f"✅ Session saved (ID: {sess_id})")

                # ── LLM Session Summary ──────────────────────────────────
                st.markdown("---")
                st.markdown("### 🤖 AI Session Summary")
                llm_vid_box = st.empty()
                try:
                    from utils.llm import video_session_summary, render_stream
                    worst = [e["Joint"] for e in sorted(avg_evals, key=lambda x: abs(x["Deviation (°)"]), reverse=True)[:3]]
                    best  = [e["Joint"] for e in sorted(avg_evals, key=lambda x: abs(x["Deviation (°)"]))[:3]]
                    gen = video_session_summary(
                        session_data={
                            "procedure": proc_name,
                            "frame_count": len(joint_seq),
                            "compliance": avg_comp,
                            "overall_status": "Pass" if avg_comp >= 75 else "Borderline" if avg_comp >= 50 else "Fail",
                            "worst_joints": worst,
                            "best_joints": best,
                        },
                        stream=True,
                    )
                    render_stream(llm_vid_box, gen)
                except Exception as _llm_e:
                    llm_vid_box.warning(f"LLM summary unavailable: {_llm_e}")


# ═══════════════════════════════════════════════════════════════
# TAB 2 — PATIENT SESSION HISTORY
# ═══════════════════════════════════════════════════════════════
with tab_history:
    if not patient:
        st.info("Select a patient in the sidebar.")
    else:
        st.subheader(f"Session History — {patient['name']}")
        sessions=get_sessions(patient["id"])
        if not sessions:
            st.info("No sessions recorded yet. Run an assessment first.")
        else:
            # Summary chart
            dates=[s["created_at"][:10] for s in sessions]
            comps=[s["compliance"] for s in sessions]
            procs=[s["procedure"] for s in sessions]

            fig_h,ax_h=plt.subplots(figsize=(10,3),facecolor="#0e1117"); ax_h.set_facecolor("#0e1117")
            colors=["#00dc50" if c>=75 else "#ffcc00" if c>=50 else "#ff4040" for c in comps]
            ax_h.bar(range(len(comps)),comps,color=colors,edgecolor="#333")
            ax_h.plot(range(len(comps)),comps,"o--",color="white",lw=1,ms=4)
            ax_h.axhline(75,color="lime",lw=1,ls="--",alpha=0.6)
            ax_h.set_xticks(range(len(dates))); ax_h.set_xticklabels(dates,rotation=30,ha="right",color="white",fontsize=7)
            ax_h.set_ylim(0,100); ax_h.set_ylabel("Compliance %",color="white")
            ax_h.set_title(f"Compliance History — {patient['name']}",color="white")
            ax_h.tick_params(colors="white")
            for s in ax_h.spines.values(): s.set_edgecolor("#333")
            plt.tight_layout(); st.pyplot(fig_h); plt.close(fig_h)

            # Sessions table
            df_s=pd.DataFrame([{"Session":s["id"],"Date":s["created_at"][:16],
                "Procedure":s["procedure"],"Compliance":f"{s['compliance']}%",
                "Result":s["overall_status"],"Notes":s.get("notes","")[:40]}
                for s in sessions])
            st.dataframe(df_s,use_container_width=True,hide_index=True)

            # Session detail expander
            sel_sess=st.selectbox("View session detail",["—"]+[str(s["id"]) for s in sessions])
            if sel_sess!="—":
                sess=next(s for s in sessions if str(s["id"])==sel_sess)
                st.markdown(f"**Procedure:** {sess['procedure']}  |  **Date:** {sess['created_at'][:16]}")
                jd=sess.get("joint_data",[])
                if jd:
                    st.dataframe(pd.DataFrame([{"Joint":e["Joint"],
                        "Current":f"{e['Current (°)']}°","Target":f"{e['Target (°)']}°",
                        "Δ":f"{e['Deviation (°)']:+.1f}°","Status":e["Status"]}
                        for e in jd]),use_container_width=True,hide_index=True)
                if sess.get("notes"): st.info(f"Notes: {sess['notes']}")


# ═══════════════════════════════════════════════════════════════
# TAB 3 — ROM PROGRESS TRACKER
# ═══════════════════════════════════════════════════════════════
with tab_rom:
    if not patient:
        st.info("Select a patient in the sidebar.")
    else:
        st.subheader(f"Range of Motion Progress — {patient['name']}")
        from utils.symmetry import plot_rom_progress

        rom_all=get_rom_history(patient["id"])
        if not rom_all:
            st.info("No ROM data yet. Run assessments to start tracking progress.")
        else:
            joint_names=sorted(set(r["joint_name"] for r in rom_all))
            selected_joints=st.multiselect("Select joints to display",joint_names,default=joint_names[:3])

            for jname in selected_joints:
                records=[r for r in rom_all if r["joint_name"]==jname]
                if len(records)<2:
                    st.caption(f"{jname}: only 1 session recorded — need more for trend.")
                    continue
                fig_r=plot_rom_progress(records,jname)
                if fig_r:
                    st.pyplot(fig_r); plt.close(fig_r)

                    # Stats below each chart
                    first=records[0]["max_angle"]; last=records[-1]["max_angle"]
                    target=records[-1]["target"]; improvement=last-first
                    gap=target-last
                    ca,cb,cc=st.columns(3)
                    ca.metric("First session",f"{first:.0f}°")
                    cb.metric("Latest",f"{last:.0f}°",f"{'▲' if improvement>0 else '▼'}{abs(improvement):.0f}° vs first")
                    cc.metric("Gap to target",f"{max(gap,0):.0f}°",
                              "✅ Target reached!" if gap<=0 else f"{gap:.0f}° remaining")
                    st.markdown("---")

            # Symmetry trend
            sym_hist=get_symmetry_history(patient["id"])
            if sym_hist:
                st.subheader("↔️ Symmetry History")
                df_sym=pd.DataFrame([{"Date":s["recorded_at"][:10],"Joint Pair":s["joint_pair"],
                    "Right":f"{s['right_angle']}°","Left":f"{s['left_angle']}°",
                    "Asymmetry":f"{s['asymmetry']}°",
                    "Risk":"🟢 OK" if s["asymmetry"]<5 else "🟡 Mild" if s["asymmetry"]<10 else "🔴 Significant"}
                    for s in sym_hist])
                st.dataframe(df_sym,use_container_width=True,hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — PATIENT RECORDS
# ═══════════════════════════════════════════════════════════════
with tab_patients:
    st.subheader(f"Patient Records — {user.get('org','')}")
    all_patients=get_patients(hospital=user.get("org"))
    if not all_patients:
        st.info("No patients registered yet.")
    else:
        df_p=pd.DataFrame([{"ID":p["id"],"Name":p["name"],"DOB":p["dob"],
            "Gender":p["gender"],"Condition":p.get("condition",""),
            "Doctor":p.get("doctor",""),"Registered":p.get("created_at","")[:10]}
            for p in all_patients])
        st.dataframe(df_p,use_container_width=True,hide_index=True)

        st.markdown("#### Recent Sessions (All Patients)")
        from utils.database import get_all_sessions
        all_sess=get_all_sessions(hospital=user.get("org"))[:20]
        if all_sess:
            df_all=pd.DataFrame([{"Patient":s["patient_name"],"Procedure":s["procedure"],
                "Date":s["created_at"][:16],"Compliance":f"{s['compliance']}%",
                "Result":s["overall_status"]}
                for s in all_sess])
            st.dataframe(df_all,use_container_width=True,hide_index=True)

with st.expander("📐 Reference Angles"):
    st.dataframe(pd.DataFrame([{"Joint":lbl,"Target (°)":t,"Tolerance":f"±{tol}°",
        "Range":f"{t-tol}°–{t+tol}°"} for lbl,_,_,_,t,tol in procedure["angles"]]),
        use_container_width=True,hide_index=True)
    st.caption("🟢 within tolerance | 🟡 within 2× tolerance | 🔴 outside 2× tolerance")
    st.caption("*AI-assisted guidance only — does not replace clinical judgement.*")
