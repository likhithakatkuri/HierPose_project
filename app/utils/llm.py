"""
Groq LLM Commentary Engine for HierPose
========================================
Uses Groq REST API directly via `requests` — no `groq` package required.
Groq is OpenAI-compatible: POST to https://api.groq.com/openai/v1/chat/completions

Model: llama-3.3-70b-versatile  (best reasoning, clinical language)
Fast:  llama-3.1-8b-instant     (live video, low latency)
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import Generator, List, Dict, Optional, Any

import requests

# Load .env file from project root (works even when running from app/pages/)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass  # dotenv not installed — fall back to system env vars

_GROQ_URL   = "https://api.groq.com/openai/v1/chat/completions"
_MODEL_FULL = "llama-3.3-70b-versatile"
_MODEL_FAST = "llama-3.1-8b-instant"


def _headers() -> dict:
    """Build headers fresh each call so the key is always current."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your .env file or export it in your terminal."
        )
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


# ─────────────────────────────────────────────────────────────────────────────
# Core streaming helper
# ─────────────────────────────────────────────────────────────────────────────

def stream_response(prompt: str, system: str = "", fast: bool = False) -> Generator[str, None, None]:
    """
    Yield text chunks from Groq streaming response.
    Use in Streamlit: st.write_stream(stream_response(prompt))
    """
    model = _MODEL_FAST if fast else _MODEL_FULL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "max_tokens": 500,
        "temperature": 0.6,
    }
    with requests.post(_GROQ_URL, headers=_headers(), json=payload, stream=True, timeout=30) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if line.startswith("data: "):
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        yield delta
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


def one_shot(prompt: str, system: str = "", fast: bool = False, max_tokens: int = 400) -> str:
    """Single call, returns full text."""
    model = _MODEL_FAST if fast else _MODEL_FULL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.6,
    }
    resp = requests.post(_GROQ_URL, headers=_headers(), json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────────────────────
# Pose Assessment Commentary
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PHYSIO = (
    "You are a highly experienced clinical physiotherapist and biomechanics expert. "
    "You give clear, specific, actionable guidance in plain English. "
    "No bullet points. Write in flowing sentences. Keep responses under 5 sentences. "
    "Always mention the most critical correction first."
)

def live_pose_commentary(
    evaluations: List[Dict],
    procedure: str,
    patient_name: str = "",
    stream: bool = True,
):
    """
    Live commentary: what is wrong right now and how to fix it.
    Called every time a new frame is assessed.
    Uses FAST model for low latency.
    """
    bad  = [e for e in evaluations if e.get("_color") != "green"]
    good = [e for e in evaluations if e.get("_color") == "green"]
    compliance = round(len(good) / max(len(evaluations), 1) * 100)

    if not bad:
        prompt = (
            f"The patient{' ' + patient_name if patient_name else ''} is performing "
            f"'{procedure}' correctly. All {len(good)} joints are within target range "
            f"(compliance 100%). Give a brief encouraging comment and tell them to hold the position."
        )
    else:
        issues = "\n".join(
            f"- {e['Joint']}: at {e.get('Current (°)', 0):.1f}° but needs {e.get('Target (°)', 0):.1f}° "
            f"(off by {e.get('Deviation (°)', 0):+.1f}°)"
            for e in bad[:3]
        )
        prompt = (
            f"Procedure: {procedure}. Patient{': ' + patient_name if patient_name else ''}. "
            f"Compliance: {compliance}% ({len(bad)} joint(s) out of range).\n\n"
            f"Issues detected:\n{issues}\n\n"
            f"Describe in plain English:\n"
            f"1. What the patient is doing wrong RIGHT NOW (1 sentence)\n"
            f"2. Exactly how to correct each issue (be specific: 'raise your left arm', 'tilt chin up')\n"
            f"3. Why it matters clinically (1 sentence)\n"
            f"Keep it under 4 sentences total. Speak directly to the patient."
        )

    if stream:
        return stream_response(prompt, system=_SYSTEM_PHYSIO, fast=True)
    return one_shot(prompt, system=_SYSTEM_PHYSIO, fast=True)


def assessment_report_narrative(
    evaluations: List[Dict],
    procedure: str,
    compliance: float,
    patient_name: str = "",
    symmetry_data: Optional[List[Dict]] = None,
    stream: bool = True,
):
    """
    Detailed post-assessment narrative for the PDF report and history tab.
    Uses FULL model for quality.
    """
    bad  = [e for e in evaluations if e.get("_color") != "green"]
    good = [e for e in evaluations if e.get("_color") == "green"]

    joint_lines = "\n".join(
        f"  {e['Joint']}: {e.get('Current (°)', 0):.1f}° / target {e.get('Target (°)', 0):.1f}° "
        f"— {'WITHIN range' if e.get('_color') == 'green' else 'OUT OF range'}"
        for e in evaluations
    )

    sym_lines = ""
    if symmetry_data:
        sym_lines = "\n\nBilateral symmetry:\n" + "\n".join(
            f"  {s['joint_pair']}: R={s['right_angle']}° L={s['left_angle']}° "
            f"asymmetry={s['asymmetry']}°"
            for s in symmetry_data
        )

    prompt = (
        f"Write a professional clinical assessment narrative for:\n"
        f"Patient: {patient_name or 'Unknown'}\n"
        f"Procedure: {procedure}\n"
        f"Overall compliance: {compliance:.0f}%\n\n"
        f"Joint measurements:\n{joint_lines}"
        f"{sym_lines}\n\n"
        f"Write 3-4 sentences covering:\n"
        f"1. Overall assessment verdict\n"
        f"2. Specific joint findings (focus on what's wrong)\n"
        f"3. Bilateral symmetry observation (if data given)\n"
        f"4. Clinical recommendation / next steps\n"
        f"Write in formal clinical report style (third person: 'The patient demonstrates...'). "
        f"No bullet points."
    )
    if stream:
        return stream_response(prompt, system=_SYSTEM_PHYSIO)
    return one_shot(prompt, system=_SYSTEM_PHYSIO)


# ─────────────────────────────────────────────────────────────────────────────
# Gait Analysis Commentary
# ─────────────────────────────────────────────────────────────────────────────

def gait_report_narrative(gait_report_dict: Dict, stream: bool = True):
    """
    Plain-English explanation of a GaitReport for the Gait Lab page.
    """
    prompt = (
        f"Explain this gait analysis report in plain English to a patient:\n\n"
        f"Gait Score: {gait_report_dict.get('overall_gait_score', 0):.0f}/100\n"
        f"GDI: {gait_report_dict.get('gait_deviation_index', 0):.0f}\n"
        f"Cadence: {gait_report_dict.get('cadence_spm', 0):.0f} steps/min\n"
        f"Step Symmetry Index: {gait_report_dict.get('step_symmetry_index', 0):.1f}%\n"
        f"Stance Ratio R: {gait_report_dict.get('stance_ratio_r', 0):.2f}\n"
        f"Trunk Sway: {gait_report_dict.get('trunk_sway_range_deg', 0):.1f}°\n"
        f"Hip Drop R: {gait_report_dict.get('hip_drop_r', 0):.1f}°\n"
        f"Hip Drop L: {gait_report_dict.get('hip_drop_l', 0):.1f}°\n"
        f"Bilateral Waveform Correlation: {gait_report_dict.get('bilateral_waveform_corr', 1):.2f}\n"
        f"Risk flags: {'; '.join(gait_report_dict.get('risk_flags', []))}\n\n"
        f"Explain what this means in simple, non-technical language. "
        f"What is happening with their walking? What should they focus on improving? "
        f"What exercises or tips would help? Keep it under 5 sentences."
    )
    system = (
        "You are a friendly clinical gait specialist. Explain findings simply, "
        "as if talking to a patient. Avoid jargon. Be encouraging but honest."
    )
    if stream:
        return stream_response(prompt, system=system)
    return one_shot(prompt, system=system)


def compensation_narrative(comp_report_dict: Dict, stream: bool = True):
    """
    Explain kinematic compensation findings to patient/clinician.
    """
    root_causes = comp_report_dict.get("root_causes", [])
    severity    = comp_report_dict.get("overall_severity", "none")
    score       = comp_report_dict.get("compensation_score", 0)
    active = [c for c in comp_report_dict.get("compensations", [])
              if c.get("severity", "none") != "none"]

    if severity == "none" or not active:
        prompt = "The movement analysis shows no significant compensation patterns. Movement is within normal kinematic relationships. Confirm this with the patient."
        return (chunk for chunk in [prompt]) if stream else prompt

    issues = "\n".join(
        f"- {c['joint_name']}: {c['residual']:+.1f}° off expected "
        f"(severity: {c['severity']}, direction: {c['direction']})"
        for c in active[:4]
    )
    prompt = (
        f"Compensation severity: {severity} (score {score:.0f}/100)\n"
        f"Root causes identified: {', '.join(root_causes) if root_causes else 'undetermined'}\n"
        f"Active compensations:\n{issues}\n\n"
        f"In plain English (3-4 sentences):\n"
        f"1. What is the patient doing to compensate?\n"
        f"2. What is likely causing this compensation (pain, weakness, restriction)?\n"
        f"3. What should the clinician or patient do about it?\n"
        f"Speak directly: 'Your body is compensating by...' "
    )
    system = "You are a clinical movement specialist. Explain compensation patterns simply and practically."
    if stream:
        return stream_response(prompt, system=system)
    return one_shot(prompt, system=system)


def injury_risk_narrative(risk_report_dict: Dict, stream: bool = True):
    """
    Explain injury risk findings.
    """
    level   = risk_report_dict.get("overall_level", "LOW")
    overall = risk_report_dict.get("overall_risk", 0)
    top_factors = risk_report_dict.get("top_risk_factors", [])
    injury_risks = risk_report_dict.get("injury_risks", {})

    high_injuries = [k for k, v in injury_risks.items()
                     if isinstance(v, dict) and v.get("risk_level") in ("HIGH","CRITICAL")]

    prompt = (
        f"Biomechanical injury risk assessment:\n"
        f"Overall: {level} ({overall:.0f}%)\n"
        f"High-risk injuries: {', '.join(high_injuries) if high_injuries else 'none'}\n"
        f"Top risk factors: {', '.join(top_factors[:3]) if top_factors else 'none detected'}\n\n"
        f"Explain in 3-4 plain English sentences:\n"
        f"1. What is the current injury risk level and what it means\n"
        f"2. What movement patterns are creating the risk\n"
        f"3. What the person should do to reduce their risk\n"
        f"Be direct and practical. Speak to the patient."
    )
    system = "You are a sports medicine expert and injury prevention specialist. Be clear and actionable."
    if stream:
        return stream_response(prompt, system=system)
    return one_shot(prompt, system=system)


# ─────────────────────────────────────────────────────────────────────────────
# Exercise Guidance
# ─────────────────────────────────────────────────────────────────────────────

def exercise_guide(
    exercise_name: str,
    joint_focus: str = "",
    patient_condition: str = "",
    stream: bool = True,
):
    """
    How to correctly perform an exercise — for the skeleton demo coach.
    Returns step-by-step instructions + key checkpoints.
    """
    prompt = (
        f"Exercise: {exercise_name}\n"
        f"{'Joint focus: ' + joint_focus if joint_focus else ''}\n"
        f"{'Patient condition: ' + patient_condition if patient_condition else ''}\n\n"
        f"Provide:\n"
        f"1. Starting position (2 sentences — exactly where body parts should be)\n"
        f"2. Movement execution (2 sentences — what moves, in what direction, how far)\n"
        f"3. Key form checkpoints (2 specific things to watch: knee alignment, spine, etc.)\n"
        f"4. Common mistakes to avoid (1-2 sentences)\n"
        f"5. Breathing pattern (1 sentence)\n\n"
        f"Be extremely specific about joint angles and body positions. "
        f"Use directions: 'feet shoulder-width apart', 'knee at 90 degrees', etc."
    )
    system = (
        "You are a clinical exercise physiologist and certified physiotherapist. "
        "Give precise biomechanical exercise instructions suitable for rehabilitation."
    )
    if stream:
        return stream_response(prompt, system=system)
    return one_shot(prompt, system=system, max_tokens=600)


def video_session_summary(
    session_data: Dict,
    stream: bool = True,
):
    """
    After a full video session analysis — what happened, what to do next.
    """
    prompt = (
        f"Session summary:\n"
        f"Procedure: {session_data.get('procedure', 'Unknown')}\n"
        f"Total frames: {session_data.get('frame_count', 0)}\n"
        f"Overall compliance: {session_data.get('compliance', 0):.0f}%\n"
        f"Status: {session_data.get('overall_status', 'Unknown')}\n"
        f"Worst joints: {', '.join(session_data.get('worst_joints', []))}\n"
        f"Best joints: {', '.join(session_data.get('best_joints', []))}\n\n"
        f"Write a 3-4 sentence session debrief:\n"
        f"1. Overall session performance verdict\n"
        f"2. What went well vs what needs work\n"
        f"3. Specific recommendations for next session\n"
        f"Write in second person ('You did well...'). Be encouraging."
    )
    system = "You are a supportive physiotherapist providing session feedback."
    if stream:
        return stream_response(prompt, system=system)
    return one_shot(prompt, system=system)


def adaptive_care_narrative(ace_report_dict: Dict, patient_name: str = "", stream: bool = True):
    """
    Explain the adaptive care findings and protocol in patient-friendly language.
    """
    dri       = ace_report_dict.get("discharge_readiness_index", 50)
    n_sessions= ace_report_dict.get("n_sessions", 0)
    reg_alerts= ace_report_dict.get("regression_alerts", [])
    rec       = ace_report_dict.get("discharge_recommendation", "")

    prompt = (
        f"Patient{': ' + patient_name if patient_name else ''} longitudinal rehab summary:\n"
        f"Sessions completed: {n_sessions}\n"
        f"Discharge Readiness Index: {dri:.0f}%\n"
        f"Regression alerts: {', '.join(reg_alerts) if reg_alerts else 'none'}\n"
        f"Recommendation: {rec}\n\n"
        f"Write a 3-4 sentence progress summary for the patient:\n"
        f"1. How well they have been doing overall\n"
        f"2. Any joints that are getting worse (be honest but sensitive)\n"
        f"3. What they should focus on this week\n"
        f"4. When they might be ready for discharge or next phase\n"
        f"Be warm, encouraging, and specific."
    )
    system = "You are a caring physiotherapist giving a patient their weekly progress update."
    if stream:
        return stream_response(prompt, system=system)
    return one_shot(prompt, system=system)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-powered pose generation — creates skeleton keyframes for any exercise
# ─────────────────────────────────────────────────────────────────────────────

def generate_exercise_pose(exercise_name: str) -> Dict:
    """
    Ask LLM to generate skeleton keyframe coordinates for any exercise.

    Returns dict with keys:
        frames      : list of {"name": str, "joints": list[list[float,float]]}
        instructions: str
        joint_focus : str
        cues        : list[str]
        category    : str

    Joint order (15 joints, JHMDB standard):
        0=NECK  1=BELLY  2=FACE
        3=R_SHOULDER  4=L_SHOULDER
        5=R_HIP  6=L_HIP
        7=R_ELBOW  8=L_ELBOW
        9=R_KNEE  10=L_KNEE
        11=R_WRIST  12=L_WRIST
        13=R_ANKLE  14=L_ANKLE

    Coordinates are normalised 0-1:
        x=0 left edge, x=1 right edge, x=0.5 center
        y=0 top, y=1 bottom  (y increases downward)
    """
    system = (
        "You are a biomechanics expert and certified physiotherapist. "
        "You return ONLY valid JSON. No markdown fences, no explanation, just the JSON object."
    )

    # Explicit numbered list so LLM can't miscounts
    STANDING = [[0.50,0.12],[0.50,0.38],[0.50,0.03],[0.37,0.22],[0.63,0.22],
                [0.40,0.52],[0.60,0.52],[0.30,0.38],[0.70,0.38],[0.40,0.72],
                [0.60,0.72],[0.28,0.52],[0.72,0.52],[0.40,0.92],[0.60,0.92]]

    joint_list = (
        "Joints MUST be listed in this exact order (15 total):\n"
        " 0:NECK      1:BELLY     2:FACE\n"
        " 3:R_SHOULDER 4:L_SHOULDER\n"
        " 5:R_HIP     6:L_HIP\n"
        " 7:R_ELBOW   8:L_ELBOW\n"
        " 9:R_KNEE   10:L_KNEE\n"
        "11:R_WRIST  12:L_WRIST\n"
        "13:R_ANKLE  14:L_ANKLE\n\n"
        f"Standing reference (copy and modify for your exercise):\n{STANDING}"
    )

    prompt = f"""You are creating a skeleton animation for: "{exercise_name}"

{joint_list}

Coordinate system:
- x: 0.0=left  0.5=center  1.0=right
- y: 0.0=top   0.5=middle  1.0=bottom  (y INCREASES downward)
- Head at y≈0.03, feet at y≈0.92 for standing

CRITICAL: Each frame's "joints" array MUST have EXACTLY 15 pairs. Count them carefully.

Return ONLY valid JSON, no markdown fences, no commentary:
{{
  "frames": [
    {{"name": "Start Position", "joints": [[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}},
    {{"name": "Peak/Execution",  "joints": [[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}},
    {{"name": "Return",          "joints": [[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y],[x,y]]}}
  ],
  "instructions": "3-5 sentence step-by-step guide.",
  "joint_focus": "main joints involved",
  "cues": ["coaching cue 1", "cue 2", "cue 3"],
  "category": "Strength / Core / Rehabilitation / Flexibility / Cardio / Gait"
}}"""

    def _fix_joints(data: Dict) -> Dict:
        """Repair frames that have wrong joint count by padding/trimming to 15."""
        for frame in data.get("frames", []):
            joints = frame.get("joints", [])
            while len(joints) < 15:
                joints.append([0.5, 0.5])   # pad missing joints to center
            frame["joints"] = joints[:15]    # trim extras
        return data

    try:
        raw = one_shot(prompt, system=system, max_tokens=1000, fast=False)
        raw = raw.strip()
        # Strip markdown fences
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                if part.startswith("json"):
                    raw = part[4:].strip()
                    break
                elif part.strip().startswith("{"):
                    raw = part.strip()
                    break
        # Find the JSON object
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]
        data = json.loads(raw)
        data = _fix_joints(data)  # auto-repair wrong joint count

        return data

    except Exception as e:
        # Fallback: return a neutral standing pose with error info
        standing = [[0.50,0.12],[0.50,0.38],[0.50,0.03],[0.37,0.22],[0.63,0.22],
                    [0.40,0.52],[0.60,0.52],[0.30,0.38],[0.70,0.38],[0.40,0.72],
                    [0.60,0.72],[0.28,0.52],[0.72,0.52],[0.40,0.92],[0.60,0.92]]
        return {
            "frames": [{"name": "Reference Pose", "joints": standing}],
            "instructions": f"Could not generate specific pose for '{exercise_name}'. Showing reference standing pose. Error: {e}",
            "joint_focus": "all joints",
            "cues": ["Consult a physiotherapist for specific guidance"],
            "category": "General",
            "_error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit helper — renders streaming text with typewriter effect
# ─────────────────────────────────────────────────────────────────────────────

def render_stream(container, generator):
    """
    Display streaming LLM output in a Streamlit container.
    Usage:
        with st.container():
            render_stream(st.empty(), live_pose_commentary(evals, proc))
    """
    full_text = ""
    for chunk in generator:
        full_text += chunk
        container.markdown(full_text + "▌")
    container.markdown(full_text)
    return full_text
