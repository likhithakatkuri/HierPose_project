"""
Skeleton Animator
=================
Generates animated skeleton demonstrations using matplotlib.
Each exercise is defined as a sequence of keyframe poses (JHMDB 15-joint).
Frames are interpolated and rendered to an animated GIF.

JHMDB joint indices:
  NECK=0  BELLY=1  FACE=2
  R_SHOULDER=3  L_SHOULDER=4
  R_HIP=5  L_HIP=6
  R_ELBOW=7  L_ELBOW=8
  R_KNEE=9  L_KNEE=10
  R_WRIST=11  L_WRIST=12
  R_ANKLE=13  L_ANKLE=14
"""
from __future__ import annotations
import io
import numpy as np
from typing import List, Tuple, Dict

# ── Skeleton connectivity (JHMDB edges) ────────────────────────────────────
SKELETON_EDGES = [
    (0, 2),   # neck → face
    (0, 1),   # neck → belly
    (0, 3),   # neck → r_shoulder
    (0, 4),   # neck → l_shoulder
    (3, 7),   # r_shoulder → r_elbow
    (7, 11),  # r_elbow → r_wrist
    (4, 8),   # l_shoulder → l_elbow
    (8, 12),  # l_elbow → l_wrist
    (1, 5),   # belly → r_hip
    (1, 6),   # belly → l_hip
    (5, 9),   # r_hip → r_knee
    (9, 13),  # r_knee → r_ankle
    (6, 10),  # l_hip → l_knee
    (10, 14), # l_knee → l_ankle
]

# Joint colors: head=white, right=blue, left=orange, center=gray
JOINT_COLORS = {
    0: "#ffffff", 1: "#aaaaaa", 2: "#ffffff",
    3: "#4499dd", 4: "#dd9944",
    5: "#4499dd", 6: "#dd9944",
    7: "#4499dd", 8: "#dd9944",
    9: "#4499dd", 10: "#dd9944",
    11: "#4499dd", 12: "#dd9944",
    13: "#4499dd", 14: "#dd9944",
}

EDGE_COLORS = {
    (0,2):"#ffffff",(0,1):"#aaaaaa",
    (0,3):"#4499dd",(0,4):"#dd9944",
    (3,7):"#4499dd",(7,11):"#4499dd",
    (4,8):"#dd9944",(8,12):"#dd9944",
    (1,5):"#4499dd",(1,6):"#dd9944",
    (5,9):"#4499dd",(9,13):"#4499dd",
    (6,10):"#dd9944",(10,14):"#dd9944",
}


# ─────────────────────────────────────────────────────────────────────────────
# Reference Poses (JHMDB 15-joint, normalised 0-1)
# Rows: NECK BELLY FACE R_SH L_SH R_HIP L_HIP R_ELB L_ELB R_KNEE L_KNEE R_WRI L_WRI R_ANK L_ANK
# ─────────────────────────────────────────────────────────────────────────────

def _pose(coords: List[Tuple[float,float]]) -> np.ndarray:
    return np.array(coords, dtype=np.float32)

# Standing neutral
STAND = _pose([
    [0.50, 0.12],  # neck
    [0.50, 0.38],  # belly
    [0.50, 0.03],  # face
    [0.37, 0.22],  # r_shoulder
    [0.63, 0.22],  # l_shoulder
    [0.40, 0.52],  # r_hip
    [0.60, 0.52],  # l_hip
    [0.30, 0.38],  # r_elbow
    [0.70, 0.38],  # l_elbow
    [0.40, 0.72],  # r_knee
    [0.60, 0.72],  # l_knee
    [0.28, 0.52],  # r_wrist
    [0.72, 0.52],  # l_wrist
    [0.40, 0.92],  # r_ankle
    [0.60, 0.92],  # l_ankle
])

# ── Squat ──────────────────────────────────────────────────────────────────
SQUAT_BOTTOM = _pose([
    [0.50, 0.32],  # neck
    [0.50, 0.56],  # belly
    [0.50, 0.22],  # face
    [0.38, 0.42],  # r_shoulder
    [0.62, 0.42],  # l_shoulder
    [0.43, 0.66],  # r_hip
    [0.57, 0.66],  # l_hip
    [0.30, 0.58],  # r_elbow (arms forward for balance)
    [0.70, 0.58],  # l_elbow
    [0.35, 0.80],  # r_knee (knees forward)
    [0.65, 0.80],  # l_knee
    [0.30, 0.72],  # r_wrist
    [0.70, 0.72],  # l_wrist
    [0.40, 0.92],  # r_ankle
    [0.60, 0.92],  # l_ankle
])

# ── Push-up top ─────────────────────────────────────────────────────────────
PUSHUP_TOP = _pose([
    [0.50, 0.30],  # neck
    [0.50, 0.45],  # belly
    [0.50, 0.22],  # face
    [0.38, 0.35],  # r_shoulder
    [0.62, 0.35],  # l_shoulder
    [0.42, 0.50],  # r_hip
    [0.58, 0.50],  # l_hip
    [0.32, 0.45],  # r_elbow
    [0.68, 0.45],  # l_elbow
    [0.42, 0.62],  # r_knee
    [0.58, 0.62],  # l_knee
    [0.32, 0.55],  # r_wrist
    [0.68, 0.55],  # l_wrist
    [0.42, 0.75],  # r_ankle
    [0.58, 0.75],  # l_ankle
])

PUSHUP_BOTTOM = _pose([
    [0.50, 0.42],  # neck
    [0.50, 0.52],  # belly
    [0.50, 0.34],  # face
    [0.38, 0.47],  # r_shoulder
    [0.62, 0.47],  # l_shoulder
    [0.42, 0.55],  # r_hip
    [0.58, 0.55],  # l_hip
    [0.36, 0.56],  # r_elbow (elbows bent)
    [0.64, 0.56],  # l_elbow
    [0.42, 0.62],  # r_knee
    [0.58, 0.62],  # l_knee
    [0.36, 0.64],  # r_wrist
    [0.64, 0.64],  # l_wrist
    [0.42, 0.75],  # r_ankle
    [0.58, 0.75],  # l_ankle
])

# ── Shoulder abduction (arm raise) ──────────────────────────────────────────
SHOULDER_ABDUCTION = _pose([
    [0.50, 0.12],  # neck
    [0.50, 0.38],  # belly
    [0.50, 0.03],  # face
    [0.24, 0.12],  # r_shoulder (arm overhead)
    [0.76, 0.22],  # l_shoulder (neutral)
    [0.40, 0.52],  # r_hip
    [0.60, 0.52],  # l_hip
    [0.15, 0.22],  # r_elbow
    [0.70, 0.38],  # l_elbow
    [0.40, 0.72],  # r_knee
    [0.60, 0.72],  # l_knee
    [0.10, 0.32],  # r_wrist
    [0.72, 0.52],  # l_wrist
    [0.40, 0.92],  # r_ankle
    [0.60, 0.92],  # l_ankle
])

# ── Knee flexion (standing straight-leg raise to bent) ──────────────────────
KNEE_FLEXION = _pose([
    [0.50, 0.12],  # neck
    [0.50, 0.38],  # belly
    [0.50, 0.03],  # face
    [0.37, 0.22],  # r_shoulder
    [0.63, 0.22],  # l_shoulder
    [0.40, 0.52],  # r_hip
    [0.60, 0.52],  # l_hip
    [0.30, 0.38],  # r_elbow
    [0.70, 0.38],  # l_elbow
    [0.40, 0.72],  # r_knee
    [0.62, 0.62],  # l_knee (knee raised and bent)
    [0.28, 0.52],  # r_wrist
    [0.72, 0.52],  # l_wrist
    [0.40, 0.92],  # r_ankle
    [0.68, 0.55],  # l_ankle (foot up)
])

# ── Hip bridge (lying down) ──────────────────────────────────────────────────
HIP_BRIDGE_DOWN = _pose([
    [0.50, 0.75],  # neck (lying)
    [0.50, 0.72],  # belly
    [0.50, 0.82],  # face
    [0.37, 0.70],  # r_shoulder
    [0.63, 0.70],  # l_shoulder
    [0.42, 0.68],  # r_hip
    [0.58, 0.68],  # l_hip
    [0.30, 0.72],  # r_elbow
    [0.70, 0.72],  # l_elbow
    [0.38, 0.55],  # r_knee (knees bent, feet flat)
    [0.62, 0.55],  # l_knee
    [0.28, 0.70],  # r_wrist
    [0.72, 0.70],  # l_wrist
    [0.35, 0.42],  # r_ankle
    [0.65, 0.42],  # l_ankle
])

HIP_BRIDGE_UP = _pose([
    [0.50, 0.75],  # neck
    [0.50, 0.60],  # belly (hips raised)
    [0.50, 0.82],  # face
    [0.37, 0.70],  # r_shoulder
    [0.63, 0.70],  # l_shoulder
    [0.42, 0.50],  # r_hip (raised)
    [0.58, 0.50],  # l_hip (raised)
    [0.30, 0.72],  # r_elbow
    [0.70, 0.72],  # l_elbow
    [0.38, 0.52],  # r_knee
    [0.62, 0.52],  # l_knee
    [0.28, 0.70],  # r_wrist
    [0.72, 0.70],  # l_wrist
    [0.35, 0.42],  # r_ankle
    [0.65, 0.42],  # l_ankle
])

# ── PA Chest X-ray position ──────────────────────────────────────────────────
XRAY_PA = _pose([
    [0.50, 0.10],  # neck
    [0.50, 0.36],  # belly
    [0.50, 0.02],  # face (chin up)
    [0.30, 0.22],  # r_shoulder (rotated forward)
    [0.70, 0.22],  # l_shoulder
    [0.40, 0.52],  # r_hip
    [0.60, 0.52],  # l_hip
    [0.22, 0.38],  # r_elbow (arms rotated out)
    [0.78, 0.38],  # l_elbow
    [0.40, 0.72],  # r_knee
    [0.60, 0.72],  # l_knee
    [0.20, 0.50],  # r_wrist
    [0.80, 0.50],  # l_wrist
    [0.40, 0.92],  # r_ankle
    [0.60, 0.92],  # l_ankle
])

# ── Walking gait frames ──────────────────────────────────────────────────────
WALK_MID = _pose([
    [0.50, 0.12],  # neck
    [0.50, 0.38],  # belly
    [0.50, 0.03],  # face
    [0.38, 0.22],  # r_shoulder
    [0.62, 0.22],  # l_shoulder
    [0.41, 0.52],  # r_hip
    [0.59, 0.52],  # l_hip
    [0.34, 0.36],  # r_elbow (arm swing)
    [0.64, 0.40],  # l_elbow
    [0.37, 0.68],  # r_knee
    [0.63, 0.75],  # l_knee (stance)
    [0.30, 0.50],  # r_wrist
    [0.70, 0.52],  # l_wrist
    [0.35, 0.88],  # r_ankle
    [0.63, 0.92],  # l_ankle
])

# ── More reference poses ──────────────────────────────────────────────────────

# Lunge (right leg forward)
LUNGE = _pose([
    [0.50, 0.20],[0.50, 0.44],[0.50, 0.11],
    [0.38, 0.30],[0.62, 0.30],
    [0.43, 0.52],[0.57, 0.52],
    [0.30, 0.44],[0.70, 0.44],
    [0.35, 0.70],[0.65, 0.70],
    [0.28, 0.56],[0.72, 0.56],
    [0.32, 0.90],[0.68, 0.88],
])

# Deadlift (hip hinge, bar at shins)
DEADLIFT_HINGE = _pose([
    [0.50, 0.28],[0.50, 0.52],[0.50, 0.18],
    [0.38, 0.38],[0.62, 0.38],
    [0.44, 0.58],[0.56, 0.58],
    [0.36, 0.60],[0.64, 0.60],
    [0.40, 0.72],[0.60, 0.72],
    [0.38, 0.72],[0.62, 0.72],
    [0.40, 0.90],[0.60, 0.90],
])

# Plank (horizontal)
PLANK = _pose([
    [0.50, 0.38],[0.50, 0.46],[0.50, 0.30],
    [0.38, 0.42],[0.62, 0.42],
    [0.44, 0.48],[0.56, 0.48],
    [0.32, 0.52],[0.68, 0.52],
    [0.44, 0.54],[0.56, 0.54],
    [0.32, 0.60],[0.68, 0.60],
    [0.44, 0.62],[0.56, 0.62],
])

# Overhead press (both arms up)
OVERHEAD_PRESS_UP = _pose([
    [0.50, 0.12],[0.50, 0.38],[0.50, 0.03],
    [0.36, 0.08],[0.64, 0.08],
    [0.40, 0.52],[0.60, 0.52],
    [0.30, 0.10],[0.70, 0.10],
    [0.40, 0.72],[0.60, 0.72],
    [0.25, 0.14],[0.75, 0.14],
    [0.40, 0.92],[0.60, 0.92],
])

OVERHEAD_PRESS_DOWN = _pose([
    [0.50, 0.12],[0.50, 0.38],[0.50, 0.03],
    [0.37, 0.22],[0.63, 0.22],
    [0.40, 0.52],[0.60, 0.52],
    [0.30, 0.28],[0.70, 0.28],
    [0.40, 0.72],[0.60, 0.72],
    [0.32, 0.38],[0.68, 0.38],
    [0.40, 0.92],[0.60, 0.92],
])

# Bicep curl (right arm)
BICEP_CURL_UP = _pose([
    [0.50, 0.12],[0.50, 0.38],[0.50, 0.03],
    [0.37, 0.22],[0.63, 0.22],
    [0.40, 0.52],[0.60, 0.52],
    [0.33, 0.32],[0.70, 0.38],
    [0.40, 0.72],[0.60, 0.72],
    [0.36, 0.24],[0.72, 0.52],
    [0.40, 0.92],[0.60, 0.92],
])

# Lateral raise (both arms at shoulder height)
LATERAL_RAISE = _pose([
    [0.50, 0.12],[0.50, 0.38],[0.50, 0.03],
    [0.22, 0.22],[0.78, 0.22],
    [0.40, 0.52],[0.60, 0.52],
    [0.14, 0.38],[0.86, 0.38],
    [0.40, 0.72],[0.60, 0.72],
    [0.10, 0.42],[0.90, 0.42],
    [0.40, 0.92],[0.60, 0.92],
])

# Romanian deadlift (hinge, legs straighter)
RDL = _pose([
    [0.50, 0.25],[0.50, 0.48],[0.50, 0.16],
    [0.38, 0.35],[0.62, 0.35],
    [0.44, 0.54],[0.56, 0.54],
    [0.38, 0.52],[0.62, 0.52],
    [0.42, 0.68],[0.58, 0.68],
    [0.38, 0.62],[0.62, 0.62],
    [0.42, 0.88],[0.58, 0.88],
])

# Step-up (right leg on step)
STEP_UP = _pose([
    [0.50, 0.14],[0.50, 0.40],[0.50, 0.05],
    [0.38, 0.24],[0.62, 0.24],
    [0.41, 0.52],[0.59, 0.58],
    [0.30, 0.40],[0.70, 0.40],
    [0.38, 0.68],[0.62, 0.76],
    [0.28, 0.52],[0.72, 0.52],
    [0.38, 0.82],[0.62, 0.92],
])

# Calf raise (on toes)
CALF_RAISE_UP = _pose([
    [0.50, 0.10],[0.50, 0.36],[0.50, 0.01],
    [0.37, 0.20],[0.63, 0.20],
    [0.40, 0.50],[0.60, 0.50],
    [0.30, 0.36],[0.70, 0.36],
    [0.40, 0.70],[0.60, 0.70],
    [0.28, 0.50],[0.72, 0.50],
    [0.40, 0.86],[0.60, 0.86],
])

# Side plank
SIDE_PLANK = _pose([
    [0.50, 0.40],[0.50, 0.52],[0.50, 0.32],
    [0.44, 0.36],[0.56, 0.44],
    [0.46, 0.58],[0.54, 0.58],
    [0.40, 0.48],[0.56, 0.32],
    [0.46, 0.64],[0.54, 0.68],
    [0.38, 0.58],[0.54, 0.24],
    [0.46, 0.76],[0.54, 0.80],
])

# Mountain climber (plank + knee drive)
MOUNTAIN_CLIMBER = _pose([
    [0.50, 0.35],[0.50, 0.44],[0.50, 0.27],
    [0.38, 0.40],[0.62, 0.40],
    [0.44, 0.48],[0.56, 0.50],
    [0.32, 0.50],[0.68, 0.50],
    [0.40, 0.55],[0.56, 0.65],
    [0.32, 0.58],[0.68, 0.58],
    [0.38, 0.62],[0.65, 0.80],
])

# Tricep dip (chair dip position)
TRICEP_DIP_DOWN = _pose([
    [0.50, 0.30],[0.50, 0.55],[0.50, 0.22],
    [0.34, 0.28],[0.66, 0.28],
    [0.42, 0.62],[0.58, 0.62],
    [0.30, 0.44],[0.70, 0.44],
    [0.40, 0.75],[0.60, 0.75],
    [0.28, 0.58],[0.72, 0.58],
    [0.40, 0.90],[0.60, 0.90],
])

# Glute kickback (on all fours, leg extended back)
GLUTE_KICKBACK = _pose([
    [0.50, 0.35],[0.50, 0.50],[0.50, 0.27],
    [0.38, 0.42],[0.62, 0.42],
    [0.44, 0.58],[0.56, 0.58],
    [0.32, 0.52],[0.68, 0.52],
    [0.44, 0.68],[0.56, 0.48],
    [0.32, 0.60],[0.68, 0.60],
    [0.44, 0.80],[0.60, 0.32],
])

# Bird dog (opposite arm and leg extended)
BIRD_DOG = _pose([
    [0.50, 0.35],[0.50, 0.50],[0.50, 0.27],
    [0.38, 0.42],[0.62, 0.42],
    [0.44, 0.58],[0.56, 0.58],
    [0.28, 0.38],[0.68, 0.52],
    [0.44, 0.68],[0.58, 0.48],
    [0.22, 0.35],[0.68, 0.60],
    [0.44, 0.80],[0.58, 0.32],
])

# Hamstring stretch (standing, forward fold)
HAMSTRING_STRETCH = _pose([
    [0.50, 0.42],[0.50, 0.58],[0.50, 0.34],
    [0.40, 0.50],[0.60, 0.50],
    [0.42, 0.62],[0.58, 0.62],
    [0.36, 0.60],[0.64, 0.60],
    [0.42, 0.72],[0.58, 0.72],
    [0.38, 0.72],[0.62, 0.72],
    [0.42, 0.88],[0.58, 0.88],
])

# Hip flexor stretch (kneeling lunge)
HIP_FLEXOR = _pose([
    [0.50, 0.20],[0.50, 0.46],[0.50, 0.11],
    [0.38, 0.30],[0.62, 0.30],
    [0.43, 0.52],[0.57, 0.62],
    [0.30, 0.44],[0.70, 0.44],
    [0.38, 0.70],[0.62, 0.80],
    [0.28, 0.56],[0.72, 0.56],
    [0.35, 0.88],[0.62, 0.92],
])

# Seated row (leaning back, arms pulled in)
SEATED_ROW_PULL = _pose([
    [0.50, 0.22],[0.50, 0.48],[0.50, 0.12],
    [0.37, 0.32],[0.63, 0.32],
    [0.42, 0.56],[0.58, 0.56],
    [0.30, 0.38],[0.70, 0.38],
    [0.40, 0.66],[0.60, 0.66],
    [0.32, 0.48],[0.68, 0.48],
    [0.40, 0.82],[0.60, 0.82],
])

# ─────────────────────────────────────────────────────────────────────────────
# Exercise registry
# ─────────────────────────────────────────────────────────────────────────────

EXERCISE_ANIMATIONS: Dict[str, List[np.ndarray]] = {
    # Strength — Lower Body
    "squat":                    [STAND, SQUAT_BOTTOM, STAND],
    "lunge":                    [STAND, LUNGE, STAND],
    "deadlift":                 [STAND, DEADLIFT_HINGE, STAND],
    "romanian deadlift":        [STAND, RDL, STAND],
    "step up":                  [STAND, STEP_UP, STAND],
    "calf raise":               [STAND, CALF_RAISE_UP, STAND],
    "glute kickback":           [PLANK, GLUTE_KICKBACK, PLANK],
    # Strength — Upper Body
    "push-up":                  [PUSHUP_TOP, PUSHUP_BOTTOM, PUSHUP_TOP],
    "overhead press":           [OVERHEAD_PRESS_DOWN, OVERHEAD_PRESS_UP, OVERHEAD_PRESS_DOWN],
    "lateral raise":            [STAND, LATERAL_RAISE, STAND],
    "bicep curl":               [STAND, BICEP_CURL_UP, STAND],
    "tricep dip":               [STAND, TRICEP_DIP_DOWN, STAND],
    # Core
    "plank":                    [PLANK],
    "side plank":               [SIDE_PLANK],
    "hip bridge":               [HIP_BRIDGE_DOWN, HIP_BRIDGE_UP, HIP_BRIDGE_DOWN],
    "mountain climber":         [PLANK, MOUNTAIN_CLIMBER, PLANK],
    "bird dog":                 [PLANK, BIRD_DOG, PLANK],
    # Rehabilitation
    "shoulder abduction":       [STAND, SHOULDER_ABDUCTION, STAND],
    "knee flexion":             [STAND, KNEE_FLEXION, STAND],
    "terminal knee extension":  [SQUAT_BOTTOM, STAND],
    "heel slide":               [HIP_BRIDGE_DOWN, KNEE_FLEXION, HIP_BRIDGE_DOWN],
    "seated row":               [SEATED_ROW_PULL, STAND],
    # Stretching
    "hamstring stretch":        [STAND, HAMSTRING_STRETCH],
    "hip flexor stretch":       [STAND, HIP_FLEXOR],
    # Posture / Gait
    "standing":                 [STAND],
    "walking gait":             [STAND, WALK_MID, STAND],
    # Medical Positioning
    "xray pa chest":            [STAND, XRAY_PA],
}

# Human-readable labels  (auto-filled for any key not listed here)
EXERCISE_LABELS = {
    "squat":                    "Squat",
    "lunge":                    "Lunge",
    "deadlift":                 "Deadlift",
    "romanian deadlift":        "Romanian Deadlift",
    "step up":                  "Step-Up",
    "calf raise":               "Calf Raise",
    "glute kickback":           "Glute Kickback",
    "push-up":                  "Push-Up",
    "overhead press":           "Overhead Press",
    "lateral raise":            "Lateral Raise",
    "bicep curl":               "Bicep Curl",
    "tricep dip":               "Tricep Dip",
    "plank":                    "Plank",
    "side plank":               "Side Plank",
    "hip bridge":               "Hip Bridge / Glute Bridge",
    "mountain climber":         "Mountain Climber",
    "bird dog":                 "Bird Dog",
    "shoulder abduction":       "Shoulder Abduction",
    "knee flexion":             "Knee Flexion",
    "terminal knee extension":  "Terminal Knee Extension",
    "heel slide":               "Heel Slide",
    "seated row":               "Seated Row",
    "hamstring stretch":        "Hamstring Stretch",
    "hip flexor stretch":       "Hip Flexor Stretch",
    "standing":                 "Standing Posture",
    "walking gait":             "Walking Gait",
    "xray pa chest":            "PA Chest X-Ray Position",
}

def get_label(key: str) -> str:
    """Return human-readable label; fall back to title-casing the key."""
    return EXERCISE_LABELS.get(key, key.replace("-", " ").title())


# ─────────────────────────────────────────────────────────────────────────────
# Interpolation
# ─────────────────────────────────────────────────────────────────────────────

def _interpolate(pose_a: np.ndarray, pose_b: np.ndarray, n_steps: int = 15) -> List[np.ndarray]:
    """Smooth interpolation between two poses using cosine easing."""
    frames = []
    for i in range(n_steps):
        t = i / (n_steps - 1)
        # Cosine ease-in-out
        t_eased = (1 - np.cos(t * np.pi)) / 2
        frames.append((1 - t_eased) * pose_a + t_eased * pose_b)
    return frames


def _build_frame_sequence(
    keyframes: List[np.ndarray], steps_per_transition: int = 18, hold_frames: int = 8
) -> List[np.ndarray]:
    """Build full frame sequence from keyframes with holds and transitions."""
    if len(keyframes) == 1:
        return [keyframes[0]] * (hold_frames * 3)

    sequence = []
    for i in range(len(keyframes) - 1):
        if i == 0:
            sequence.extend([keyframes[0]] * hold_frames)
        sequence.extend(_interpolate(keyframes[i], keyframes[i + 1], steps_per_transition))
        sequence.extend([keyframes[i + 1]] * hold_frames)
    return sequence


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _draw_skeleton_frame(
    ax,
    joints: np.ndarray,
    title: str = "",
    highlight_joints: List[int] = None,
    show_angles: Dict[str, float] = None,
    comparison_joints: np.ndarray = None,
):
    """Draw one skeleton frame on a matplotlib axis."""
    ax.clear()
    ax.set_facecolor("#0e1117")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.05, -0.05)   # y-axis flipped (image coords)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, color="white", fontsize=10, pad=4)

    # Ghost comparison skeleton (e.g. ideal pose overlay)
    if comparison_joints is not None:
        for (a, b) in SKELETON_EDGES:
            if comparison_joints[a].sum() > 0 and comparison_joints[b].sum() > 0:
                ax.plot(
                    [comparison_joints[a, 0], comparison_joints[b, 0]],
                    [comparison_joints[a, 1], comparison_joints[b, 1]],
                    color="#00cc66", lw=2, alpha=0.35, zorder=1,
                )
        for i, jt in enumerate(comparison_joints):
            if jt.sum() > 0:
                ax.scatter(jt[0], jt[1], s=40, color="#00cc66", alpha=0.3, zorder=2)

    # Skeleton edges
    for (a, b) in SKELETON_EDGES:
        if joints[a].sum() > 0 and joints[b].sum() > 0:
            color = EDGE_COLORS.get((a, b), "#888888")
            ax.plot(
                [joints[a, 0], joints[b, 0]],
                [joints[a, 1], joints[b, 1]],
                color=color, lw=3, alpha=0.9, zorder=3, solid_capstyle="round",
            )

    # Joint dots
    hl = set(highlight_joints or [])
    for i, jt in enumerate(joints):
        if jt.sum() == 0:
            continue
        base_color = JOINT_COLORS.get(i, "#888888")
        ring_color = "#ff4444" if i in hl else base_color
        size = 120 if i in hl else 70
        ax.scatter(jt[0], jt[1], s=size, color=ring_color,
                   edgecolors="white", linewidth=1, zorder=5)

    # Angle annotations
    if show_angles:
        for label, val in show_angles.items():
            ax.text(0.02, 0.98, f"{label}: {val:.0f}°",
                    transform=ax.transAxes, color="yellow",
                    fontsize=8, va="top")


def _draw_human_frame(
    ax,
    joints: np.ndarray,
    title: str = "",
):
    """
    Draw a realistic human body figure on a matplotlib axis using the
    JHMDB 15-joint skeleton as the underlying structure.

    Rendering layers (bottom to top):
      1. Shadow ellipse on ground
      2. Torso — filled trapezoid (shoulders → hips)
      3. Limbs — thick rounded tubes with shading
      4. Head — filled circle with face highlight
      5. Hands/feet — small filled ovals
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch, Circle
    from matplotlib.collections import LineCollection

    ax.clear()
    ax.set_facecolor("#1a1a2e")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(1.08, -0.08)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color="white", fontsize=10, pad=4)

    j = joints  # shorthand

    # ── Colour palette ─────────────────────────────────────────────────────
    SKIN      = "#d4956a"
    SKIN_DARK = "#b07040"
    SHIRT     = "#3a7bd5"
    SHIRT_DRK = "#2255aa"
    PANTS     = "#2c3e6f"
    PANTS_DRK = "#1a2a4a"
    SHOE      = "#222222"
    HAIR      = "#2c1810"

    def tube(ax, p1, p2, color, dark, lw=18, zorder=3):
        """Draw a limb as a thick rounded line segment."""
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=dark, lw=lw + 3, solid_capstyle="round",
                solid_joinstyle="round", zorder=zorder)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, lw=lw, solid_capstyle="round",
                solid_joinstyle="round", zorder=zorder + 1)

    def joint_circle(ax, pos, r, color, zorder=8):
        c = Circle(pos, r, color=color, zorder=zorder)
        ax.add_patch(c)

    # ── Ground shadow ──────────────────────────────────────────────────────
    # Find lowest y (highest value in image coords)
    ankle_y = max(j[13, 1], j[14, 1]) + 0.015
    ankle_x = (j[13, 0] + j[14, 0]) / 2
    shadow = Ellipse((ankle_x, ankle_y + 0.01), 0.18, 0.025,
                     color="#000000", alpha=0.25, zorder=1)
    ax.add_patch(shadow)

    # ── Torso — filled quadrilateral ───────────────────────────────────────
    rs, ls = j[3], j[4]   # shoulders
    rh, lh = j[5], j[6]   # hips
    if rs.sum() and ls.sum() and rh.sum() and lh.sum():
        torso_x = [rs[0], ls[0], lh[0], rh[0]]
        torso_y = [rs[1], ls[1], lh[1], rh[1]]
        ax.fill(torso_x, torso_y, color=SHIRT, zorder=4, alpha=0.95)
        # Torso outline / shading line
        mid_sh = (rs + ls) / 2
        mid_hi = (rh + lh) / 2
        ax.plot([mid_sh[0], mid_hi[0]], [mid_sh[1], mid_hi[1]],
                color=SHIRT_DRK, lw=1.5, alpha=0.5, zorder=5)

    # ── Legs ───────────────────────────────────────────────────────────────
    for hip, knee, ankle in [(j[5], j[9], j[13]), (j[6], j[10], j[14])]:
        if hip.sum() and knee.sum() and ankle.sum():
            tube(ax, hip, knee,   PANTS, PANTS_DRK, lw=16, zorder=2)
            tube(ax, knee, ankle, PANTS, PANTS_DRK, lw=14, zorder=2)
            # Shoe
            joint_circle(ax, ankle, 0.025, SHOE, zorder=6)

    # ── Arms ───────────────────────────────────────────────────────────────
    for sh, elb, wrist in [(j[3], j[7], j[11]), (j[4], j[8], j[12])]:
        if sh.sum() and elb.sum() and wrist.sum():
            tube(ax, sh, elb,   SHIRT, SHIRT_DRK, lw=13, zorder=3)
            tube(ax, elb, wrist, SKIN,  SKIN_DARK, lw=10, zorder=3)
            # Hand
            joint_circle(ax, wrist, 0.018, SKIN, zorder=7)

    # ── Knee caps ──────────────────────────────────────────────────────────
    for knee in [j[9], j[10]]:
        if knee.sum():
            joint_circle(ax, knee, 0.022, PANTS_DRK, zorder=6)

    # ── Head ───────────────────────────────────────────────────────────────
    neck = j[0]
    face = j[2]
    if neck.sum() and face.sum():
        head_center = face
        head_r = 0.065
        # Hair (back of head)
        hair = Circle(head_center, head_r + 0.006, color=HAIR, zorder=7)
        ax.add_patch(hair)
        # Face
        face_c = Circle(head_center, head_r, color=SKIN, zorder=8)
        ax.add_patch(face_c)
        # Highlight
        highlight = Circle(
            (head_center[0] - head_r * 0.25, head_center[1] - head_r * 0.3),
            head_r * 0.22, color="white", alpha=0.18, zorder=9)
        ax.add_patch(highlight)
        # Eyes (simple dots)
        for ex_off in [-0.018, 0.018]:
            eye = Circle(
                (head_center[0] + ex_off, head_center[1] + 0.008),
                0.008, color="#2c1c10", zorder=10)
            ax.add_patch(eye)
        # Neck tube
        if neck.sum():
            tube(ax, neck, head_center, SKIN, SKIN_DARK, lw=10, zorder=6)

    # ── Shoulder caps ──────────────────────────────────────────────────────
    for sh in [j[3], j[4]]:
        if sh.sum():
            joint_circle(ax, sh, 0.028, SHIRT_DRK, zorder=5)


def create_dual_exercise_gif(
    exercise_key: str,
    fps: int = 10,
    size: Tuple[int, int] = (820, 460),
) -> bytes:
    """
    Generate a side-by-side animated GIF:
      LEFT  — skeleton view (pose classification model)
      RIGHT — realistic human body view

    Args:
        exercise_key: key in EXERCISE_ANIMATIONS
        fps: frames per second
        size: total output pixel dimensions (width, height)
    Returns:
        GIF bytes
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter

    keyframes = EXERCISE_ANIMATIONS.get(exercise_key, [STAND])
    frames = _build_frame_sequence(keyframes, steps_per_transition=18, hold_frames=10)
    label  = get_label(exercise_key)
    phases = _phase_labels(exercise_key, len(frames))

    dpi = 100
    fig, (ax_sk, ax_hu) = plt.subplots(
        1, 2,
        figsize=(size[0] / dpi, size[1] / dpi),
        facecolor="#0e1117",
        gridspec_kw={"wspace": 0.04},
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02)

    # Column headers
    fig.text(0.26, 0.96, "Pose Skeleton (AI Model)", color="#7ec8e3",
             ha="center", fontsize=11, fontweight="bold")
    fig.text(0.74, 0.96, "Human Body View", color="#f5a623",
             ha="center", fontsize=11, fontweight="bold")

    def animate(i):
        phase = phases[i] if phases else ""
        title = f"{label}\n{phase}" if phase else label
        _draw_skeleton_frame(ax_sk, frames[i], title=title)
        _draw_human_frame(ax_hu, frames[i], title=title)

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                   interval=int(1000 / fps), blit=False)

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    try:
        ani.save(tmp.name, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def create_dual_custom_gif(
    keyframes: List[np.ndarray],
    label: str,
    fps: int = 8,
    size: Tuple[int, int] = (820, 460),
) -> bytes:
    """Same as create_dual_exercise_gif but accepts raw keyframe arrays (for LLM-generated poses)."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter

    frames = _build_frame_sequence(keyframes, steps_per_transition=18, hold_frames=10)
    n = len(frames)

    dpi = 100
    fig, (ax_sk, ax_hu) = plt.subplots(
        1, 2,
        figsize=(size[0] / dpi, size[1] / dpi),
        facecolor="#0e1117",
        gridspec_kw={"wspace": 0.04},
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.02)
    fig.text(0.26, 0.96, "Pose Skeleton (AI Model)", color="#7ec8e3",
             ha="center", fontsize=11, fontweight="bold")
    fig.text(0.74, 0.96, "Human Body View", color="#f5a623",
             ha="center", fontsize=11, fontweight="bold")

    def animate(i):
        _draw_skeleton_frame(ax_sk, frames[i], title=label)
        _draw_human_frame(ax_hu, frames[i], title=label)

    ani = animation.FuncAnimation(fig, animate, frames=n,
                                   interval=int(1000 / fps), blit=False)
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    try:
        ani.save(tmp.name, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def create_exercise_gif(
    exercise_key: str,
    fps: int = 12,
    size: Tuple[int, int] = (400, 450),
    comparison_pose: np.ndarray = None,
    title_prefix: str = "",
) -> bytes:
    """
    Generate an animated GIF for an exercise.

    Args:
        exercise_key: key in EXERCISE_ANIMATIONS (e.g. "squat")
        fps: frames per second
        size: output pixel size (width, height)
        comparison_pose: optional (15,2) pose to show as green ghost
        title_prefix: prepend to frame title

    Returns:
        GIF bytes (can be written to file or passed to st.image)
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter

    keyframes = EXERCISE_ANIMATIONS.get(exercise_key, [STAND])
    frames = _build_frame_sequence(keyframes, steps_per_transition=18, hold_frames=10)

    label = EXERCISE_LABELS.get(exercise_key, exercise_key.title())
    phase_labels = _phase_labels(exercise_key, len(frames))

    dpi = 100
    fig, ax = plt.subplots(1, 1,
                            figsize=(size[0] / dpi, size[1] / dpi),
                            facecolor="#0e1117")
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)

    def animate(i):
        title = f"{title_prefix}{label}"
        if phase_labels:
            title += f"\n{phase_labels[i]}"
        _draw_skeleton_frame(
            ax, frames[i],
            title=title,
            comparison_joints=comparison_pose,
        )

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                   interval=int(1000 / fps), blit=False)

    # PillowWriter requires a real file path on some platforms
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    try:
        ani.save(tmp.name, writer=PillowWriter(fps=fps), dpi=dpi)
        plt.close(fig)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def _phase_labels(exercise_key: str, n_frames: int) -> List[str]:
    """Generate per-frame phase labels shown in the animation title."""
    label_map = {
        "squat":                    ["Stand tall", "Descend — knees over toes", "Hold 90°", "Drive up"],
        "lunge":                    ["Stand tall", "Step forward", "Lower — knee above floor", "Return"],
        "deadlift":                 ["Stand tall", "Hip hinge — flat back", "Grip & brace", "Drive hips forward"],
        "romanian deadlift":        ["Stand tall", "Hip hinge — slight knee bend", "Feel stretch", "Return"],
        "step up":                  ["Stand behind step", "Step up — drive knee", "Stand tall", "Step down"],
        "calf raise":               ["Feet flat", "Rise onto toes", "Hold top", "Lower slowly"],
        "glute kickback":           ["All fours", "Kick leg back — straight", "Squeeze glute", "Lower"],
        "push-up":                  ["Plank", "Lower — elbows 45°", "Hold", "Push up"],
        "overhead press":           ["Bar at shoulders", "Press overhead", "Lockout — elbows soft", "Lower"],
        "lateral raise":            ["Arms at sides", "Raise to shoulder height", "Brief hold", "Lower slowly"],
        "bicep curl":               ["Arms straight", "Curl up — elbow fixed", "Squeeze top", "Lower"],
        "tricep dip":               ["Arms straight", "Lower — elbows behind", "Hold 90°", "Press up"],
        "plank":                    ["Forearms down", "Hips level — brace core", "Hold — breathe"],
        "side plank":               ["Side on — forearm down", "Lift hips up", "Hold — shoulders stacked"],
        "hip bridge":               ["Lie flat — knees bent", "Drive hips up", "Squeeze glutes — hold", "Lower"],
        "mountain climber":         ["Plank", "Drive knee to chest", "Alternate quickly"],
        "bird dog":                 ["All fours", "Extend opposite arm + leg", "Hold — flat back", "Return"],
        "shoulder abduction":       ["Arm at side", "Raise laterally — elbow straight", "90° hold", "Lower slowly"],
        "knee flexion":             ["Standing", "Raise knee to hip height", "Hold", "Lower"],
        "terminal knee extension":  ["Slight squat", "Drive knee straight", "Hold", "Release"],
        "heel slide":               ["Lie flat", "Slide heel toward glute", "Hold bend", "Slide back"],
        "seated row":               ["Lean forward", "Pull handles — elbows back", "Squeeze shoulder blades", "Return"],
        "hamstring stretch":        ["Stand tall", "Hinge forward — flat back", "Feel stretch in hamstrings"],
        "hip flexor stretch":       ["Stand tall", "Step forward into lunge", "Press hip down", "Feel front hip stretch"],
        "standing":                 ["Ears over shoulders over hips", "Soft knee bend", "Weight through heels"],
        "walking gait":             ["Heel strike", "Mid-stance weight shift", "Toe-off swing"],
        "xray pa chest":            ["Natural stand", "Roll shoulders forward", "Chin slightly raised"],
    }
    phases = label_map.get(exercise_key, ["Start", "Execute", "Return"])
    if not phases:
        return [""] * n_frames
    step = max(1, n_frames // len(phases))
    return [phases[min(i // step, len(phases) - 1)] for i in range(n_frames)]


def create_comparison_gif(
    current_joints: np.ndarray,
    ideal_joints: np.ndarray,
    title: str = "Current vs Ideal",
    fps: int = 3,
) -> bytes:
    """
    Side-by-side static comparison: current pose (gray) vs ideal (green).
    Returns GIF bytes.
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.animation import PillowWriter

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), facecolor="#0e1117")
    fig.suptitle(title, color="white", fontsize=11, fontweight="bold")

    def animate(i):
        # Alternate between showing current only and overlay
        _draw_skeleton_frame(axes[0], current_joints, title="Current Pose")
        _draw_skeleton_frame(axes[1], ideal_joints,   title="Target Pose",
                              comparison_joints=current_joints)

    ani = animation.FuncAnimation(fig, animate, frames=2, interval=1500, blit=False)
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    tmp.close()
    try:
        ani.save(tmp.name, writer=PillowWriter(fps=1), dpi=100)
        plt.close(fig)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def render_static_skeleton(joints: np.ndarray, title: str = "", highlight: List[int] = None):
    """Return a matplotlib Figure of a single skeleton frame (for st.pyplot)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4.5), facecolor="#0e1117")
    fig.subplots_adjust(left=0, right=1, top=0.92, bottom=0)
    _draw_skeleton_frame(ax, joints, title=title, highlight_joints=highlight)
    return fig


def joints_from_mediapipe(landmarks) -> np.ndarray:
    """Convert MediaPipe pose landmarks to JHMDB (15,2) array."""
    MP2JHMDB = {0:2,11:3,12:4,23:5,24:6,13:7,14:8,25:9,26:10,15:11,16:12,27:13,28:14,1:0}
    j = np.zeros((15, 2), dtype=np.float32)
    for mp_i, jh_i in MP2JHMDB.items():
        if mp_i < len(landmarks):
            j[jh_i] = [landmarks[mp_i].x, landmarks[mp_i].y]
    if j[3].sum() and j[4].sum():
        j[1] = (j[3] + j[4]) / 2
    if j[0].sum():
        j[2] = j[0] + np.array([0, -0.05])
    return j
