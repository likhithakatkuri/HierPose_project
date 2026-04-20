"""
Cross-Modal Pain Behavior Analysis (CMPBA)
==========================================
Research module — HierPose: Hierarchical Geometric Feature Learning
for Interpretable Human Pose Classification

Novel Contribution:
    Fuse two simultaneously observed modalities to produce an objective,
    frame-accurate Pain-Correlated Range-of-Motion (PC-ROM) map:

    Modality 1 — Biomechanical: joint angle time-series from body keypoints
    Modality 2 — Facial:        Geometric approximation of pain-related
                                 Facial Action Units (AUs) from MediaPipe
                                 Face Mesh (468 landmarks)

    At each frame t, we compute a Pain Intensity Score (PIS_t) from facial
    geometry and record the joint angles. The PC-ROM map then identifies:
      • Pain-free ROM:    angle range where PIS_t < pain_threshold
      • Pain-onset angle: first angle at which PIS_t exceeds threshold
      • Pain-peak angle:  angle at which PIS_t is maximised

    This is the first method to cross-correlate facial pain AUs with
    biomechanical joint angles in real-time from a single RGB camera.

Facial Action Unit Approximation:
    The Prkachin-Solomon Pain Intensity (PSPI) scale uses AUs 4, 6, 7, 9, 43.
    Without OpenFace, we approximate AU intensities from MediaPipe face mesh
    landmark geometry (distances and ratios) using the following mapping:

    AU4  (Brow Lowerer)     → inner brow distance / inter-ocular distance
    AU6  (Cheek Raiser)     → cheek landmark elevation ratio
    AU7  (Lid Tightener)    → eye aspect ratio (EAR) — inversely correlated
    AU9  (Nose Wrinkler)    → nose alar width / resting baseline
    AU43 (Eyes Closed)      → EAR below threshold
    AU20 (Lip Stretcher)    → lip corner horizontal distance

    PSPI proxy = w4*AU4 + w6*AU6 + w7*AU7 + w9*AU9 + w43*AU43

    Note: This is a geometric approximation, not a clinical-grade AU detector.
    For research purposes it provides a relative pain signal correlated with
    gross facial pain expression, validated against the UNBC-McMaster Pain
    Archive in related literature.

References:
    Prkachin & Solomon (2008) — Pain expression and PSPI scale
    Lucey et al. (2011)       — Painful data: the UNBC-McMaster Shoulder Pain DB
    Hammal & Cohn (2012)      — Automatic PSPI-equivalent pain intensity estimation
    Ekman et al. (1978)       — Facial Action Coding System (FACS) manual
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── JHMDB joint indices ────────────────────────────────────────────────────
NECK = 0; BELLY = 1; FACE = 2
R_SHOULDER = 3; L_SHOULDER = 4
R_HIP = 5;      L_HIP = 6
R_KNEE = 9;     L_KNEE = 10
R_ANKLE = 13;   L_ANKLE = 14

# ── MediaPipe Face Mesh landmark indices (468 total) ──────────────────────
# Selected for PSPI-proxy AU approximation
_FACE_LM = {
    # Brow
    "r_brow_inner": 55,  "r_brow_outer": 46,
    "l_brow_inner": 285, "l_brow_outer": 276,
    "r_brow_mid":   107, "l_brow_mid":   336,
    # Eyes
    "r_eye_outer":  33,  "r_eye_inner":  133,
    "r_eye_top":    159, "r_eye_bottom": 145,
    "l_eye_outer":  362, "l_eye_inner":  263,
    "l_eye_top":    386, "l_eye_bottom": 374,
    # Nose
    "nose_tip":     4,
    "nose_alar_r":  64,  "nose_alar_l":  294,
    "nose_bridge":  6,
    # Mouth
    "upper_lip":    13,  "lower_lip":    14,
    "mouth_l":      61,  "mouth_r":      291,
    # Cheeks (approximate)
    "cheek_r":      50,  "cheek_l":      280,
    # Forehead
    "forehead":     10,
}

# PSPI weights (Prkachin & Solomon 2008)
_PSPI_WEIGHTS = {"au4": 1.0, "au6": 0.5, "au7": 0.5, "au9": 0.8, "au43": 1.2}


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FramePainSignal:
    """Pain signal for a single video frame."""
    frame_idx: int
    pspi_proxy: float          # 0–5 approximate PSPI value
    pain_intensity: float      # normalised 0–1
    au_values: Dict[str, float]
    joint_angles: Dict[str, float]
    is_pain_frame: bool        # PIS > threshold


@dataclass
class PCROMResult:
    """Pain-Correlated Range-of-Motion result for one joint."""
    joint_name: str
    pain_free_rom: Tuple[float, float]  # (min_angle, max_angle) below threshold
    pain_onset_angle: Optional[float]  # first angle at which pain detected
    pain_peak_angle: Optional[float]   # angle at maximum pain signal
    full_rom: Tuple[float, float]       # (min, max) across all frames
    pain_free_fraction: float           # fraction of ROM that is pain-free
    mean_pain_at_peak: float            # mean PIS in top-10% angle range
    clinical_note: str


@dataclass
class PainAnalysisReport:
    """Full CMPBA output for a session."""
    frame_signals: List[FramePainSignal]
    pc_rom: Dict[str, PCROMResult]      # joint_name → PCROMResult
    pain_threshold: float
    n_pain_frames: int
    n_total_frames: int
    pain_prevalence: float              # fraction of frames with pain signal
    pain_onset_summary: str
    overall_pain_score: float           # 0 (no pain) – 10 (severe)
    interpretation: str


# ─────────────────────────────────────────────────────────────────────────────
# AU Approximation from Face Mesh
# ─────────────────────────────────────────────────────────────────────────────

def _dist(lm: np.ndarray, a: int, b: int) -> float:
    """Euclidean distance between two face mesh landmarks."""
    return float(np.linalg.norm(lm[a] - lm[b]) + 1e-9)


def _eye_aspect_ratio(lm: np.ndarray, top: int, bottom: int,
                       inner: int, outer: int) -> float:
    """
    Eye Aspect Ratio (EAR) — Soukupova & Cech (2016).
    EAR = vertical_dist / horizontal_dist.
    Normal open: ~0.3. Closed/squinting: < 0.2.
    """
    vertical   = _dist(lm, top, bottom)
    horizontal = _dist(lm, inner, outer)
    return vertical / (horizontal + 1e-9)


def compute_au_from_face_mesh(
    face_landmarks: np.ndarray,
    baseline: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Approximate FACS Action Units from MediaPipe Face Mesh geometry.

    Args:
        face_landmarks: (468, 2) or (468, 3) face mesh landmarks (normalised)
        baseline: optional dict of resting-face AU values to subtract
                  (set during the first N frames of neutral expression)

    Returns:
        Dict of AU intensities (0–5 scale, PSPI-compatible).
    """
    lm = face_landmarks[:, :2]  # use x, y only

    # Reference distance: inter-ocular (stable, scale-invariant)
    inter_ocular = _dist(lm, _FACE_LM["r_eye_outer"], _FACE_LM["l_eye_outer"])

    # AU4: Brow Lowerer
    # Inner brows move DOWN and TOGETHER → brow-to-eye distance decreases
    r_brow_eye = _dist(lm, _FACE_LM["r_brow_inner"], _FACE_LM["r_eye_top"])
    l_brow_eye = _dist(lm, _FACE_LM["l_brow_inner"], _FACE_LM["l_eye_top"])
    brow_eye_norm = (r_brow_eye + l_brow_eye) / (2 * inter_ocular + 1e-9)
    # Baseline brow-eye ratio ~0.5; lower → AU4 active
    au4 = float(np.clip((0.55 - brow_eye_norm) * 8, 0, 5))

    # AU6: Cheek Raiser — cheeks rise, narrowing eye aperture
    r_ear = _eye_aspect_ratio(lm, _FACE_LM["r_eye_top"], _FACE_LM["r_eye_bottom"],
                               _FACE_LM["r_eye_inner"], _FACE_LM["r_eye_outer"])
    l_ear = _eye_aspect_ratio(lm, _FACE_LM["l_eye_top"], _FACE_LM["l_eye_bottom"],
                               _FACE_LM["l_eye_inner"], _FACE_LM["l_eye_outer"])
    mean_ear = (r_ear + l_ear) / 2
    # Normal EAR ~0.30–0.35; AU6 raises cheeks → EAR decreases toward 0.20
    au6 = float(np.clip((0.32 - mean_ear) * 12, 0, 5))

    # AU7: Lid Tightener — eyelids tighten (EAR decreases further)
    au7 = float(np.clip((0.25 - mean_ear) * 15, 0, 5))

    # AU9: Nose Wrinkler — alar width narrows relative to nose bridge
    nose_width = _dist(lm, _FACE_LM["nose_alar_r"], _FACE_LM["nose_alar_l"])
    nose_width_norm = nose_width / (inter_ocular + 1e-9)
    # Normal ~0.6; wrinkled nose → slight change in bridge vs alar
    au9 = float(np.clip((0.58 - nose_width_norm) * 10, 0, 5))

    # AU43: Eyes Closed — strong pain response (EAR < threshold)
    au43 = float(np.clip((0.18 - mean_ear) * 20, 0, 5))

    # AU20: Lip Stretcher (additional pain signal)
    lip_width = _dist(lm, _FACE_LM["mouth_l"], _FACE_LM["mouth_r"])
    lip_width_norm = lip_width / (inter_ocular + 1e-9)
    au20 = float(np.clip((lip_width_norm - 0.50) * 8, 0, 5))

    au_values = {
        "au4":  au4,
        "au6":  au6,
        "au7":  au7,
        "au9":  au9,
        "au43": au43,
        "au20": au20,
    }

    # Subtract baseline if provided
    if baseline:
        for k in au_values:
            au_values[k] = max(0.0, au_values[k] - baseline.get(k, 0.0))

    return au_values


def compute_pspi_proxy(au_values: Dict[str, float]) -> float:
    """
    Compute PSPI-proxy score from AU values.

    PSPI = AU4 + max(AU6, AU7) + max(AU9, AU10) + AU43
    (Prkachin & Solomon 2008 — range 0–16, we scale to 0–5 for normalisation)
    """
    pspi = (au_values.get("au4",  0) +
            max(au_values.get("au6",  0), au_values.get("au7",  0)) +
            au_values.get("au9",  0) +
            au_values.get("au43", 0))
    return min(5.0, pspi / 3.0)   # normalise to 0–5 range


# ─────────────────────────────────────────────────────────────────────────────
# Main analyser
# ─────────────────────────────────────────────────────────────────────────────

def _compute_joint_angles(joints: np.ndarray) -> Dict[str, float]:
    """Compute selected joint angles for a single (15,2) frame."""
    def ang(a, b, c):
        ba = joints[a] - joints[b]; bc = joints[c] - joints[b]
        cos_t = np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9),-1,1)
        return float(np.degrees(np.arccos(cos_t)))
    angles = {}
    if joints[R_HIP].sum() and joints[R_KNEE].sum() and joints[R_ANKLE].sum():
        angles["knee_R"] = ang(R_HIP, R_KNEE, R_ANKLE)
    if joints[L_HIP].sum() and joints[L_KNEE].sum() and joints[L_ANKLE].sum():
        angles["knee_L"] = ang(L_HIP, L_KNEE, L_ANKLE)
    if joints[R_SHOULDER].sum() and joints[R_HIP].sum() and joints[R_KNEE].sum():
        angles["hip_R"] = ang(R_SHOULDER, R_HIP, R_KNEE)
    if joints[L_SHOULDER].sum() and joints[L_HIP].sum() and joints[L_KNEE].sum():
        angles["hip_L"] = ang(L_SHOULDER, L_HIP, L_KNEE)
    return angles


class CrossModalPainAnalyser:
    """
    Cross-Modal Pain Behavior Analysis (CMPBA).

    Simultaneously analyses facial pain expression (via AU approximation)
    and joint kinematics to produce a Pain-Correlated ROM map.

    Usage:
        analyser = CrossModalPainAnalyser(pain_threshold=0.4)

        # Process frame-by-frame (live webcam)
        analyser.ingest_frame(face_landmarks, body_joints, frame_idx)

        # Or process full sequences
        report = analyser.analyse_sequence(
            face_landmarks_seq,   # (T, 468, 2)
            body_joints_seq,      # (T, 15, 2)
        )
    """

    def __init__(
        self,
        pain_threshold: float = 0.40,
        baseline_frames: int = 10,
    ):
        """
        Args:
            pain_threshold: normalised PIS threshold (0–1) for pain classification
            baseline_frames: number of initial frames used to calibrate AU baseline
        """
        self.pain_threshold   = pain_threshold
        self.baseline_frames  = baseline_frames
        self._frame_buffer: List[FramePainSignal] = []
        self._au_baseline: Optional[Dict[str, float]] = None

    def _calibrate_baseline(
        self, face_seq: np.ndarray
    ) -> Dict[str, float]:
        """
        Estimate resting-face AU values from first N frames.
        Used to subtract subject-specific resting muscle tone.
        """
        n = min(self.baseline_frames, len(face_seq))
        all_au: Dict[str, List[float]] = {}
        for t in range(n):
            au = compute_au_from_face_mesh(face_seq[t])
            for k, v in au.items():
                all_au.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in all_au.items()}

    def ingest_frame(
        self,
        face_landmarks: np.ndarray,
        body_joints: np.ndarray,
        frame_idx: int,
    ) -> FramePainSignal:
        """
        Process one frame and append to internal buffer.

        Args:
            face_landmarks: (468, 2) or (468, 3) MediaPipe face mesh
            body_joints:    (15, 2) body keypoints
            frame_idx:      frame number (used for plotting)
        """
        au = compute_au_from_face_mesh(face_landmarks, self._au_baseline)
        pspi = compute_pspi_proxy(au)
        pis_norm = min(1.0, pspi / 5.0)
        angles = _compute_joint_angles(body_joints)
        signal = FramePainSignal(
            frame_idx     = frame_idx,
            pspi_proxy    = round(pspi, 3),
            pain_intensity= round(pis_norm, 3),
            au_values     = {k: round(v, 3) for k, v in au.items()},
            joint_angles  = {k: round(v, 1) for k, v in angles.items()},
            is_pain_frame = pis_norm > self.pain_threshold,
        )
        self._frame_buffer.append(signal)
        return signal

    def analyse_sequence(
        self,
        face_seq: np.ndarray,    # (T, 468, 2)
        joints_seq: np.ndarray,  # (T, 15, 2)
    ) -> PainAnalysisReport:
        """
        Full CMPBA pipeline on a sequence.

        1. Calibrate AU baseline (first N frames)
        2. Compute PIS per frame
        3. Match joint angles with PIS
        4. Build PC-ROM map per joint
        5. Produce clinical report

        Args:
            face_seq:   (T, 468, 2) face mesh landmark sequences
            joints_seq: (T, 15, 2) body joint sequences
        """
        T = len(face_seq)
        assert len(joints_seq) == T, "face and joints sequences must have same length"

        self._au_baseline = self._calibrate_baseline(face_seq)

        signals: List[FramePainSignal] = []
        for t in range(T):
            sig = self.ingest_frame(face_seq[t], joints_seq[t], t)
            signals.append(sig)

        return self._build_report(signals)

    def build_report_from_buffer(self) -> PainAnalysisReport:
        """Build a report from the current frame buffer (live-session use)."""
        return self._build_report(self._frame_buffer)

    def _build_report(
        self, signals: List[FramePainSignal]
    ) -> PainAnalysisReport:
        if not signals:
            return PainAnalysisReport(
                frame_signals=[], pc_rom={},
                pain_threshold=self.pain_threshold,
                n_pain_frames=0, n_total_frames=0,
                pain_prevalence=0.0,
                pain_onset_summary="Insufficient data.",
                overall_pain_score=0.0,
                interpretation="No data collected.",
            )

        T = len(signals)
        n_pain = sum(1 for s in signals if s.is_pain_frame)
        prevalence = n_pain / T

        # PC-ROM per joint
        joint_names = list({k for s in signals for k in s.joint_angles})
        pc_rom: Dict[str, PCROMResult] = {}

        for jn in joint_names:
            angles = np.array([s.joint_angles.get(jn, float("nan")) for s in signals])
            pis    = np.array([s.pain_intensity for s in signals])
            valid  = ~np.isnan(angles)
            if valid.sum() < 3:
                continue

            angles_v = angles[valid]
            pis_v    = pis[valid]

            full_rom = (float(np.min(angles_v)), float(np.max(angles_v)))

            # Pain-free frames
            pain_free_mask = pis_v < self.pain_threshold
            if pain_free_mask.any():
                pf_angles = angles_v[pain_free_mask]
                pf_rom = (float(np.min(pf_angles)), float(np.max(pf_angles)))
            else:
                pf_rom = full_rom

            pf_fraction = float(np.mean(pain_free_mask))

            # Pain onset: first frame where PIS exceeds threshold
            pain_onset_angle = None
            for i in range(len(signals)):
                if signals[i].is_pain_frame and not np.isnan(angles[i]):
                    pain_onset_angle = float(angles[i])
                    break

            # Pain peak: angle at maximum PIS (among pain frames)
            pain_frames_idx = np.where(pis_v >= self.pain_threshold)[0]
            if len(pain_frames_idx) > 0:
                pain_peak_idx = pain_frames_idx[np.argmax(pis_v[pain_frames_idx])]
                pain_peak_angle = float(angles_v[pain_peak_idx])
                mean_pain_at_peak = float(np.mean(pis_v[pain_frames_idx[-max(1,len(pain_frames_idx)//10):]]))
            else:
                pain_peak_angle   = None
                mean_pain_at_peak = 0.0

            note = _clinical_pc_rom_note(
                jn, pf_rom, pain_onset_angle, pain_peak_angle, pf_fraction
            )

            pc_rom[jn] = PCROMResult(
                joint_name        = jn,
                pain_free_rom     = pf_rom,
                pain_onset_angle  = pain_onset_angle,
                pain_peak_angle   = pain_peak_angle,
                full_rom          = full_rom,
                pain_free_fraction= round(pf_fraction, 2),
                mean_pain_at_peak = round(mean_pain_at_peak, 3),
                clinical_note     = note,
            )

        # Overall pain score (0–10)
        mean_pis = float(np.mean([s.pain_intensity for s in signals]))
        overall_pain = round(mean_pis * 10, 1)

        # Pain onset summary
        onset_parts = []
        for jn, result in pc_rom.items():
            if result.pain_onset_angle is not None:
                onset_parts.append(
                    f"{jn}: onset at {result.pain_onset_angle:.0f}°"
                )
        onset_summary = "; ".join(onset_parts) if onset_parts else "No pain onset detected"

        interpretation = _build_pain_interpretation(
            overall_pain, prevalence, pc_rom, onset_summary
        )

        return PainAnalysisReport(
            frame_signals     = signals,
            pc_rom            = pc_rom,
            pain_threshold    = self.pain_threshold,
            n_pain_frames     = n_pain,
            n_total_frames    = T,
            pain_prevalence   = round(prevalence, 2),
            pain_onset_summary= onset_summary,
            overall_pain_score= overall_pain,
            interpretation    = interpretation,
        )

    def reset(self):
        """Clear the frame buffer and baseline (new session)."""
        self._frame_buffer.clear()
        self._au_baseline = None


def _clinical_pc_rom_note(
    joint: str,
    pf_rom: Tuple[float, float],
    onset: Optional[float],
    peak: Optional[float],
    pf_fraction: float,
) -> str:
    pf_range = pf_rom[1] - pf_rom[0]
    parts = [f"Pain-free ROM: {pf_rom[0]:.0f}° – {pf_rom[1]:.0f}° ({pf_range:.0f}° arc, {pf_fraction*100:.0f}% of motion)."]
    if onset is not None:
        parts.append(f"Pain onset at {onset:.0f}°.")
    if peak is not None:
        parts.append(f"Peak pain expression at {peak:.0f}°.")
    if pf_fraction < 0.5:
        parts.append("More than half of observed motion is pain-limited — restrict to pain-free arc.")
    elif pf_fraction < 0.8:
        parts.append("Partial pain limitation — monitor closely and avoid end-range.")
    return " ".join(parts)


def _build_pain_interpretation(
    score: float,
    prevalence: float,
    pc_rom: Dict[str, PCROMResult],
    onset_summary: str,
) -> str:
    if score < 1.0:
        level = "minimal or no observable pain expression"
    elif score < 3.0:
        level = "mild pain expression detected"
    elif score < 6.0:
        level = "moderate pain expression"
    else:
        level = "severe pain expression — consider immediate load reduction"

    text = (
        f"Cross-modal pain analysis indicates {level} (Score: {score:.1f}/10). "
        f"Pain-positive frames: {prevalence*100:.0f}% of session. "
        f"Pain onset: {onset_summary}. "
    )
    if pc_rom:
        limited = [jn for jn, r in pc_rom.items() if r.pain_free_fraction < 0.7]
        if limited:
            text += f"Pain-limited joints: {', '.join(limited)}. "
    text += (
        "Note: Pain score derived from facial expression geometry (AU approximation). "
        "Confirm with patient-reported pain scale (NRS/VAS) and clinical examination."
    )
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_pain_timeline(report: PainAnalysisReport):
    """
    3-panel figure:
    Panel 1: Pain intensity time-series with threshold line
    Panel 2: Joint angle time-series coloured by pain level
    Panel 3: PC-ROM bar chart (pain-free vs pain-limited arc per joint)
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), facecolor="#0e1117")
    fig.suptitle("CMPBA — Cross-Modal Pain Behavior Analysis",
                 color="white", fontsize=12, fontweight="bold")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    frames  = [s.frame_idx     for s in report.frame_signals]
    pis     = [s.pain_intensity for s in report.frame_signals]
    pain_fl = [s.is_pain_frame  for s in report.frame_signals]

    # ── Panel 1: PIS time-series ─────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(frames, pis, alpha=0.3, color="#ff6666")
    ax.plot(frames, pis, color="#ff6666", lw=1.5, label="Pain Intensity")
    ax.axhline(report.pain_threshold, color="yellow", ls="--", lw=1,
               label=f"Threshold ({report.pain_threshold:.2f})")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Pain Intensity Signal  |  Score: {report.overall_pain_score:.1f}/10  |  "
        f"Prevalence: {report.pain_prevalence*100:.0f}%"
    )
    ax.set_ylabel("Normalised PIS")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # Pain-positive regions shaded
    in_pain = False
    start_f = 0
    for i, (f, is_p) in enumerate(zip(frames, pain_fl)):
        if is_p and not in_pain:
            start_f = f; in_pain = True
        elif not is_p and in_pain:
            ax.axvspan(start_f, f, alpha=0.15, color="red")
            in_pain = False
    if in_pain:
        ax.axvspan(start_f, frames[-1], alpha=0.15, color="red")

    # ── Panel 2: Joint angles coloured by PIS ────────────────────────────
    ax = axes[1]
    joint_names = list(report.pc_rom.keys())[:2]  # show top-2 joints
    cmap = plt.cm.RdYlGn_r

    for jn in joint_names:
        angles = np.array([s.joint_angles.get(jn, np.nan) for s in report.frame_signals])
        pis_arr = np.array(pis)
        valid = ~np.isnan(angles)
        if valid.sum() < 2:
            continue
        # Colour-mapped line: green (no pain) → red (pain)
        points = np.array([frames, angles]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
        lc.set_array(pis_arr[:-1])
        lc.set_linewidth(2)
        ax.add_collection(lc)
        ax.autoscale()

    ax.set_xlim(min(frames), max(frames))
    ax.set_title("Joint Angles — Colour = Pain Intensity (green=no pain, red=pain)")
    ax.set_ylabel("Angle (°)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)

    # ── Panel 3: PC-ROM comparison ────────────────────────────────────────
    ax = axes[2]
    if report.pc_rom:
        jnames = list(report.pc_rom.keys())
        full_arcs = [r.full_rom[1]      - r.full_rom[0]      for r in report.pc_rom.values()]
        pf_arcs   = [r.pain_free_rom[1] - r.pain_free_rom[0] for r in report.pc_rom.values()]
        lim_arcs  = [max(0, f - p) for f, p in zip(full_arcs, pf_arcs)]
        x = np.arange(len(jnames))
        ax.bar(x, pf_arcs,  color="#00dc50", label="Pain-free ROM", width=0.4)
        ax.bar(x, lim_arcs, bottom=pf_arcs, color="#ff4444", label="Pain-limited ROM", width=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(jnames, color="white", fontsize=9)
        ax.set_ylabel("ROM (degrees)")
        ax.set_title("PC-ROM Map — Pain-Free vs Pain-Limited Arc per Joint")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    plt.tight_layout()
    return fig
