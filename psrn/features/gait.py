"""
Hierarchical Gait Decomposition (HGD)
======================================
Research module — HierPose: Hierarchical Geometric Feature Learning
for Interpretable Human Pose Classification

Novel Contribution:
    A 3-level hierarchical decomposition of gait kinematics from 2D body
    keypoints, enabling clinical-grade gait analysis without force plates
    or motion-capture laboratories.

Hierarchy Levels:
    L1 — Joint-level  : Per-joint angle time-series (15 joints × T frames).
                        Geometric angles computed from keypoint triplets.
    L2 — Segment-level: Gait-cycle events (heel-strike, toe-off), phase
                        durations, step symmetry index, cadence.
    L3 — Whole-body   : Spatial-temporal parameters, Gait Deviation Index
                        (GDI) proxy, bilateral waveform correlation,
                        trunk sway, normative z-scores.

Alignment with Thesis:
    Gait cycles provide the temporal-kinematic context (L2/L3) built on
    top of frame-level joint geometry (L1) from HierarchicalFeatureExtractor.
    This two-stream design — static pose + dynamic gait — is the core
    architectural contribution of the HierPose framework.

References:
    Baker (2006)                 — Gait analysis methods in rehabilitation
    Schwartz & Rozumalski (2008) — Gait Deviation Index definition
    Robinson et al. (1987)       — Symmetry Index for bilateral comparison
    Pijnappels et al. (2001)     — Heel-strike detection from ankle kinematics
    Winter (2009)                — Biomechanics and Motor Control of Human Movement
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── JHMDB joint indices (15-joint skeleton) ────────────────────────────────
NECK = 0; BELLY = 1; FACE = 2
R_SHOULDER = 3; L_SHOULDER = 4
R_HIP = 5;      L_HIP = 6
R_ELBOW = 7;    L_ELBOW = 8
R_KNEE = 9;     L_KNEE = 10
R_WRIST = 11;   L_WRIST = 12
R_ANKLE = 13;   L_ANKLE = 14

# ── Normative gait parameters (Winter 2009, healthy adults 18-65) ──────────
# (mean, std) for each parameter — used to compute z-scores
NORMATIVE = {
    "cadence_spm":            (110.0, 12.0),   # steps per minute
    "stance_ratio":           (0.60,  0.03),   # fraction of cycle
    "swing_ratio":            (0.40,  0.03),   # fraction of cycle
    "step_symmetry_index":    (2.0,   2.0),    # percent (0 = perfect)
    "trunk_sway_range_deg":   (4.0,   2.0),    # degrees
    "knee_flexion_at_strike": (5.0,   4.0),    # degrees at heel-strike
    "peak_knee_flexion_swing":(65.0,  7.0),    # degrees during swing
    "hip_flexion_range":      (40.0,  5.0),    # full hip ROM during cycle
}


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GaitEvent:
    """Single gait event (heel-strike or toe-off)."""
    frame_idx: int
    side: str          # "right" | "left"
    event_type: str    # "heel_strike" | "toe_off"
    confidence: float  # 0–1


@dataclass
class GaitCycle:
    """One complete gait cycle (ipsilateral heel-strike to heel-strike)."""
    side: str
    start_frame: int
    end_frame: int
    stance_end_frame: int          # toe-off frame
    duration_frames: int
    stance_frames: int
    swing_frames: int

    @property
    def stance_ratio(self) -> float:
        return self.stance_frames / max(self.duration_frames, 1)

    @property
    def swing_ratio(self) -> float:
        return self.swing_frames / max(self.duration_frames, 1)


@dataclass
class GaitReport:
    """
    Full hierarchical gait analysis report.

    Produced by HierarchicalGaitFeatureExtractor.analyse().
    Used directly in the Gait Lab UI and PDF clinical reports.
    """

    # ── L1: Joint angle profiles ──────────────────────────────────────────
    angle_profiles: Dict[str, np.ndarray]   # joint_name → (T,) angle series

    # ── L2: Gait cycle / phase parameters ────────────────────────────────
    cycles: List[GaitCycle]
    events: List[GaitEvent]
    cadence_spm: float                      # steps per minute
    step_symmetry_index: float              # Robinson SI (%) — 0 = symmetric
    stance_ratio_r: float                   # right leg stance fraction
    stance_ratio_l: float                   # left leg stance fraction
    heel_strike_knee_angle: Dict[str, float]    # "right"/"left" → degrees
    peak_knee_flexion_swing: Dict[str, float]   # "right"/"left" → degrees
    hip_flexion_range: Dict[str, float]         # "right"/"left" → ROM degrees

    # ── L3: Whole-body spatial-temporal ──────────────────────────────────
    walking_speed_proxy: float              # normalised hip midpoint velocity (px/frame)
    gait_deviation_index: float             # 100 = normal; <75 = significant deficit
    bilateral_waveform_corr: float          # Pearson r of R vs L knee angle waveforms
    trunk_sway_range_deg: float             # lateral trunk oscillation (degrees)
    hip_drop_r: float                       # max right Trendelenburg angle (degrees)
    hip_drop_l: float                       # max left Trendelenburg angle (degrees)
    normative_zscores: Dict[str, float]     # z-score vs Winter (2009) norms

    # ── Clinical output ───────────────────────────────────────────────────
    risk_flags: List[str]                   # e.g. ["Hip drop L > 5°", "Asymmetric cadence"]
    overall_gait_score: float               # 0 (severe deficit) – 100 (normal)
    interpretation: str                     # paragraph for clinical report

    # ── Metadata ─────────────────────────────────────────────────────────
    n_frames: int
    fps_assumed: float = 25.0
    n_cycles_detected: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helper geometry
# ─────────────────────────────────────────────────────────────────────────────

def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b formed by vectors b→a and b→c (degrees)."""
    ba = a - b
    bc = c - b
    n_ba = np.linalg.norm(ba) + 1e-9
    n_bc = np.linalg.norm(bc) + 1e-9
    cos_theta = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _signed_trunk_angle(neck: np.ndarray, belly: np.ndarray) -> float:
    """
    Lateral trunk angle (degrees) — positive = lean right, negative = lean left.
    Computed as the angle of the neck-belly vector with respect to vertical.
    """
    vec = neck - belly          # upward vector
    # atan2(x, -y) gives angle from vertical in image coords (y down)
    return float(np.degrees(np.arctan2(vec[0], -vec[1])))


def _find_peaks_simple(signal: np.ndarray, min_prominence: float = 0.02,
                        min_distance: int = 8) -> np.ndarray:
    """
    Minimal 1-D peak finder (local maxima with distance constraint).
    Used for gait event detection without scipy dependency.
    """
    peaks = []
    n = len(signal)
    for i in range(1, n - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            # Prominence check: must exceed neighbours by min_prominence
            local_min = min(signal[max(0, i - min_distance):i + min_distance + 1])
            if signal[i] - local_min >= min_prominence:
                # Distance constraint
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
    return np.array(peaks, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# Core extractor
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalGaitFeatureExtractor:
    """
    Extract a 3-level hierarchical gait feature set from a 2D pose sequence.

    Input:  joints  — (T, 15, 2) normalised keypoint array
    Output: GaitReport containing all three hierarchy levels

    Algorithm overview:
        L1 — Compute joint angle time-series for 7 bilateral angle pairs.
        L2 — Detect gait events via ankle-height oscillation analysis
             (Pijnappels et al. 2001). Segment cycles; compute phase ratios,
             cadence, and Robinson Symmetry Index.
        L3 — Derive spatial-temporal parameters and GDI proxy.
             Compute bilateral waveform cross-correlation (Pearson r).
             Detect trunk sway and hip drop (Trendelenburg sign).
             Normalise all parameters against Winter (2009) norms.
    """

    def __init__(self, fps: float = 25.0, min_cycles: int = 1):
        self.fps = fps
        self.min_cycles = min_cycles

    # ── L1 ────────────────────────────────────────────────────────────────

    def _extract_angle_profiles(
        self, joints: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        L1: Compute per-joint angle time-series.

        Returns dict of {name: (T,) array of angles in degrees}.
        """
        T = joints.shape[0]
        profiles: Dict[str, np.ndarray] = {}

        angle_defs = {
            "hip_R":    (R_SHOULDER, R_HIP,   R_KNEE),
            "hip_L":    (L_SHOULDER, L_HIP,   L_KNEE),
            "knee_R":   (R_HIP,      R_KNEE,  R_ANKLE),
            "knee_L":   (L_HIP,      L_KNEE,  L_ANKLE),
            "ankle_R":  (R_KNEE,     R_ANKLE, R_HIP),     # dorsiflexion proxy
            "ankle_L":  (L_KNEE,     L_ANKLE, L_HIP),
            "trunk":    (FACE,       BELLY,   R_HIP),     # sagittal trunk lean
        }

        for name, (a, b, c) in angle_defs.items():
            arr = np.zeros(T)
            for t in range(T):
                j = joints[t]
                if j[a].sum() == 0 or j[b].sum() == 0 or j[c].sum() == 0:
                    arr[t] = np.nan
                else:
                    arr[t] = _angle3(j[a], j[b], j[c])
            # Fill NaN by linear interpolation
            nans = np.isnan(arr)
            if nans.all():
                arr[:] = 0.0
            elif nans.any():
                idx = np.arange(T)
                arr[nans] = np.interp(idx[nans], idx[~nans], arr[~nans])
            profiles[name] = arr

        # Trunk lateral sway (signed angle)
        trunk_sway = np.zeros(T)
        for t in range(T):
            j = joints[t]
            if j[NECK].sum() > 0 and j[BELLY].sum() > 0:
                trunk_sway[t] = _signed_trunk_angle(j[NECK], j[BELLY])
        profiles["trunk_lateral"] = trunk_sway

        return profiles

    # ── L2 ────────────────────────────────────────────────────────────────

    def _detect_gait_events(
        self, joints: np.ndarray
    ) -> Tuple[List[GaitEvent], List[GaitCycle]]:
        """
        L2: Detect heel-strike and toe-off events from ankle height oscillation.

        Method (Pijnappels et al. 2001):
            In 2D image coordinates (y increases downward), the ankle joint
            reaches its MAXIMUM y-value (lowest position) at heel-strike
            and its MINIMUM y-value (highest position) during mid-swing.

            1. Extract ankle_R_y and ankle_L_y time-series.
            2. Smooth with Gaussian kernel (σ = 3 frames) to suppress noise.
            3. Detect local maxima → heel-strike events.
            4. Detect local minima (negated maxima) → toe-off events.
            5. Pair events into complete gait cycles.
        """
        T = joints.shape[0]
        ankle_R_y = joints[:, R_ANKLE, 1]
        ankle_L_y = joints[:, L_ANKLE, 1]

        # Gaussian smoothing (σ = 3 frames)
        def gauss_smooth(x: np.ndarray, sigma: int = 3) -> np.ndarray:
            from math import exp
            k = sigma * 3
            kernel = np.array([exp(-0.5 * (i / sigma) ** 2)
                               for i in range(-k, k + 1)], dtype=float)
            kernel /= kernel.sum()
            return np.convolve(x, kernel, mode="same")

        r_smooth = gauss_smooth(ankle_R_y)
        l_smooth = gauss_smooth(ankle_L_y)

        # Adaptive min_distance: at 25fps, step ~0.5s → ~12 frames minimum
        min_dist = max(6, int(self.fps * 0.35))
        prominence = np.std(ankle_R_y) * 0.3

        hs_r = _find_peaks_simple(r_smooth,  min_prominence=prominence, min_distance=min_dist)
        hs_l = _find_peaks_simple(l_smooth,  min_prominence=prominence, min_distance=min_dist)
        to_r = _find_peaks_simple(-r_smooth, min_prominence=prominence, min_distance=min_dist)
        to_l = _find_peaks_simple(-l_smooth, min_prominence=prominence, min_distance=min_dist)

        events: List[GaitEvent] = []
        for idx in hs_r:
            events.append(GaitEvent(int(idx), "right", "heel_strike", 0.9))
        for idx in hs_l:
            events.append(GaitEvent(int(idx), "left",  "heel_strike", 0.9))
        for idx in to_r:
            events.append(GaitEvent(int(idx), "right", "toe_off",     0.8))
        for idx in to_l:
            events.append(GaitEvent(int(idx), "left",  "toe_off",     0.8))
        events.sort(key=lambda e: e.frame_idx)

        # Build cycles per side
        cycles: List[GaitCycle] = []
        for side, hs_arr, to_arr in [("right", hs_r, to_r), ("left", hs_l, to_l)]:
            if len(hs_arr) < 2:
                continue
            for i in range(len(hs_arr) - 1):
                start = int(hs_arr[i])
                end   = int(hs_arr[i + 1])
                # Find first toe-off within [start, end]
                within_to = to_arr[(to_arr > start) & (to_arr < end)]
                stance_end = int(within_to[0]) if len(within_to) > 0 else int(start + (end - start) * 0.6)
                dur = end - start
                stance = stance_end - start
                swing  = end - stance_end
                cycles.append(GaitCycle(
                    side=side, start_frame=start, end_frame=end,
                    stance_end_frame=stance_end,
                    duration_frames=dur, stance_frames=stance, swing_frames=swing,
                ))

        return events, cycles

    def _compute_phase_params(
        self, cycles: List[GaitCycle]
    ) -> Dict[str, float]:
        """
        L2: Aggregate phase parameters from detected cycles.

        Returns cadence, stance/swing ratios, step symmetry index.
        """
        if not cycles:
            return {"cadence_spm": 0.0, "stance_ratio_r": 0.6,
                    "stance_ratio_l": 0.6, "step_symmetry_index": 0.0}

        r_cycles = [c for c in cycles if c.side == "right"]
        l_cycles = [c for c in cycles if c.side == "left"]

        # Cadence: total steps / total time
        total_steps  = len(r_cycles) + len(l_cycles)
        if total_steps == 0:
            cadence = 0.0
        else:
            total_frames = max(c.end_frame for c in cycles) - min(c.start_frame for c in cycles)
            total_time_s = total_frames / self.fps
            cadence = (total_steps / total_time_s) * 60.0 if total_time_s > 0 else 0.0

        # Stance ratios
        stance_r = float(np.mean([c.stance_ratio for c in r_cycles])) if r_cycles else 0.60
        stance_l = float(np.mean([c.stance_ratio for c in l_cycles])) if l_cycles else 0.60

        # Robinson Symmetry Index: SI = 2|R - L| / (R + L) × 100
        # Applied to cycle duration (stride time symmetry)
        mean_dur_r = np.mean([c.duration_frames for c in r_cycles]) if r_cycles else 1.0
        mean_dur_l = np.mean([c.duration_frames for c in l_cycles]) if l_cycles else 1.0
        si = 200.0 * abs(mean_dur_r - mean_dur_l) / (mean_dur_r + mean_dur_l + 1e-9)

        return {
            "cadence_spm":         cadence,
            "stance_ratio_r":      stance_r,
            "stance_ratio_l":      stance_l,
            "step_symmetry_index": float(si),
        }

    def _compute_event_angles(
        self, profiles: Dict[str, np.ndarray], events: List[GaitEvent]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        L2: Extract clinically important angles at gait events.

        Returns:
            heel_strike_knee_angle  — knee flexion at heel-strike (per side)
            peak_knee_flexion_swing — maximum knee flexion during swing (per side)
            hip_flexion_range       — full hip ROM per side
        """
        hs_knee: Dict[str, float]  = {"right": 0.0, "left": 0.0}
        pk_knee: Dict[str, float]  = {"right": 65.0, "left": 65.0}
        hip_rom: Dict[str, float]  = {"right": 40.0, "left": 40.0}

        for side in ("right", "left"):
            s = side[0].upper()    # "R" or "L"
            knee_profile = profiles.get(f"knee_{s[0]}", np.array([]))
            hip_profile  = profiles.get(f"hip_{s[0]}",  np.array([]))

            hs_events = [e for e in events
                         if e.side == side and e.event_type == "heel_strike"]
            to_events = [e for e in events
                         if e.side == side and e.event_type == "toe_off"]

            if len(knee_profile) == 0:
                continue

            # Heel-strike knee angle: mean across all detected HS
            if hs_events:
                angles_at_hs = [knee_profile[min(e.frame_idx, len(knee_profile) - 1)]
                                 for e in hs_events]
                hs_knee[side] = float(np.mean(angles_at_hs))

            # Peak knee flexion during swing: between TO and next HS
            swing_peaks = []
            for to_ev in to_events:
                next_hs = next((e for e in hs_events
                                if e.frame_idx > to_ev.frame_idx), None)
                if next_hs and len(knee_profile) > to_ev.frame_idx:
                    swing_slice = knee_profile[to_ev.frame_idx:next_hs.frame_idx + 1]
                    if len(swing_slice) > 0:
                        swing_peaks.append(float(np.max(swing_slice)))
            if swing_peaks:
                pk_knee[side] = float(np.mean(swing_peaks))

            # Hip flexion ROM
            if len(hip_profile) > 0:
                hip_rom[side] = float(np.max(hip_profile) - np.min(hip_profile))

        return hs_knee, pk_knee, hip_rom

    # ── L3 ────────────────────────────────────────────────────────────────

    def _compute_spatiotemporal(
        self,
        joints: np.ndarray,
        profiles: Dict[str, np.ndarray],
    ) -> Dict[str, float]:
        """
        L3: Compute whole-body spatial-temporal parameters.

        Parameters computed:
        ┌─────────────────────────────┬──────────────────────────────────────┐
        │ walking_speed_proxy         │ Mean hip-midpoint horizontal velocity │
        │ trunk_sway_range_deg        │ Peak-to-peak lateral trunk oscillation│
        │ hip_drop_r / hip_drop_l     │ Max Trendelenburg angle per side      │
        │ bilateral_waveform_corr     │ Pearson r of R/L knee angle waveforms │
        │ gait_deviation_index        │ GDI proxy (Schwartz & Rozumalski 2008)│
        └─────────────────────────────┴──────────────────────────────────────┘
        """
        T = joints.shape[0]

        # Walking speed proxy: mean |Δx| of hip midpoint per frame
        hip_mid_x = (joints[:, R_HIP, 0] + joints[:, L_HIP, 0]) / 2.0
        speed_proxy = float(np.mean(np.abs(np.diff(hip_mid_x))))

        # Trunk lateral sway range
        trunk_lat = profiles.get("trunk_lateral", np.zeros(T))
        trunk_sway = float(np.nanmax(trunk_lat) - np.nanmin(trunk_lat)) if len(trunk_lat) > 0 else 0.0

        # Hip drop (Trendelenburg): R_HIP.y - L_HIP.y (positive = right hip dropped)
        # In image coords: y↓, so if R_HIP.y > L_HIP.y → right pelvis drops
        hip_diff = joints[:, R_HIP, 1] - joints[:, L_HIP, 1]
        # Convert from normalised pixels to approximate degrees via small-angle theorem
        # hip_width ≈ distance between hips; 1 px offset at 0.2 width ≈ 5°
        hip_width = np.mean(np.linalg.norm(
            joints[:, R_HIP, :] - joints[:, L_HIP, :], axis=1
        )) + 1e-9
        hip_drop_r = float(np.nanmax( hip_diff) / hip_width * 90.0 / np.pi * 2)
        hip_drop_l = float(np.nanmax(-hip_diff) / hip_width * 90.0 / np.pi * 2)

        # Bilateral waveform correlation (R vs L knee angle profiles)
        knee_r = profiles.get("knee_R", np.zeros(T))
        knee_l = profiles.get("knee_L", np.zeros(T))
        if len(knee_r) > 1 and np.std(knee_r) > 0 and np.std(knee_l) > 0:
            corr = float(np.corrcoef(knee_r, knee_l)[0, 1])
        else:
            corr = 1.0

        # GDI proxy: camera-angle-invariant version using Range of Motion.
        # Rather than comparing absolute angles (which depend on camera view),
        # compare the observed joint ROM against normative ROM ranges.
        # Normal ROM during walking (Winter 2009):
        #   Knee: 55-70° ROM  |  Hip: 40-50° ROM  |  Ankle: 20-30° ROM
        # This works for frontal, lateral, or any camera angle.
        ROM_NORMS = [
            ("knee_R", 60.0, 15.0),   # (key, norm_ROM_deg, sigma)
            ("knee_L", 60.0, 15.0),
            ("hip_R",  45.0, 12.0),
            ("hip_L",  45.0, 12.0),
        ]
        deviations = []
        for key, norm_rom, sigma in ROM_NORMS:
            profile = profiles.get(key, np.array([]))
            if len(profile) > 1 and np.ptp(profile) > 0.5:  # skip if no movement
                observed_rom = float(np.ptp(profile))        # peak-to-peak range
                z = (observed_rom - norm_rom) / (sigma + 1e-9)
                deviations.append(z ** 2)
        if deviations:
            gdi = max(0.0, 100.0 - 10.0 * float(np.sqrt(np.mean(deviations))))
        else:
            # Fallback: if no joint movement detected, score based on bilateral symmetry
            gdi = 75.0 if corr > 0.7 else 55.0

        return {
            "walking_speed_proxy":   speed_proxy,
            "trunk_sway_range_deg":  trunk_sway,
            "hip_drop_r":            max(0.0, hip_drop_r),
            "hip_drop_l":            max(0.0, hip_drop_l),
            "bilateral_waveform_corr": corr,
            "gait_deviation_index":  gdi,
        }

    def _normative_zscores(self, params: Dict[str, float]) -> Dict[str, float]:
        """
        L3: Compute z-scores relative to Winter (2009) normative database.
        z = (observed - mean) / std
        |z| > 2 → outside normal range (flag for clinical review).
        """
        zscores = {}
        for key, (mu, sigma) in NORMATIVE.items():
            val = params.get(key, mu)
            zscores[key] = round(float((val - mu) / (sigma + 1e-9)), 2)
        return zscores

    def _risk_flags(
        self,
        params: Dict[str, float],
        zscores: Dict[str, float],
        hip_drop_r: float,
        hip_drop_l: float,
    ) -> List[str]:
        """Generate clinical risk flag strings from computed parameters."""
        flags = []

        si = params.get("step_symmetry_index", 0)
        if si > 15:
            flags.append(f"Severe gait asymmetry: SI = {si:.0f}% (normal < 5%)")
        elif si > 8:
            flags.append(f"Mild gait asymmetry: SI = {si:.0f}%")

        cad = params.get("cadence_spm", 110)
        if cad > 0:
            if cad < 80:
                flags.append(f"Low cadence: {cad:.0f} spm (normal 100-120) — pain/weakness suspected")
            elif cad > 140:
                flags.append(f"High cadence: {cad:.0f} spm — short step strategy suspected")

        if hip_drop_r > 8:
            flags.append(f"Trendelenburg sign RIGHT: {hip_drop_r:.1f}° — hip abductor weakness")
        if hip_drop_l > 8:
            flags.append(f"Trendelenburg sign LEFT: {hip_drop_l:.1f}° — hip abductor weakness")

        sway = params.get("trunk_sway_range_deg", 4)
        if sway > 12:
            flags.append(f"Excessive trunk sway: {sway:.1f}° (normal < 8°) — balance deficit")

        corr = params.get("bilateral_waveform_corr", 1.0)
        if corr < 0.6:
            flags.append(f"Bilateral knee waveform dissimilarity (r={corr:.2f}) — asymmetric loading")

        gdi = params.get("gait_deviation_index", 100)
        if gdi < 75:
            flags.append(f"GDI {gdi:.0f} — significant gait pathology")
        elif gdi < 88:
            flags.append(f"GDI {gdi:.0f} — borderline gait quality")

        return flags

    def _overall_score(self, params: Dict[str, float], n_flags: int) -> float:
        """
        Composite gait health score (0–100).

        Computed as a weighted combination of:
          - GDI proxy           (40%)
          - Symmetry index      (25%)  — inverted
          - Cadence normalcy    (20%)
          - Trunk sway          (15%)  — inverted
        Penalised by 5 points per risk flag (capped at 0).
        """
        gdi  = params.get("gait_deviation_index", 100.0)
        si   = params.get("step_symmetry_index",   0.0)
        cad  = params.get("cadence_spm",           110.0)
        sway = params.get("trunk_sway_range_deg",   4.0)

        gdi_score  = np.clip(gdi, 0, 100)
        si_score   = np.clip(100 - si * 2, 0, 100)
        cad_score  = np.clip(100 - abs(cad - 110) * 1.5, 0, 100) if cad > 0 else 50.0
        # Sway: cap penalty at 25° range (beyond that treat as max penalty)
        # Prevents extreme values from frontal-view noise dominating the score
        sway_capped = min(sway, 25.0)
        sway_score  = np.clip(100 - sway_capped * 4, 0, 100)

        composite = (0.40 * gdi_score + 0.25 * si_score +
                     0.20 * cad_score + 0.15 * sway_score)
        # Cap flag penalty so score never drops below 10 from flags alone
        composite -= min(n_flags * 5, composite - 10)
        return float(np.clip(composite, 0, 100))

    def _interpretation(
        self, score: float, flags: List[str], params: Dict[str, float]
    ) -> str:
        """Generate paragraph-level clinical interpretation."""
        if score >= 85:
            level = "within normal limits"
        elif score >= 65:
            level = "mildly impaired"
        elif score >= 45:
            level = "moderately impaired"
        else:
            level = "severely impaired"

        cad  = params.get("cadence_spm", 0)
        si   = params.get("step_symmetry_index", 0)
        gdi  = params.get("gait_deviation_index", 100)

        text = (
            f"Gait analysis shows {level} gait biomechanics (Score: {score:.0f}/100). "
            f"Cadence: {cad:.0f} spm. Symmetry Index: {si:.1f}% "
            f"({'normal' if si < 5 else 'elevated — see flags'}). "
            f"GDI proxy: {gdi:.0f}/100."
        )
        if flags:
            text += " Clinical flags: " + "; ".join(flags[:3]) + "."
        text += (" Note: Analysis based on 2D camera data (25 fps assumed). "
                 "Confirm findings with instrumented gait analysis where indicated.")
        return text

    # ── Public API ────────────────────────────────────────────────────────

    def analyse(self, joints: np.ndarray) -> GaitReport:
        """
        Full HGD analysis pipeline.

        Args:
            joints: (T, 15, 2) normalised keypoint array.
                    T ≥ 20 frames recommended for reliable gait event detection.

        Returns:
            GaitReport with all three hierarchy levels populated.
        """
        T = joints.shape[0]
        if T < 10:
            raise ValueError(f"Too few frames ({T}) for gait analysis. Need ≥ 10.")

        # ── L1 ───────────────────────────────────────────────────────────
        profiles = self._extract_angle_profiles(joints)

        # ── L2 ───────────────────────────────────────────────────────────
        events, cycles = self._detect_gait_events(joints)
        phase_params   = self._compute_phase_params(cycles)
        hs_knee, pk_knee, hip_rom = self._compute_event_angles(profiles, events)

        # ── L3 ───────────────────────────────────────────────────────────
        spat = self._compute_spatiotemporal(joints, profiles)

        all_params = {**phase_params, **spat,
                      "knee_flexion_at_strike": float(np.mean(list(hs_knee.values()))),
                      "peak_knee_flexion_swing": float(np.mean(list(pk_knee.values()))),
                      "hip_flexion_range": float(np.mean(list(hip_rom.values())))}

        zscores   = self._normative_zscores(all_params)
        flags     = self._risk_flags(all_params, zscores,
                                     spat["hip_drop_r"], spat["hip_drop_l"])
        score     = self._overall_score(all_params, len(flags))
        interp    = self._interpretation(score, flags, all_params)

        return GaitReport(
            angle_profiles        = profiles,
            cycles                = cycles,
            events                = events,
            cadence_spm           = phase_params["cadence_spm"],
            step_symmetry_index   = phase_params["step_symmetry_index"],
            stance_ratio_r        = phase_params["stance_ratio_r"],
            stance_ratio_l        = phase_params["stance_ratio_l"],
            heel_strike_knee_angle= hs_knee,
            peak_knee_flexion_swing=pk_knee,
            hip_flexion_range     = hip_rom,
            walking_speed_proxy   = spat["walking_speed_proxy"],
            gait_deviation_index  = spat["gait_deviation_index"],
            bilateral_waveform_corr=spat["bilateral_waveform_corr"],
            trunk_sway_range_deg  = spat["trunk_sway_range_deg"],
            hip_drop_r            = spat["hip_drop_r"],
            hip_drop_l            = spat["hip_drop_l"],
            normative_zscores     = zscores,
            risk_flags            = flags,
            overall_gait_score    = score,
            interpretation        = interp,
            n_frames              = T,
            fps_assumed           = self.fps,
            n_cycles_detected     = len(cycles),
        )

    def extract_feature_vector(self, joints: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """
        Extract a flat feature vector from GaitReport for ML training.

        This bridges HGD output into the HierarchicalFeatureExtractor pipeline,
        allowing gait features to be appended to the main feature matrix for
        classification and ablation studies.

        Returns:
            features: (n_features,) float32 array
            names:    list of feature name strings
        """
        report = self.analyse(joints)

        feats: List[float] = []
        names: List[str]   = []

        # L1 aggregates
        for joint_name, profile in report.angle_profiles.items():
            for stat, val in [
                ("mean", np.mean(profile)),
                ("std",  np.std(profile)),
                ("range", np.max(profile) - np.min(profile)),
            ]:
                feats.append(float(val))
                names.append(f"gait_L1_{joint_name}_{stat}")

        # L2 parameters
        for key, val in [
            ("cadence_spm",          report.cadence_spm),
            ("step_symmetry_index",  report.step_symmetry_index),
            ("stance_ratio_r",       report.stance_ratio_r),
            ("stance_ratio_l",       report.stance_ratio_l),
            ("hs_knee_R",            report.heel_strike_knee_angle.get("right", 5.0)),
            ("hs_knee_L",            report.heel_strike_knee_angle.get("left",  5.0)),
            ("pk_knee_swing_R",      report.peak_knee_flexion_swing.get("right", 65.0)),
            ("pk_knee_swing_L",      report.peak_knee_flexion_swing.get("left",  65.0)),
            ("hip_rom_R",            report.hip_flexion_range.get("right", 40.0)),
            ("hip_rom_L",            report.hip_flexion_range.get("left",  40.0)),
        ]:
            feats.append(float(val))
            names.append(f"gait_L2_{key}")

        # L3 parameters
        for key, val in [
            ("speed_proxy",          report.walking_speed_proxy),
            ("gait_deviation_index", report.gait_deviation_index),
            ("bilateral_corr",       report.bilateral_waveform_corr),
            ("trunk_sway_deg",       report.trunk_sway_range_deg),
            ("hip_drop_r",           report.hip_drop_r),
            ("hip_drop_l",           report.hip_drop_l),
            ("overall_score",        report.overall_gait_score),
        ]:
            feats.append(float(val))
            names.append(f"gait_L3_{key}")

        return np.array(feats, dtype=np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_gait_dashboard(report: GaitReport):
    """
    4-panel gait analysis dashboard figure.

    Panel 1: L1 — Bilateral knee angle time-series with event markers
    Panel 2: L1 — Hip angle profiles (R vs L)
    Panel 3: L2 — Phase composition bar chart (stance/swing per side)
    Panel 4: L3 — Normative z-score radar
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#0e1117")
    fig.suptitle("Hierarchical Gait Analysis — HGD Report",
                 color="white", fontsize=13, fontweight="bold")
    for ax in axes.flat:
        ax.set_facecolor("#1a1a2e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    T = report.n_frames
    frames = np.arange(T)

    # ── Panel 1: Bilateral knee profiles ─────────────────────────────────
    ax = axes[0, 0]
    knee_r = report.angle_profiles.get("knee_R", np.zeros(T))
    knee_l = report.angle_profiles.get("knee_L", np.zeros(T))
    ax.plot(frames, knee_r, color="#4499dd", lw=2, label="Right knee")
    ax.plot(frames, knee_l, color="#dd9944", lw=2, label="Left knee",  ls="--")
    for ev in report.events:
        color = "#00dc50" if ev.event_type == "heel_strike" else "#ff6666"
        ax.axvline(ev.frame_idx, color=color, alpha=0.4, lw=1)
    ax.set_title("L1 — Bilateral Knee Angles")
    ax.set_xlabel("Frame"); ax.set_ylabel("Angle (°)")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    if report.bilateral_waveform_corr > 0:
        ax.text(0.02, 0.95, f"Waveform r = {report.bilateral_waveform_corr:.2f}",
                transform=ax.transAxes, color="yellow", fontsize=8, va="top")

    # ── Panel 2: Hip angle profiles ───────────────────────────────────────
    ax = axes[0, 1]
    hip_r = report.angle_profiles.get("hip_R", np.zeros(T))
    hip_l = report.angle_profiles.get("hip_L", np.zeros(T))
    ax.plot(frames, hip_r, color="#ff7f7f", lw=2, label="Right hip")
    ax.plot(frames, hip_l, color="#7fff7f", lw=2, label="Left hip", ls="--")
    ax.set_title("L1 — Hip Angle Profiles")
    ax.set_xlabel("Frame"); ax.set_ylabel("Angle (°)")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # ── Panel 3: Phase composition ────────────────────────────────────────
    ax = axes[1, 0]
    sides = ["Right", "Left"]
    stance = [report.stance_ratio_r * 100, report.stance_ratio_l * 100]
    swing  = [(1 - report.stance_ratio_r) * 100,
              (1 - report.stance_ratio_l) * 100]
    x = np.arange(2)
    ax.bar(x, stance, color="#4499dd", label="Stance %", width=0.4)
    ax.bar(x, swing,  bottom=stance, color="#dd9944", label="Swing %", width=0.4)
    ax.axhline(60, color="lime", ls="--", lw=1, alpha=0.6, label="Normal stance (60%)")
    ax.set_xticks(x); ax.set_xticklabels(sides, color="white")
    ax.set_title(f"L2 — Phase Composition (Cadence: {report.cadence_spm:.0f} spm)")
    ax.set_ylabel("% Gait Cycle"); ax.set_ylim(0, 120)
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # ── Panel 4: Normative z-score bar chart ──────────────────────────────
    ax = axes[1, 1]
    z_keys   = list(report.normative_zscores.keys())
    z_values = [report.normative_zscores[k] for k in z_keys]
    short_labels = [k.replace("_", "\n") for k in z_keys]
    colors = ["#ff4040" if abs(z) > 2 else "#ffcc00" if abs(z) > 1 else "#00dc50"
              for z in z_values]
    ax.bar(range(len(z_keys)), z_values, color=colors, width=0.6)
    ax.axhline( 2, color="red",   ls="--", lw=1, alpha=0.5)
    ax.axhline(-2, color="red",   ls="--", lw=1, alpha=0.5)
    ax.axhline( 0, color="white", ls="-",  lw=0.5, alpha=0.3)
    ax.set_xticks(range(len(z_keys)))
    ax.set_xticklabels(short_labels, color="white", fontsize=7)
    ax.set_title(f"L3 — Normative Z-scores  |  GDI: {report.gait_deviation_index:.0f}")
    ax.set_ylabel("Z-score")
    ax.text(0.98, 0.97, f"Score: {report.overall_gait_score:.0f}/100",
            transform=ax.transAxes, color="white", fontsize=9,
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="#333", ec="#666"))

    plt.tight_layout()
    return fig
