"""Bilateral symmetry analysis — compare left vs right joint angles."""
from __future__ import annotations
from typing import List, Dict
import numpy as np

# JHMDB joint indices
NECK=0;BELLY=1;FACE=2;R_SHOULDER=3;L_SHOULDER=4;R_HIP=5;L_HIP=6
R_ELBOW=7;L_ELBOW=8;R_KNEE=9;L_KNEE=10;R_WRIST=11;L_WRIST=12;R_ANKLE=13;L_ANKLE=14

# (pair_label, right triplet, left triplet)
SYMMETRY_PAIRS = [
    ("Shoulder",
     (NECK, R_SHOULDER, R_ELBOW),
     (NECK, L_SHOULDER, L_ELBOW)),
    ("Elbow",
     (R_SHOULDER, R_ELBOW, R_WRIST),
     (L_SHOULDER, L_ELBOW, L_WRIST)),
    ("Hip",
     (R_SHOULDER, R_HIP, R_KNEE),
     (L_SHOULDER, L_HIP, L_KNEE)),
    ("Knee",
     (R_HIP, R_KNEE, R_ANKLE),
     (L_HIP, L_KNEE, L_ANKLE)),
    ("Hip-Knee angle",
     (NECK, R_HIP, R_KNEE),
     (NECK, L_HIP, L_KNEE)),
]


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba, bc = a - b, c - b
    return float(np.degrees(np.arccos(
        np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9), -1, 1)
    )))


def analyse_symmetry(joints: np.ndarray) -> List[Dict]:
    """
    Returns list of dicts with bilateral comparison per joint pair.
    joints: (15, 2) array
    """
    results = []
    for label, (ra, rb, rc), (la, lb, lc) in SYMMETRY_PAIRS:
        # Skip if any joint is missing
        if (joints[ra].sum() == 0 or joints[rb].sum() == 0 or joints[rc].sum() == 0 or
                joints[la].sum() == 0 or joints[lb].sum() == 0 or joints[lc].sum() == 0):
            continue

        r_angle = compute_angle(joints[ra], joints[rb], joints[rc])
        l_angle = compute_angle(joints[la], joints[lb], joints[lc])
        asym    = abs(r_angle - l_angle)

        if asym < 5:
            status, color = "✅ Symmetric", "green"
            note = "Within normal bilateral variation (<5°)."
        elif asym < 10:
            status, color = "🟡 Mild asymmetry", "orange"
            side = "right" if r_angle > l_angle else "left"
            note = f"Mild {asym:.1f}° difference. {side.capitalize()} side is higher. Monitor."
        else:
            status, color = "🔴 Significant asymmetry", "red"
            side = "right" if r_angle > l_angle else "left"
            note = (f"Significant {asym:.1f}° asymmetry. {side.capitalize()} side is higher. "
                    "May indicate compensation, pain avoidance, or muscle imbalance.")

        results.append({
            "joint_pair":    label,
            "right_angle":   round(r_angle, 1),
            "left_angle":    round(l_angle, 1),
            "asymmetry":     round(asym, 1),
            "status":        status,
            "_color":        color,
            "note":          note,
        })
    return results


def symmetry_score(sym_data: List[Dict]) -> float:
    """Overall symmetry score 0–100% (100 = perfectly symmetric)."""
    if not sym_data:
        return 100.0
    scores = []
    for s in sym_data:
        asym = s["asymmetry"]
        score = max(0.0, 1.0 - asym / 20.0)   # 0° = 100%, 20°+ = 0%
        scores.append(score)
    return round(np.mean(scores) * 100, 1)


def plot_symmetry(sym_data: List[Dict]):
    """Horizontal bar chart comparing R vs L angles."""
    import matplotlib.pyplot as plt
    if not sym_data:
        return None

    labels = [s["joint_pair"] for s in sym_data]
    r_vals = [s["right_angle"] for s in sym_data]
    l_vals = [s["left_angle"]  for s in sym_data]
    colors = [{"green": "#00dc50", "orange": "#ffcc00", "red": "#ff4040"}[s["_color"]]
              for s in sym_data]

    n = len(labels)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(8, max(3, n * 0.7)), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    ax.barh(y - 0.2, r_vals, height=0.35, label="Right",
            color="#4499dd", edgecolor="#333", alpha=0.9)
    ax.barh(y + 0.2, l_vals, height=0.35, label="Left",
            color="#dd9944", edgecolor="#333", alpha=0.9)

    # Asymmetry markers
    for i, s in enumerate(sym_data):
        ax.plot([s["right_angle"], s["left_angle"]], [i - 0.2, i + 0.2],
                color=colors[i], lw=2, zorder=5)
        ax.text(max(r_vals + l_vals) * 1.02, i,
                f"Δ {s['asymmetry']:.1f}°",
                va="center", color=colors[i], fontsize=8)

    ax.set_yticks(y); ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Angle (°)", color="white"); ax.tick_params(colors="white")
    ax.set_title("Bilateral Symmetry — Right vs Left", color="white", fontsize=11)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def plot_rom_progress(rom_records: List[Dict], joint_name: str):
    """Plot ROM improvement over sessions for one joint."""
    import matplotlib.pyplot as plt
    if len(rom_records) < 2:
        return None

    dates  = [r["recorded_at"][:10] for r in rom_records]
    max_a  = [r["max_angle"] for r in rom_records]
    target = rom_records[-1].get("target", None)

    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(range(len(dates)), max_a, color="deepskyblue", lw=2,
            marker="o", markersize=6, label="Max angle achieved")
    ax.fill_between(range(len(dates)), max_a, alpha=0.2, color="deepskyblue")

    if target:
        ax.axhline(target, color="lime", lw=1.5, ls="--", alpha=0.8,
                   label=f"Target: {target}°")

    # Trend line
    if len(max_a) >= 3:
        z = np.polyfit(range(len(max_a)), max_a, 1)
        p = np.poly1d(z)
        ax.plot(range(len(dates)), p(range(len(max_a))),
                color="yellow", lw=1, ls=":", alpha=0.7, label="Trend")

    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels(dates, rotation=30, ha="right", color="white", fontsize=7)
    ax.set_ylabel("Max Angle (°)", color="white")
    ax.set_title(f"ROM Progress: {joint_name}", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # Improvement annotation
    improvement = max_a[-1] - max_a[0]
    color = "#00dc50" if improvement > 0 else "#ff4040"
    ax.annotate(f"{'▲' if improvement > 0 else '▼'} {abs(improvement):.0f}° overall",
                xy=(len(dates) - 1, max_a[-1]),
                xytext=(-40, 15), textcoords="offset points",
                color=color, fontsize=9, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=color))

    plt.tight_layout()
    return fig
