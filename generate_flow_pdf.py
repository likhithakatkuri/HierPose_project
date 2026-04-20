"""Generate HierPose Project Flow PDF — ready to present."""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import PageBreak

W, H = A4
MARGIN = 1.8 * cm

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#0d2137")
BLUE   = colors.HexColor("#1565C0")
LBLUE  = colors.HexColor("#1976D2")
CYAN   = colors.HexColor("#00ACC1")
GREEN  = colors.HexColor("#2E7D32")
LGREEN = colors.HexColor("#43A047")
ORANGE = colors.HexColor("#E65100")
PURPLE = colors.HexColor("#6A1B9A")
GRAY   = colors.HexColor("#37474F")
LGRAY  = colors.HexColor("#ECEFF1")
WHITE  = colors.white
GOLD   = colors.HexColor("#F9A825")

# ── Styles ─────────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def style(name, **kw):
    s = ParagraphStyle(name, **kw)
    return s

TITLE  = style("title",  fontSize=22, textColor=WHITE,  alignment=TA_CENTER,
               fontName="Helvetica-Bold", spaceAfter=4, leading=28)
SUBT   = style("subt",   fontSize=11, textColor=colors.HexColor("#B0BEC5"),
               alignment=TA_CENTER, fontName="Helvetica", spaceAfter=2)
META   = style("meta",   fontSize=9,  textColor=colors.HexColor("#90A4AE"),
               alignment=TA_CENTER, fontName="Helvetica")

SEC    = style("sec",    fontSize=13, textColor=WHITE, fontName="Helvetica-Bold",
               spaceAfter=6, leading=18)
BODY   = style("body",   fontSize=9,  textColor=GRAY,  fontName="Helvetica",
               spaceAfter=3, leading=14)
BODYW  = style("bodyw",  fontSize=9,  textColor=WHITE, fontName="Helvetica",
               spaceAfter=3, leading=14)
BOLD   = style("bold",   fontSize=9,  textColor=NAVY,  fontName="Helvetica-Bold",
               spaceAfter=3, leading=14)
SMALL  = style("small",  fontSize=8,  textColor=GRAY,  fontName="Helvetica",
               spaceAfter=2, leading=12)
SMALLW = style("smallw", fontSize=8,  textColor=WHITE, fontName="Helvetica",
               spaceAfter=2, leading=12)
LABEL  = style("label",  fontSize=8,  textColor=WHITE, fontName="Helvetica-Bold",
               alignment=TA_CENTER)
STEP   = style("step",   fontSize=9,  textColor=WHITE, fontName="Helvetica-Bold",
               alignment=TA_CENTER, leading=12)
STEPC  = style("stepc",  fontSize=8,  textColor=colors.HexColor("#B0BEC5"),
               alignment=TA_CENTER, leading=11)
TH     = style("th",     fontSize=8,  textColor=WHITE, fontName="Helvetica-Bold",
               alignment=TA_CENTER)
TD     = style("td",     fontSize=8,  textColor=GRAY,  fontName="Helvetica",
               alignment=TA_LEFT)
TDC    = style("tdc",    fontSize=8,  textColor=GRAY,  fontName="Helvetica",
               alignment=TA_CENTER)

# ── Helpers ────────────────────────────────────────────────────────────────────
def banner(text, bg=NAVY, fg=WHITE, fontsize=13):
    t = Table([[Paragraph(text, style("b", fontSize=fontsize, textColor=fg,
                fontName="Helvetica-Bold", alignment=TA_CENTER))]],
              colWidths=[W - 2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), bg),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING",(0,0), (-1,-1), 12),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), [6,6,6,6]),
    ]))
    return t

def flow_box(number, title, subtitle, color=BLUE):
    content = [
        [Paragraph(f"<b>Step {number}</b>", STEP),
         Paragraph(title, STEP),
         Paragraph(subtitle, STEPC)]
    ]
    t = Table(content, colWidths=[1.5*cm, 6*cm, W-2*MARGIN-8.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",  (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
        ("BOX", (0,0), (-1,-1), 0.5, color),
    ]))
    return t

def arrow():
    t = Table([[Paragraph("▼", style("arr", fontSize=14, textColor=CYAN,
                fontName="Helvetica-Bold", alignment=TA_CENTER))]],
              colWidths=[W-2*MARGIN])
    t.setStyle(TableStyle([
        ("TOPPADDING",  (0,0),(-1,-1), 2),
        ("BOTTOMPADDING",(0,0),(-1,-1), 2),
    ]))
    return t

def section_header(text, color=BLUE):
    t = Table([[Paragraph(text, style("sh", fontSize=11, textColor=WHITE,
                fontName="Helvetica-Bold"))]], colWidths=[W-2*MARGIN])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("TOPPADDING",  (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING", (0,0),(-1,-1), 12),
    ]))
    return t

def data_table(headers, rows, col_widths, header_bg=LBLUE):
    data = [[Paragraph(h, TH) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), TDC if i > 0 else TD)
                     for i, c in enumerate(row)])
    t = Table(data, colWidths=col_widths)
    ts = TableStyle([
        ("BACKGROUND", (0,0), (-1,0), header_bg),
        ("GRID",       (0,0), (-1,-1), 0.3, colors.HexColor("#CFD8DC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [WHITE, LGRAY]),
        ("TOPPADDING",    (0,0),(-1,-1), 5),
        ("BOTTOMPADDING", (0,0),(-1,-1), 5),
        ("LEFTPADDING",   (0,0),(-1,-1), 6),
        ("RIGHTPADDING",  (0,0),(-1,-1), 6),
        ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
    ])
    t.setStyle(ts)
    return t

def domain_card(icon, title, desc, color):
    t = Table([[Paragraph(f"{icon}  <b>{title}</b>", style("dc",
                fontSize=9, textColor=WHITE, fontName="Helvetica-Bold")),
                Paragraph(desc, SMALLW)]],
              colWidths=[4.5*cm, W-2*MARGIN-5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), color),
        ("VALIGN", (0,0),(-1,-1), "TOP"),
        ("TOPPADDING",   (0,0),(-1,-1), 7),
        ("BOTTOMPADDING",(0,0),(-1,-1), 7),
        ("LEFTPADDING",  (0,0),(-1,-1), 8),
        ("RIGHTPADDING", (0,0),(-1,-1), 8),
        ("BOX", (0,0),(-1,-1), 0.3, colors.HexColor("#546E7A")),
    ]))
    return t

# ── Build document ─────────────────────────────────────────────────────────────
def build():
    doc = SimpleDocTemplate(
        "HierPose_Flow.pdf",
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
        title="HierPose — Project Flow",
        author="K. Likhitha Reddy, M. Devansh"
    )

    story = []
    sp = lambda n=0.3: story.append(Spacer(1, n*cm))
    hr = lambda: story.append(HRFlowable(width="100%", thickness=0.5,
                               color=colors.HexColor("#CFD8DC"), spaceAfter=4))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 1 — TITLE + OVERVIEW
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Title block
    title_block = Table([[
        Paragraph("HierPose", style("tt", fontSize=28, textColor=WHITE,
                  fontName="Helvetica-Bold", alignment=TA_CENTER)),
        Paragraph("Multi-Domain Intelligent Pose Analysis Platform",
                  style("ts", fontSize=13, textColor=colors.HexColor("#90CAF9"),
                  fontName="Helvetica", alignment=TA_CENTER, leading=18)),
        Paragraph("Major Project · PID: AIML/2025-26/PID-2.1 · CBIT, Dept. of AIML",
                  style("tm", fontSize=9, textColor=colors.HexColor("#78909C"),
                  fontName="Helvetica", alignment=TA_CENTER)),
        Paragraph("K. Likhitha Reddy (160122729010) · M. Devansh (160122729044) · Guide: Dr. Y. Rama Devi",
                  style("ta", fontSize=8, textColor=colors.HexColor("#607D8B"),
                  fontName="Helvetica", alignment=TA_CENTER)),
    ]], colWidths=[W-2*MARGIN])
    title_block.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), NAVY),
        ("TOPPADDING",    (0,0),(-1,-1), 18),
        ("BOTTOMPADDING", (0,0),(-1,-1), 18),
        ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ("RIGHTPADDING",  (0,0),(-1,-1), 12),
    ]))
    story.append(title_block)
    sp(0.4)

    # Key metrics strip
    metrics = Table([
        [Paragraph("84.1%", style("mv", fontSize=18, textColor=GOLD,
                   fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph("592+", style("mv", fontSize=18, textColor=GOLD,
                   fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph("8", style("mv", fontSize=18, textColor=GOLD,
                   fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph("5", style("mv", fontSize=18, textColor=GOLD,
                   fontName="Helvetica-Bold", alignment=TA_CENTER)),
         Paragraph("7", style("mv", fontSize=18, textColor=GOLD,
                   fontName="Helvetica-Bold", alignment=TA_CENTER))],
        [Paragraph("Action Accuracy", style("ml", fontSize=7.5, textColor=WHITE,
                   fontName="Helvetica", alignment=TA_CENTER)),
         Paragraph("Feature Dims", style("ml", fontSize=7.5, textColor=WHITE,
                   fontName="Helvetica", alignment=TA_CENTER)),
         Paragraph("App Domains", style("ml", fontSize=7.5, textColor=WHITE,
                   fontName="Helvetica", alignment=TA_CENTER)),
         Paragraph("Novel Algorithms", style("ml", fontSize=7.5, textColor=WHITE,
                   fontName="Helvetica", alignment=TA_CENTER)),
         Paragraph("Research Modules", style("ml", fontSize=7.5, textColor=WHITE,
                   fontName="Helvetica", alignment=TA_CENTER))],
    ], colWidths=[(W-2*MARGIN)/5]*5)
    metrics.setStyle(TableStyle([
        ("BACKGROUND", (0,0),(-1,-1), BLUE),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("INNERGRID", (0,0),(-1,-1), 0.3, colors.HexColor("#1E88E5")),
    ]))
    story.append(metrics)
    sp(0.4)

    # ── Section 1: Full Pipeline Flow ─────────────────────────────────────────
    story.append(section_header("FULL SYSTEM FLOW — TOP TO BOTTOM", BLUE))
    sp(0.3)

    steps = [
        (1, "USER INPUT",           "Video file · Live webcam · Photo · JHMDB .mat file",       BLUE),
        (2, "MEDIAPIPE POSE",        "33 body landmarks extracted → mapped to 15 JHMDB joints",  colors.HexColor("#00838F")),
        (3, "JHMDB JOINT MAPPING",   "neck · belly · face · shoulders · hips · elbows · knees · wrists · ankles", colors.HexColor("#006064")),
        (4, "FEATURE EXTRACTION",    "592 hierarchical features: angles · distances · symmetry · velocity · ROM", colors.HexColor("#1B5E20")),
        (5, "ML CLASSIFICATION",     "SVM (RBF C=50) + SelectKBest(200) → 21 action classes, 84.1% accuracy", colors.HexColor("#1565C0")),
        (6, "SHAP EXPLAINABILITY",   "Multiclass SHAP (K×N×d tensor) → per-joint importance heatmap",          PURPLE),
        (7, "CPG ENGINE",            "Counterfactual Pose Guidance: L-BFGS-B → minimal joint corrections",     colors.HexColor("#880E4F")),
        (8, "DOMAIN MODULE",         "Routes output to: Medical · Sports · Gait · Ergonomics · Adaptive Care", ORANGE),
        (9, "LLM COMMENTARY",        "Groq llama-3.3-70b generates clinical/coaching narrative (streamed)",    colors.HexColor("#4A148C")),
        (10,"USER FEEDBACK",         "Compliance %, Form score, Risk level, Discharge index, Rep count",       GREEN),
    ]

    for i, (num, title, sub, col) in enumerate(steps):
        story.append(flow_box(num, title, sub, col))
        if i < len(steps) - 1:
            story.append(arrow())

    sp(0.4)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 2 — FEATURE ENGINEERING + MODELS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story.append(section_header("FEATURE ENGINEERING — 592 HIERARCHICAL FEATURES", BLUE))
    sp(0.3)

    feat_data = [
        ["Feature Group", "Count", "What it captures"],
        ["Joint Angles", "28", "Hip, knee, elbow, shoulder (bilateral pairs)"],
        ["Bone Distances", "24", "Limb lengths, cross-body distances"],
        ["Bilateral Symmetry", "12", "L/R difference scores — pathology marker"],
        ["Body Ratios", "8", "Trunk-to-limb, torso proportions"],
        ["Velocity", "30", "Per-joint frame-to-frame delta"],
        ["Acceleration", "30", "Second-order temporal derivative"],
        ["ROM (clip-level)", "30", "Peak − min angle per joint across entire clip"],
        ["Spatial / BBox", "12", "Convex hull area, pose spread, centroid"],
        ["Aggregation ×4", "×4", "Each frame feature: mean · std · Q1 · Q3"],
        ["TOTAL", "~592", "SelectKBest(200) picks most discriminative subset"],
    ]
    story.append(data_table(
        feat_data[0], feat_data[1:],
        [5*cm, 2*cm, W-2*MARGIN-7.5*cm],
        header_bg=BLUE
    ))
    sp(0.4)

    story.append(section_header("MODEL COMPARISON — AUTO-SELECTION RESULTS", LBLUE))
    sp(0.3)

    model_data = [
        ["Model", "Accuracy", "Macro F1", "Part", "Notes"],
        ["LightGBM", "83.23%", "0.829", "Part 1", "Baseline — fast, high-dimensional"],
        ["XGBoost",  "82.18%", "0.819", "Part 1", "Robust to noise"],
        ["Random Forest", "79.4%", "0.791", "Part 2", "Low variance, OOB estimate"],
        ["SVM RBF C=50 ★", "84.1%", "0.838", "Part 2", "BEST — selected by auto-selector"],
        ["Soft Voting Ensemble", "~84.5%", "~0.842", "Part 2", "Top-3 combined (production)"],
    ]
    t = data_table(model_data[0], model_data[1:],
                   [4.5*cm, 2.5*cm, 2.5*cm, 2*cm, W-2*MARGIN-12*cm],
                   header_bg=LBLUE)
    # highlight best row
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,4),(-1,4), colors.HexColor("#E8F5E9")),
        ("FONTNAME", (0,4),(-1,4), "Helvetica-Bold"),
    ]))
    story.append(t)
    sp(0.3)

    story.append(section_header("FEATURE ABLATION — DROP IN ACCURACY WHEN GROUP REMOVED", colors.HexColor("#4527A0")))
    sp(0.3)
    ablation = [
        ["Feature Group Removed", "Accuracy Drop", "Importance"],
        ["Joint Angles",          "−6.8%",          "Critical"],
        ["Temporal Velocity",     "−4.2%",          "High"],
        ["ROM (clip-level)",      "−2.1%",          "Medium"],
        ["Bilateral Symmetry",    "−1.4%",          "Medium"],
        ["Body Ratios",           "−0.8%",          "Low"],
    ]
    story.append(data_table(ablation[0], ablation[1:],
                            [6*cm, 4*cm, W-2*MARGIN-10.5*cm],
                            header_bg=colors.HexColor("#4527A0")))
    sp(0.4)

    story.append(section_header("SHAP EXPLAINABILITY — FIXED MULTICLASS", PURPLE))
    sp(0.3)

    shap_rows = [
        ["Part 1 (WRONG)", "shap_values[0]", "Only class 0 — misleading"],
        ["Part 2 (CORRECT)", "mean(|shap_values|, axis=(0,1))", "All 21 classes aggregated"],
        ["Tensor shape", "(K=21, N=samples, d=200)", "Full multiclass SHAP"],
    ]
    story.append(data_table(["Version", "Code", "Effect"],
                             shap_rows,
                             [3*cm, 7*cm, W-2*MARGIN-10.5*cm],
                             header_bg=PURPLE))
    sp(0.3)

    shap_importance = [
        ["Feature Group", "SHAP Mass", "Interpretation"],
        ["Upper arm angles", "38%", "Primary discriminator for action type"],
        ["Temporal velocity", "27%", "Motion dynamics distinguish actions"],
        ["Hip / trunk angles", "18%", "Whole-body posture encoding"],
        ["Bilateral symmetry", "10%", "Asymmetry flags pathology / fatigue"],
        ["Spatial / BBox", "7%", "Global pose extent"],
    ]
    story.append(data_table(shap_importance[0], shap_importance[1:],
                            [4.5*cm, 3*cm, W-2*MARGIN-8*cm],
                            header_bg=PURPLE))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 3 — DOMAIN MODULES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story.append(section_header("8 APPLICATION DOMAINS — WHAT EACH MODULE DOES", NAVY))
    sp(0.3)

    domains = [
        ("Medical Assistant", "🏥", colors.HexColor("#0D47A1"),
         "12 clinical procedures (knee flexion 30/60/90°, PA chest X-ray, shoulder abduction, etc.) · "
         "Bilateral symmetry analysis · Cross-session ROM tracking · "
         "PDF session reports · Voice instructions (gTTS) · LLM commentary · SQLite patient history"),
        ("Sports Coach", "🏋️", colors.HexColor("#1B5E20"),
         "7 movements (squat, deadlift, golf, sprint, yoga, press, lunge) · "
         "Rep counting via angle state machine · Form score 0–100% · "
         "Angle trajectory plot · Key frame extraction (best/worst)"),
        ("Fitness Coach", "💪", colors.HexColor("#1A237E"),
         "6 gym exercises (push-up, squat, lunge, curl, plank, crunch) · "
         "Per-rep quality scoring (depth OK / full extension) · "
         "Live webcam rep counter · Streaming LLM coaching"),
        ("Ergonomics Monitor", "🖥️", colors.HexColor("#BF360C"),
         "RULA-proxy score (1–3 scale: Low / Medium / High risk) · "
         "8 postural checks (upper arm 30%, neck 30%, trunk 25%, wrist 15%) · "
         "Session timeline · Per-body-part injury guidance"),
        ("Gait Lab (HGD)", "🦶", colors.HexColor("#004D40"),
         "HGD: L1 joint angles → L2 gait events → L3 whole-body GDI · "
         "HKRA: kinematic chain root-cause detection (hip → knee → ankle) · "
         "PBRS: Bayesian injury risk from 5 factors with co-occurrence multipliers"),
        ("Adaptive Care Engine", "🧠", colors.HexColor("#4A148C"),
         "EWMA fault memory (α=0.40) · Mann-Kendall trend test (n=3–20) · "
         "Discharge Readiness Index (DRI ≥ 85% = discharge) · "
         "Auto rehab protocol generation per AAOS guidelines · 4-session outcome prediction"),
        ("AI Pose Coach", "🤸", colors.HexColor("#1B5E20"),
         "25-exercise catalogue (strength, flexibility, rehab, gait, medical positioning) · "
         "Dual animation: skeleton + photorealistic body (820×460 GIF, cosine easing) · "
         "LLM custom exercise: user describes movement → keyframes generated live"),
        ("Action Recognition", "🎬", colors.HexColor("#37474F"),
         "JHMDB 21-class classification (84.1% accuracy) · "
         ".mat file / video / webcam input · "
         "Top-3 confidence display · SHAP skeleton heatmap"),
    ]

    for icon_title, icon, color, desc in domains:
        story.append(domain_card(icon, icon_title, desc, color))
        sp(0.15)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 4 — NOVEL ALGORITHMS + CPG + RESULTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story.append(section_header("NOVEL ALGORITHMS — 5 IEEE-WORTHY CONTRIBUTIONS", NAVY))
    sp(0.3)

    novel = [
        ("1. Hierarchical Gait Decomposition (HGD)",
         colors.HexColor("#006064"),
         "3-tier analysis: joint angles (L1) → gait cycle events (L2) → whole-body GDI (L3). "
         "Camera-angle invariant — uses ROM not absolute angles. No force plates required."),
        ("2. EWMA + Mann-Kendall Adaptive Rehab Engine",
         colors.HexColor("#4A148C"),
         "Combines exponentially-weighted compliance memory (α=0.40) with "
         "Mann-Kendall non-parametric trend test to detect Improving / Stable / Regressing "
         "rehabilitation progress. Predicts discharge readiness (DRI ≥ 85%)."),
        ("3. Counterfactual Pose Guidance (CPG)",
         colors.HexColor("#880E4F"),
         "scipy.optimize.minimize (L-BFGS-B) finds the minimal feature perturbation that flips "
         "the ML prediction to a target class. Feature deltas mapped back to anatomical joint "
         "corrections: 'Rotate left shoulder 12° forward (currently 34°, target 46°)'."),
        ("4. Dual Animation + LLM Keyframe Generation",
         colors.HexColor("#1B5E20"),
         "Any described movement → LLM generates 15-joint keyframe coordinates → "
         "rendered as 820×460 GIF with cosine easing (left: skeleton, right: human body patches). "
         "Custom exercises synthesized in real-time."),
        ("5. Multi-Domain Pose Intelligence Framework",
         colors.HexColor("#0D47A1"),
         "Single hierarchical feature extractor powering 8 clinically-validated domains. "
         "What changes per domain: class definitions, reference poses, feedback templates, "
         "severity thresholds. Runs on CPU. Open-source. Streamlit-deployable."),
    ]

    for title, color, desc in novel:
        t = Table([[Paragraph(f"<b>{title}</b>",
                              style("nt", fontSize=9.5, textColor=WHITE,
                                    fontName="Helvetica-Bold")),
                    Paragraph(desc, style("nd", fontSize=8.5, textColor=WHITE,
                                         fontName="Helvetica", leading=13))]],
                  colWidths=[6*cm, W-2*MARGIN-6.5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(0,-1), color),
            ("BACKGROUND", (1,0),(-1,-1), colors.HexColor("#263238")),
            ("VALIGN", (0,0),(-1,-1), "TOP"),
            ("TOPPADDING",   (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ("RIGHTPADDING", (0,0),(-1,-1), 8),
            ("BOX", (0,0),(-1,-1), 0.3, colors.HexColor("#546E7A")),
        ]))
        story.append(t)
        sp(0.15)

    sp(0.3)
    story.append(section_header("STATE-OF-THE-ART COMPARISON", GRAY))
    sp(0.3)

    sota = [
        ["System", "Accuracy", "Explainable", "Multi-Domain", "Real-Time", "Clinical Output"],
        ["OpenPose + DNN", "78.3%", "No", "No", "Yes", "No"],
        ["VideoPose3D", "81.1%", "No", "No", "No", "No"],
        ["SlowFast (DL baseline)", "76.2%", "No", "No", "No", "No"],
        ["PoseFormer", "83.0%", "No", "No", "No", "No"],
        ["HierPose (Ours) ★", "84.1%", "Yes ✓", "8 domains ✓", "Yes ✓", "Yes ✓"],
    ]
    t = data_table(sota[0], sota[1:],
                   [4.5*cm, 2*cm, 2.5*cm, 2.5*cm, 2*cm, W-2*MARGIN-14*cm],
                   header_bg=GRAY)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,5),(-1,5), colors.HexColor("#E8F5E9")),
        ("FONTNAME", (0,5),(-1,5), "Helvetica-Bold"),
    ]))
    story.append(t)
    sp(0.3)

    story.append(section_header("PART 1 vs PART 2 — WHAT CHANGED", colors.HexColor("#37474F")))
    sp(0.3)

    compare = [
        ["Aspect", "Part 1", "Part 2"],
        ["Accuracy", "83.23% (LightGBM)", "84.1% (SVM RBF C=50)"],
        ["Features", "200 selected", "592 engineered → 200 selected"],
        ["Domains", "Action recognition only", "8 domains (medical, sports, gait, rehab, etc.)"],
        ["Explainability", "SHAP class-0 only (bug)", "Fixed multiclass SHAP (K,N,d) + CPG"],
        ["UI", "None", "9-page Streamlit app, role-based access"],
        ["LLM", "None", "Groq llama-3.3-70b, streaming, 5 modules"],
        ["Gait Analysis", "None", "HGD + HKRA + PBRS (3 novel algorithms)"],
        ["Rehab Tracking", "None", "EWMA + Mann-Kendall + Discharge Readiness Index"],
        ["Animation", "None", "Dual GIF: skeleton + photorealistic human"],
        ["Clinical Reports", "None", "Auto PDF + SQLite patient history"],
    ]
    story.append(data_table(compare[0], compare[1:],
                            [3.5*cm, 5*cm, W-2*MARGIN-9*cm],
                            header_bg=colors.HexColor("#37474F")))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PAGE 5 — QUICK DEMO GUIDE + CREDENTIALS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    story.append(PageBreak())
    story.append(section_header("DEMO GUIDE — RUN & SHOW", NAVY))
    sp(0.3)

    creds_data = [
        ["Username", "Password", "Role", "Modules Accessible"],
        ["demo", "demo", "Guest", "All modules (full access)"],
        ["hospital_admin", "med2025", "Medical", "Medical + Action + Ergonomics"],
        ["physio_user", "rehab123", "Physiotherapist", "Medical + Sports"],
        ["coach_user", "sport123", "Sports Coach", "Sports + Action"],
        ["safety_user", "safe2025", "Safety Officer", "Ergonomics + Action"],
    ]
    story.append(data_table(creds_data[0], creds_data[1:],
                            [3.5*cm, 3*cm, 4*cm, W-2*MARGIN-11*cm],
                            header_bg=NAVY))
    sp(0.3)

    story.append(section_header("3-MINUTE LIVE DEMO FLOW", LGREEN))
    sp(0.3)

    demo_steps = [
        ("1", "Start App", "streamlit run app/main.py   →   open http://localhost:8501", BLUE),
        ("2", "Login", "Username: demo  ·  Password: demo   →   full access granted", colors.HexColor("#006064")),
        ("3", "Sports Coach", "Upload squat video → view form score + rep count + angle trajectory", LGREEN),
        ("4", "Adaptive Care", "Enter 4–6 patient sessions → see EWMA trend + Discharge Readiness Index", PURPLE),
        ("5", "AI Pose Coach", "Search 'deadlift' → LLM generates instructions + dual animation GIF", ORANGE),
        ("6", "Gait Lab", "Upload walking video → view HGD 3-level decomposition + PBRS injury risk", colors.HexColor("#004D40")),
        ("7", "Medical", "Take webcam photo in PA chest position → CPG correction output", colors.HexColor("#880E4F")),
    ]

    for num, title, desc, color in demo_steps:
        t = Table([[Paragraph(f"<b>{num}</b>",
                              style("dn", fontSize=14, textColor=WHITE,
                                    fontName="Helvetica-Bold", alignment=TA_CENTER)),
                    Paragraph(f"<b>{title}</b>",
                              style("dt", fontSize=9.5, textColor=WHITE,
                                    fontName="Helvetica-Bold")),
                    Paragraph(desc, style("dd", fontSize=8.5, textColor=colors.HexColor("#B0BEC5"),
                                          fontName="Helvetica", leading=13))]],
                  colWidths=[1*cm, 3.5*cm, W-2*MARGIN-5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0),(0,-1), color),
            ("BACKGROUND", (1,0),(-1,-1), colors.HexColor("#1C2833")),
            ("VALIGN", (0,0),(-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0),(-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LEFTPADDING",  (0,0),(-1,-1), 8),
            ("RIGHTPADDING", (0,0),(-1,-1), 8),
            ("BOX", (0,0),(-1,-1), 0.3, colors.HexColor("#546E7A")),
        ]))
        story.append(t)
        sp(0.12)

    sp(0.4)
    story.append(section_header("WHAT TO SHOW ON EACH APP PAGE", colors.HexColor("#37474F")))
    sp(0.3)

    show_what = [
        ["Page", "Click Here", "What to Highlight"],
        ["Home", "Platform stats strip", "592+ features · 7 novel modules · role-based access"],
        ["Action Recognition", "Upload .mat or webcam", "84.1% accuracy · top-3 confidence · skeleton"],
        ["Medical Assistant", "Select 'Knee Flexion 90°' → webcam", "Compliance % · bilateral symmetry · PDF report"],
        ["Sports Coach", "Upload squat video", "Rep counter · form score · angle trajectory plot"],
        ["Ergonomics", "Webcam photo", "RULA score 1–3 · per-joint risk colour · session risk timeline"],
        ["Gait Lab", "Upload walking clip", "HGD 3-level · HKRA root cause · PBRS injury %"],
        ["Adaptive Care", "Enter 5 sessions manually", "EWMA trend · Mann-Kendall · Discharge Readiness gauge"],
        ["AI Pose Coach", "Search 'deadlift'", "LLM step-by-step + dual animated GIF auto-generated"],
    ]
    story.append(data_table(show_what[0], show_what[1:],
                            [3.5*cm, 5*cm, W-2*MARGIN-9*cm],
                            header_bg=colors.HexColor("#37474F")))

    sp(0.4)
    # Footer
    footer = Table([[
        Paragraph("Chaitanya Bharathi Institute of Technology (Autonomous) · Dept. of AIML · PID: AIML/2025-26/PID-2.1",
                  style("f", fontSize=7.5, textColor=colors.HexColor("#78909C"),
                        fontName="Helvetica", alignment=TA_CENTER)),
        Paragraph("K. Likhitha Reddy · M. Devansh · Guide: Dr. Y. Rama Devi · April 2026",
                  style("f2", fontSize=7.5, textColor=colors.HexColor("#78909C"),
                        fontName="Helvetica", alignment=TA_CENTER)),
    ]], colWidths=[(W-2*MARGIN)/2]*2)
    footer.setStyle(TableStyle([
        ("TOPPADDING",  (0,0),(-1,-1), 6),
        ("LINEABOVE", (0,0),(-1,0), 0.5, colors.HexColor("#CFD8DC")),
    ]))
    story.append(footer)

    doc.build(story)
    print("Generated: HierPose_Flow.pdf")

if __name__ == "__main__":
    build()
