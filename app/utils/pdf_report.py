"""PDF report generation for PoseAI assessments using fpdf2."""
from __future__ import annotations
import io, base64
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np


def generate_pdf(
    patient: Dict,
    procedure: str,
    department: str,
    evaluations: List[Dict],
    compliance: float,
    annotated_frame: Optional[np.ndarray] = None,
    symmetry_data: Optional[List[Dict]] = None,
    rom_data: Optional[List[Dict]] = None,
    session_notes: str = "",
    clinician: str = "",
) -> bytes:
    """Generate a clinical assessment PDF. Returns raw bytes."""
    from fpdf import FPDF

    def _safe(text: str) -> str:
        """Replace characters unsupported by built-in PDF fonts."""
        return (text
            .replace("\u2014", "-")   # em dash
            .replace("\u2013", "-")   # en dash
            .replace("\u2019", "'")   # right single quote
            .replace("\u2018", "'")   # left single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u2022", "*")   # bullet
            .replace("\u00b0", " deg") # degree sign (just in case)
        )

    class PDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_fill_color(30, 60, 100)
            self.set_text_color(255, 255, 255)
            self.rect(0, 0, 210, 20, "F")
            self.set_xy(10, 5)
            self.cell(0, 10, "PoseAI - Clinical Assessment Report", align="L")
            self.set_text_color(0, 0, 0)
            self.ln(18)

        def footer(self):
            self.set_y(-12)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5,
                f"Generated {datetime.now().strftime('%d %b %Y %H:%M')} | Page {self.page_no()} | "
                "AI-assisted guidance only - does not replace clinical judgement.",
                align="C"
            )

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)

    # ── Patient info block ────────────────────────────────────────────────────
    pdf.set_fill_color(240, 245, 255)
    pdf.rect(10, pdf.get_y(), 190, 34, "F")
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Patient Information", ln=True)
    pdf.set_font("Helvetica", size=10)

    left_col = [
        ("Name",       _safe(patient.get("name", "N/A"))),
        ("Date of Birth", _safe(patient.get("dob", "N/A"))),
        ("Gender",     _safe(patient.get("gender", "N/A"))),
    ]
    right_col = [
        ("Condition",  _safe(patient.get("condition", "N/A"))),
        ("Hospital",   _safe(patient.get("hospital", "N/A"))),
        ("Clinician",  _safe(clinician or patient.get("doctor", "N/A"))),
    ]
    y_start = pdf.get_y()
    for (lbl, val) in left_col:
        pdf.set_x(12)
        pdf.set_font("Helvetica", "B", 9); pdf.cell(35, 5, lbl + ":")
        pdf.set_font("Helvetica", size=9); pdf.cell(60, 5, str(val), ln=True)
    pdf.set_y(y_start)
    for (lbl, val) in right_col:
        pdf.set_x(110)
        pdf.set_font("Helvetica", "B", 9); pdf.cell(35, 5, lbl + ":")
        pdf.set_font("Helvetica", size=9); pdf.cell(55, 5, str(val), ln=True)
    pdf.ln(4)

    # ── Assessment details ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Assessment Details", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(50, 6, "Procedure:"); pdf.cell(0, 6, _safe(procedure), ln=True)
    pdf.cell(50, 6, "Department:"); pdf.cell(0, 6, _safe(department), ln=True)
    pdf.cell(50, 6, "Date / Time:"); pdf.cell(0, 6, datetime.now().strftime("%d %b %Y  %H:%M"), ln=True)

    # Compliance score with colour
    status = "PASS" if compliance >= 75 else "BORDERLINE" if compliance >= 50 else "FAIL"
    r, g, b = (0, 160, 80) if compliance >= 75 else (200, 130, 0) if compliance >= 50 else (200, 40, 40)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(50, 8, "Compliance Score:")
    pdf.set_text_color(r, g, b)
    pdf.cell(0, 8, f"{compliance:.0f}%  -  {status}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # ── Annotated image ───────────────────────────────────────────────────────
    if annotated_frame is not None:
        try:
            import tempfile, cv2, os
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(tmp.name, bgr)
            tmp.close()
            # Fit in page width, max height 70mm
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Annotated Assessment Frame:", ln=True)
            img_w = min(120, 190)
            x_center = (210 - img_w) / 2
            pdf.image(tmp.name, x=x_center, w=img_w)
            os.unlink(tmp.name)
            pdf.ln(3)
        except Exception:
            pass

    # ── Joint angles table ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Joint Angle Analysis", ln=True)

    # Table header
    col_w = [48, 28, 28, 28, 28, 40]
    headers = ["Joint", "Current (°)", "Target (°)", "Deviation (°)", "Tolerance", "Status"]
    pdf.set_fill_color(30, 60, 100); pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 9)
    for w, h in zip(col_w, headers):
        pdf.cell(w, 7, h, border=1, fill=True, align="C")
    pdf.ln()
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Helvetica", size=9)
    for i, ev in enumerate(evaluations):
        fill = i % 2 == 0
        pdf.set_fill_color(248, 250, 255) if fill else pdf.set_fill_color(255, 255, 255)
        color = ev.get("_color", "green")
        if color == "green":   pdf.set_text_color(0, 140, 60)
        elif color == "orange": pdf.set_text_color(180, 100, 0)
        else:                   pdf.set_text_color(180, 30, 30)

        row = [
            _safe(ev.get("Joint", "")),
            str(ev.get("Current (°)", "")),
            str(ev.get("Target (°)", "")),
            f"{ev.get('Deviation (°)', 0):+.1f}",
            _safe(ev.get("Tolerance", "")),
            _safe(ev.get("Status", "").replace("✅","OK").replace("🟡","~").replace("🔴","!"))
        ]
        for w, val in zip(col_w, row):
            pdf.cell(w, 6, val, border=1, fill=fill, align="C")
        pdf.ln()
        pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # ── Correction recommendations ────────────────────────────────────────────
    bad = [e for e in evaluations if e.get("_color") != "green"]
    if bad:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Correction Instructions", ln=True)
        pdf.set_font("Helvetica", size=10)
        for ev in bad:
            pdf.set_text_color(180, 30, 30)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, f"  {_safe(ev['Joint'])}:", ln=True)
            pdf.set_text_color(60, 60, 60)
            pdf.set_font("Helvetica", size=9)
            instr = _safe(ev.get("Instruction", ""))
            pdf.multi_cell(0, 5, f"    {instr}")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # ── Bilateral symmetry ────────────────────────────────────────────────────
    if symmetry_data:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Bilateral Symmetry Analysis", ln=True)
        col_w2 = [50, 30, 30, 30, 60]
        headers2 = ["Joint Pair", "Right (°)", "Left (°)", "Asymmetry (°)", "Assessment"]
        pdf.set_fill_color(30, 60, 100); pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 9)
        for w, h in zip(col_w2, headers2):
            pdf.cell(w, 7, h, border=1, fill=True, align="C")
        pdf.ln(); pdf.set_text_color(0, 0, 0); pdf.set_font("Helvetica", size=9)
        for i, s in enumerate(symmetry_data):
            asym = s.get("asymmetry", 0)
            assessment = "Symmetric" if asym < 5 else "Mild asymmetry" if asym < 10 else "Significant asymmetry"
            fill = i % 2 == 0
            pdf.set_fill_color(248, 250, 255) if fill else pdf.set_fill_color(255, 255, 255)
            if asym >= 10: pdf.set_text_color(180, 30, 30)
            elif asym >= 5: pdf.set_text_color(180, 100, 0)
            else: pdf.set_text_color(0, 140, 60)
            for w, val in zip(col_w2, [_safe(s.get("joint_pair","")), f"{s.get('right_angle',0):.1f}",
                                        f"{s.get('left_angle',0):.1f}", f"{asym:.1f}", assessment]):
                pdf.cell(w, 6, val, border=1, fill=fill, align="C")
            pdf.ln(); pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

    # ── ROM history ───────────────────────────────────────────────────────────
    if rom_data:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Range of Motion History", ln=True)
        pdf.set_font("Helvetica", size=9)
        for joint_name in set(r["joint_name"] for r in rom_data):
            records = [r for r in rom_data if r["joint_name"] == joint_name]
            if records:
                target = records[-1].get("target", 0)
                latest = records[-1].get("max_angle", 0)
                first  = records[0].get("max_angle", 0)
                improvement = latest - first
                trend = f"+{improvement:.0f}°" if improvement > 0 else f"{improvement:.0f}°"
                color_trend = (0, 140, 60) if improvement > 0 else (180, 30, 30)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(60, 5, f"  {joint_name}:")
                pdf.set_font("Helvetica", size=9)
                pdf.cell(50, 5, f"Latest: {latest:.0f}° (target {target:.0f}°)")
                pdf.set_text_color(*color_trend)
                pdf.cell(0, 5, f"Trend: {trend} over {len(records)} sessions", ln=True)
                pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    # ── Session notes ─────────────────────────────────────────────────────────
    if session_notes.strip():
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Session Notes", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5, _safe(session_notes))
        pdf.ln(2)

    # ── Signature block ───────────────────────────────────────────────────────
    pdf.ln(5)
    pdf.set_draw_color(100, 100, 100)
    pdf.line(10, pdf.get_y(), 100, pdf.get_y())
    pdf.line(115, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("Helvetica", size=9); pdf.set_text_color(100, 100, 100)
    pdf.cell(95, 5, "Clinician Signature", align="C")
    pdf.cell(5, 5, "")
    pdf.cell(90, 5, "Date & Stamp", align="C")

    return bytes(pdf.output())
