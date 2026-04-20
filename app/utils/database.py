"""SQLite patient & session database for PoseAI."""
from __future__ import annotations
import sqlite3, json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

DB_PATH = Path("app/data/poseai.db")

def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with _conn() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            dob         TEXT,
            gender      TEXT,
            condition   TEXT,
            hospital    TEXT,
            doctor      TEXT,
            notes       TEXT,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER NOT NULL,
            procedure       TEXT NOT NULL,
            department      TEXT,
            compliance      REAL,
            overall_status  TEXT,
            joint_data      TEXT,       -- JSON: list of eval dicts
            frame_count     INTEGER,
            notes           TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );

        CREATE TABLE IF NOT EXISTS rom_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  INTEGER NOT NULL,
            procedure   TEXT,
            joint_name  TEXT,
            min_angle   REAL,
            max_angle   REAL,
            target      REAL,
            session_id  INTEGER,
            recorded_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );

        CREATE TABLE IF NOT EXISTS symmetry_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id  INTEGER NOT NULL,
            session_id  INTEGER,
            joint_pair  TEXT,
            right_angle REAL,
            left_angle  REAL,
            asymmetry   REAL,
            recorded_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(patient_id) REFERENCES patients(id)
        );
        """)

# ── Patients ──────────────────────────────────────────────────────────────────

def add_patient(name, dob, gender, condition, hospital, doctor, notes="") -> int:
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO patients(name,dob,gender,condition,hospital,doctor,notes) VALUES(?,?,?,?,?,?,?)",
            (name, dob, gender, condition, hospital, doctor, notes)
        )
        return cur.lastrowid

def get_patients(hospital: Optional[str] = None) -> List[Dict]:
    with _conn() as con:
        if hospital:
            rows = con.execute("SELECT * FROM patients WHERE hospital=? ORDER BY name", (hospital,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM patients ORDER BY name").fetchall()
        return [dict(r) for r in rows]

def get_patient(patient_id: int) -> Optional[Dict]:
    with _conn() as con:
        row = con.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        return dict(row) if row else None

def update_patient_notes(patient_id: int, notes: str):
    with _conn() as con:
        con.execute("UPDATE patients SET notes=? WHERE id=?", (notes, patient_id))

# ── Sessions ──────────────────────────────────────────────────────────────────

def save_session(patient_id, procedure, department, compliance,
                 overall_status, joint_data, frame_count, notes="") -> int:
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO sessions
               (patient_id,procedure,department,compliance,overall_status,
                joint_data,frame_count,notes)
               VALUES(?,?,?,?,?,?,?,?)""",
            (patient_id, procedure, department, round(compliance, 1),
             overall_status, json.dumps(joint_data), frame_count, notes)
        )
        return cur.lastrowid

def get_sessions(patient_id: int) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM sessions WHERE patient_id=? ORDER BY created_at DESC",
            (patient_id,)
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["joint_data"] = json.loads(d["joint_data"] or "[]")
            result.append(d)
        return result

def get_all_sessions(hospital: Optional[str] = None) -> List[Dict]:
    with _conn() as con:
        query = """SELECT s.*, p.name as patient_name, p.condition, p.hospital
                   FROM sessions s JOIN patients p ON s.patient_id=p.id"""
        if hospital:
            query += " WHERE p.hospital=?"
            rows = con.execute(query + " ORDER BY s.created_at DESC", (hospital,)).fetchall()
        else:
            rows = con.execute(query + " ORDER BY s.created_at DESC").fetchall()
        return [dict(r) for r in rows]

# ── ROM History ───────────────────────────────────────────────────────────────

def save_rom(patient_id, procedure, joint_name, min_angle,
             max_angle, target, session_id=None):
    with _conn() as con:
        con.execute(
            """INSERT INTO rom_history
               (patient_id,procedure,joint_name,min_angle,max_angle,target,session_id)
               VALUES(?,?,?,?,?,?,?)""",
            (patient_id, procedure, joint_name,
             round(min_angle, 1), round(max_angle, 1), target, session_id)
        )

def get_rom_history(patient_id: int, joint_name: Optional[str] = None,
                    procedure: Optional[str] = None) -> List[Dict]:
    with _conn() as con:
        q = "SELECT * FROM rom_history WHERE patient_id=?"
        params: list = [patient_id]
        if joint_name:
            q += " AND joint_name=?"; params.append(joint_name)
        if procedure:
            q += " AND procedure=?"; params.append(procedure)
        q += " ORDER BY recorded_at ASC"
        return [dict(r) for r in con.execute(q, params).fetchall()]

# ── Symmetry Log ──────────────────────────────────────────────────────────────

def save_symmetry(patient_id, session_id, joint_pair,
                  right_angle, left_angle, asymmetry):
    with _conn() as con:
        con.execute(
            """INSERT INTO symmetry_log
               (patient_id,session_id,joint_pair,right_angle,left_angle,asymmetry)
               VALUES(?,?,?,?,?,?)""",
            (patient_id, session_id, joint_pair,
             round(right_angle, 1), round(left_angle, 1), round(asymmetry, 1))
        )

def get_symmetry_history(patient_id: int) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM symmetry_log WHERE patient_id=? ORDER BY recorded_at",
            (patient_id,)
        ).fetchall()
        return [dict(r) for r in rows]

# ── Dashboard stats ───────────────────────────────────────────────────────────

def dashboard_stats(hospital: Optional[str] = None) -> Dict[str, Any]:
    with _conn() as con:
        if hospital:
            n_pat = con.execute("SELECT COUNT(*) FROM patients WHERE hospital=?", (hospital,)).fetchone()[0]
            n_ses = con.execute(
                "SELECT COUNT(*) FROM sessions s JOIN patients p ON s.patient_id=p.id WHERE p.hospital=?",
                (hospital,)
            ).fetchone()[0]
            avg_c = con.execute(
                "SELECT AVG(s.compliance) FROM sessions s JOIN patients p ON s.patient_id=p.id WHERE p.hospital=?",
                (hospital,)
            ).fetchone()[0] or 0
        else:
            n_pat = con.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            n_ses = con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            avg_c = con.execute("SELECT AVG(compliance) FROM sessions").fetchone()[0] or 0
        return {"patients": n_pat, "sessions": n_ses, "avg_compliance": round(avg_c, 1)}

# Initialise on import
init_db()
