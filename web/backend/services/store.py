"""SQLite-backed scan history and daily stats."""

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import List

from web.backend.models import DailyStats, PatientSummary, ScanRecord

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "scans.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            patient_name TEXT NOT NULL DEFAULT '',
            suspicion TEXT NOT NULL,
            confidence REAL NOT NULL,
            detections_count INTEGER NOT NULL,
            modality TEXT NOT NULL,
            turnaround_s REAL NOT NULL
        )
        """
    )
    # Migration: add patient_name if missing
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(scans)").fetchall()]
    if "patient_name" not in cols:
        conn.execute("ALTER TABLE scans ADD COLUMN patient_name TEXT NOT NULL DEFAULT ''")
    conn.commit()
    conn.close()


def save_scan(
    filename: str,
    suspicion: str,
    confidence: float,
    detections_count: int,
    modality: str,
    turnaround_s: float,
    patient_name: str = "",
) -> int:
    conn = _connect()
    cur = conn.execute(
        """
        INSERT INTO scans (timestamp, filename, patient_name, suspicion, confidence, detections_count, modality, turnaround_s)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (datetime.utcnow().isoformat(), filename, patient_name, suspicion, confidence, detections_count, modality, turnaround_s),
    )
    conn.commit()
    scan_id = cur.lastrowid
    conn.close()
    return scan_id


def get_history(limit: int = 50, offset: int = 0) -> List[ScanRecord]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM scans ORDER BY id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    conn.close()
    return [ScanRecord(**dict(r)) for r in rows]


def get_patient_scans(patient_name: str) -> List[ScanRecord]:
    conn = _connect()
    rows = conn.execute(
        "SELECT * FROM scans WHERE patient_name = ? ORDER BY id DESC",
        (patient_name,),
    ).fetchall()
    conn.close()
    return [ScanRecord(**dict(r)) for r in rows]


def get_patients() -> List[PatientSummary]:
    conn = _connect()
    rows = conn.execute(
        """
        SELECT patient_name, COUNT(*) as scan_count, MAX(timestamp) as last_scan
        FROM scans
        WHERE patient_name != ''
        GROUP BY patient_name
        ORDER BY last_scan DESC
        """
    ).fetchall()

    result = []
    for r in rows:
        worst = conn.execute(
            """
            SELECT suspicion FROM scans
            WHERE patient_name = ?
            ORDER BY CASE suspicion
                WHEN 'HIGH' THEN 1
                WHEN 'MODERATE' THEN 2
                WHEN 'REVIEW' THEN 3
                WHEN 'LOW' THEN 4
                ELSE 5
            END
            LIMIT 1
            """,
            (r["patient_name"],),
        ).fetchone()
        result.append(PatientSummary(
            name=r["patient_name"],
            scan_count=r["scan_count"],
            last_scan=r["last_scan"],
            worst_suspicion=worst["suspicion"] if worst else "LOW",
        ))

    conn.close()
    return result


def get_daily_stats() -> DailyStats:
    today = date.today().isoformat()
    conn = _connect()
    rows = conn.execute(
        "SELECT suspicion, turnaround_s FROM scans WHERE timestamp LIKE ?",
        (f"{today}%",),
    ).fetchall()
    conn.close()

    if not rows:
        return DailyStats()

    total = len(rows)
    high = sum(1 for r in rows if r["suspicion"] == "HIGH")
    review = sum(1 for r in rows if r["suspicion"] == "REVIEW")
    avg_turnaround = round(sum(r["turnaround_s"] for r in rows) / total, 2)
    return DailyStats(total=total, high=high, review=review, avg_turnaround=avg_turnaround)


def clear_history() -> int:
    conn = _connect()
    cur = conn.execute("DELETE FROM scans")
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count
