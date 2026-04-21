"""
database.py
-----------
SQLite database for storing prediction history.

Tables:
  predictions — every prediction made through the web app
"""

import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path("data/predictions.db")


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                filename        TEXT,
                top_prediction  TEXT    NOT NULL,
                confidence      REAL    NOT NULL,
                top2            TEXT,
                top3            TEXT,
                conf2           REAL,
                conf3           REAL
            )
        """)
        conn.commit()


def save_prediction(
    filename: str,
    predictions: List[Dict],
) -> int:
    """
    Save a prediction result to the database.

    Parameters
    ----------
    filename : str
        Uploaded image filename.
    predictions : list of dicts
        Output from the model — list of {class, confidence, rank}.

    Returns
    -------
    int — row id of the inserted record.
    """
    top  = predictions[0] if len(predictions) > 0 else {}
    top2 = predictions[1] if len(predictions) > 1 else {}
    top3 = predictions[2] if len(predictions) > 2 else {}

    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO predictions
                (timestamp, filename, top_prediction, confidence, top2, top3, conf2, conf3)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().isoformat(timespec="seconds"),
            filename,
            top.get("class", ""),
            top.get("confidence", 0.0),
            top2.get("class", ""),
            top3.get("class", ""),
            top2.get("confidence", 0.0),
            top3.get("confidence", 0.0),
        ))
        conn.commit()
        return cursor.lastrowid


def get_history(limit: int = 50) -> List[Dict]:
    """Return the last `limit` predictions, newest first."""
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT * FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> Dict:
    """Return summary statistics."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        by_class = conn.execute("""
            SELECT top_prediction, COUNT(*) as count
            FROM predictions
            GROUP BY top_prediction
            ORDER BY count DESC
        """).fetchall()
        avg_conf = conn.execute(
            "SELECT AVG(confidence) FROM predictions"
        ).fetchone()[0]

    return {
        "total_predictions": total,
        "avg_confidence":    round(avg_conf * 100, 1) if avg_conf else 0,
        "by_class":          [dict(r) for r in by_class],
    }
