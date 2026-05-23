"""
Observability & Telemetry
Tracks every conversation turn in:
  - JSONL flat file  (easy to grep / ship to ELK)
  - SQLite database  (easy to query / build dashboards on top of)

Metrics collected per turn:
  - model name, latency, token estimate, safety decision
  - rolling counts: total turns, blocked turns, warned turns, errors
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config import config
from app.guardrails.safety import SafetyResult


#  Dataclasses 

@dataclass
class TurnRecord:
    record_id: str
    session_id: str
    turn_index: int
    model_type: str          # "oss" | "frontier"
    model_name: str
    user_message: str
    assistant_message: str
    latency_ms: float
    prompt_tokens_est: int
    completion_tokens_est: int
    safety_input_decision: str
    safety_output_decision: str
    tool_called: Optional[str]
    tool_result: Optional[str]
    error: Optional[str]
    timestamp: str


#  Token estimator (simple heuristic — no tokenizer dependency) 

def estimate_tokens(text: str) -> int:
    """Rough estimate: ~4 chars per token (GPT-style)."""
    return max(1, len(text) // 4)


#  JSONL writer 

class JSONLWriter:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: TurnRecord) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except Exception as exc:
            print(f"[Observability] JSONL write error: {exc}")


#  SQLite store 

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS turns (
    record_id              TEXT PRIMARY KEY,
    session_id             TEXT NOT NULL,
    turn_index             INTEGER NOT NULL,
    model_type             TEXT NOT NULL,
    model_name             TEXT NOT NULL,
    user_message           TEXT,
    assistant_message      TEXT,
    latency_ms             REAL,
    prompt_tokens_est      INTEGER,
    completion_tokens_est  INTEGER,
    safety_input_decision  TEXT,
    safety_output_decision TEXT,
    tool_called            TEXT,
    tool_result            TEXT,
    error                  TEXT,
    timestamp              TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    model_type   TEXT,
    created_at   TEXT,
    turn_count   INTEGER DEFAULT 0,
    error_count  INTEGER DEFAULT 0,
    block_count  INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_model   ON turns(model_type);
CREATE INDEX IF NOT EXISTS idx_turns_ts      ON turns(timestamp);
"""


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.executescript(_CREATE_SQL)
        self._conn.commit()

    def upsert_session(self, session_id: str, model_type: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, model_type, created_at) VALUES (?,?,?)",
            (session_id, model_type, datetime.now(timezone.utc).isoformat()),
        )
        self._conn.commit()

    def write_turn(self, record: TurnRecord) -> None:
        try:
            d = asdict(record)
            cols = ", ".join(d.keys())
            placeholders = ", ".join("?" for _ in d)
            self._conn.execute(
                f"INSERT OR REPLACE INTO turns ({cols}) VALUES ({placeholders})",
                list(d.values()),
            )
            # Update session counters
            self._conn.execute(
                "UPDATE sessions SET turn_count = turn_count + 1 WHERE session_id = ?",
                (record.session_id,),
            )
            if record.error:
                self._conn.execute(
                    "UPDATE sessions SET error_count = error_count + 1 WHERE session_id = ?",
                    (record.session_id,),
                )
            if "blocked" in (record.safety_input_decision or ""):
                self._conn.execute(
                    "UPDATE sessions SET block_count = block_count + 1 WHERE session_id = ?",
                    (record.session_id,),
                )
            self._conn.commit()
        except Exception as exc:
            print(f"[Observability] SQLite write error: {exc}")

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def summary_stats(self) -> dict:
        rows = self.query("""
            SELECT
                model_type,
                COUNT(*) AS total_turns,
                ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                SUM(CASE WHEN safety_input_decision = 'blocked' THEN 1 ELSE 0 END) AS blocked_inputs,
                SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                ROUND(AVG(completion_tokens_est), 1) AS avg_completion_tokens
            FROM turns
            GROUP BY model_type
        """)
        return {r["model_type"]: r for r in rows}


#  Tracker (main public interface) 

class Tracker:
    """
    Central observability tracker.
    Instantiate once and call `record_turn()` from each assistant.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            self._jsonl = JSONLWriter(config.LOG_FILE)
            self._db = SQLiteStore(config.DB_FILE)
        self._session_turns: dict[str, int] = {}

    def start_session(self, session_id: str, model_type: str) -> None:
        """Call when a new conversation begins."""
        if self.enabled:
            self._db.upsert_session(session_id, model_type)

    def record_turn(
        self,
        *,
        session_id: str,
        model_type: str,
        model_name: str,
        user_message: str,
        assistant_message: str,
        latency_ms: float,
        safety_input: Optional[SafetyResult] = None,
        safety_output: Optional[SafetyResult] = None,
        tool_called: Optional[str] = None,
        tool_result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> TurnRecord:
        turn_index = self._session_turns.get(session_id, 0)
        self._session_turns[session_id] = turn_index + 1

        record = TurnRecord(
            record_id=str(uuid.uuid4()),
            session_id=session_id,
            turn_index=turn_index,
            model_type=model_type,
            model_name=model_name,
            user_message=user_message,
            assistant_message=assistant_message,
            latency_ms=latency_ms,
            prompt_tokens_est=estimate_tokens(user_message),
            completion_tokens_est=estimate_tokens(assistant_message),
            safety_input_decision=(safety_input.decision.value if safety_input else "safe"),
            safety_output_decision=(safety_output.decision.value if safety_output else "safe"),
            tool_called=tool_called,
            tool_result=tool_result,
            error=error,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        if self.enabled:
            self._jsonl.write(record)
            self._db.write_turn(record)

        return record

    def get_stats(self) -> dict:
        if not self.enabled:
            return {}
        return self._db.summary_stats()


# Module-level singleton
tracker = Tracker(enabled=config.ENABLE_LOGGING)
