import os
import re
from psycopg2 import pool
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter()

TOKEN = os.getenv("PLANNER_TOKEN", "")
DB_URL = os.getenv("DATABASE_URL", "")

# Fix: Allow hyphens and underscores for Claude-generated IDs
SAFE_ID = re.compile(r"^[a-zA-Z0-9\-_]+$")

_db_pool = None

def _get_pool():
    global _db_pool
    if not DB_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")
    if _db_pool is None:
        try:
            # Fix: Use a ThreadedConnectionPool to prevent Postgres connection exhaustion
            _db_pool = pool.ThreadedConnectionPool(1, 20, DB_URL, connect_timeout=8)
        except Exception:
            raise HTTPException(status_code=500, detail="could not reach the database")
    return _db_pool

def _ensure_table():
    db_pool = _get_pool()
    conn = db_pool.getconn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS planner_kv ("
                "  key TEXT PRIMARY KEY,"
                "  value TEXT NOT NULL,"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
    finally:
        db_pool.putconn(conn)

# Initialize table on startup safely
if DB_URL:
    try:
        _ensure_table()
    except Exception:
        pass

def _auth(authorization):
    if not TOKEN:
        raise HTTPException(status_code=500, detail="PLANNER_TOKEN is not configured")
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")

def _get(key):
    db_pool = _get_pool()
    conn = db_pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM planner_kv WHERE key = %s", (key,))
            row = cur.fetchone()
            return row[0] if row else None
    finally:
        db_pool.putconn(conn)

def _put(key, value):
    db_pool = _get_pool()
    conn = db_pool.getconn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO planner_kv (key, value, updated_at) VALUES (%s, %s, now()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, value),
            )
    finally:
        db_pool.putconn(conn)

def _delete(key):
    db_pool = _get_pool()
    conn = db_pool.getconn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM planner_kv WHERE key = %s", (key,))
    finally:
        db_pool.putconn(conn)

class Payload(BaseModel):
    value: str

# ---------------- state ----------------

@router.get("")
def get_state(authorization: str | None = Header(default=None)):
    _auth(authorization)
    return {"value": _get("state")}

@router.put("")
def put_state(body: Payload, authorization: str | None = Header(default=None)):
    _auth(authorization)
    current = _get("state")
    if current is not None and current != body.value:
        _put("state_prev", current)
    _put("state", body.value)
    return {"ok": True}

@router.get("/prev")
def get_prev_state(authorization: str | None = Header(default=None)):
    _auth(authorization)
    return {"value": _get("state_prev")}

# ---------------- images ----------------

@router.get("/img/{img_id}")
def get_img(img_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if not SAFE_ID.match(img_id):
        raise HTTPException(status_code=400, detail="bad id")
    value = _get("img:" + img_id)
    if value is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"value": value}

@router.put("/img/{img_id}")
def put_img(img_id: str, body: Payload, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if not SAFE_ID.match(img_id):
        raise HTTPException(status_code=400, detail="bad id")
    _put("img:" + img_id, body.value)
    return {"ok": True}

@router.delete("/img/{img_id}")
def delete_img(img_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if not SAFE_ID.match(img_id):
        raise HTTPException(status_code=400, detail="bad id")
    _delete("img:" + img_id)
    return {"ok": True}
