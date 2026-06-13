# =====================================================================
# Ryan Retro Creator Planner — sync API (FastAPI + Postgres via psycopg2)
#
# Storage: your Railway Postgres (private; survives deploys).
# NOT the volume — app.py serves the volume publicly at /data.
#
# Setup recap:
#   app.py (after `app = FastAPI()` / CORS):
#       from planner_api import router as planner_router
#       app.include_router(planner_router, prefix="/api/planner")
#   Railway vars on ryanretro service:
#       PLANNER_TOKEN = long random string
#       DATABASE_URL  = reference to Postgres.DATABASE_URL
# =====================================================================

import os
import re
import threading
import traceback

import psycopg2
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter()

TOKEN = os.getenv("PLANNER_TOKEN", "")
DB_URL = os.getenv("DATABASE_URL", "")

SAFE_ID = re.compile(r"^[a-zA-Z0-9]+$")
_table_ready = False
_table_lock = threading.Lock()


def _connect():
    """Open a connection. Railway's external proxy host needs SSL; the
    internal host does not. Try plain first, fall back to sslmode=require."""
    last = None
    for kwargs in ({}, {"sslmode": "require"}):
        try:
            conn = psycopg2.connect(DB_URL, connect_timeout=8, **kwargs)
            conn.autocommit = True
            return conn
        except Exception as e:  # noqa
            last = e
    raise last


def _conn():
    if not DB_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")
    try:
        conn = _connect()
    except Exception as e:  # surface the real reason instead of a blank 500
        raise HTTPException(status_code=500, detail="db connect failed: " + str(e))

    global _table_ready
    if not _table_ready:
        with _table_lock:
            if not _table_ready:
                try:
                    with conn.cursor() as cur:
                        cur.execute(
                            "CREATE TABLE IF NOT EXISTS planner_kv ("
                            "  key TEXT PRIMARY KEY,"
                            "  value TEXT NOT NULL,"
                            "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                            ")"
                        )
                    _table_ready = True
                except Exception as e:
                    raise HTTPException(status_code=500, detail="create table failed: " + str(e))
    return conn


def _auth(authorization):
    if not TOKEN:
        raise HTTPException(status_code=500, detail="PLANNER_TOKEN is not configured")
    if authorization != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _get(key):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT value FROM planner_kv WHERE key = %s", (key,))
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        raise HTTPException(status_code=500, detail="db read failed: " + str(e))
    finally:
        conn.close()


def _put(key, value):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO planner_kv (key, value, updated_at) VALUES (%s, %s, now()) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                (key, value),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail="db write failed: " + str(e))
    finally:
        conn.close()


def _delete(key):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM planner_kv WHERE key = %s", (key,))
    except Exception as e:
        raise HTTPException(status_code=500, detail="db delete failed: " + str(e))
    finally:
        conn.close()


class Payload(BaseModel):
    value: str


# ---- diagnostics: open this in a browser to see the real status ----
@router.get("/health")
def health():
    info = {
        "token_set": bool(TOKEN),
        "database_url_set": bool(DB_URL),
        "db_host": "",
        "connect_ok": False,
        "table_ok": False,
        "error": "",
    }
    if DB_URL:
        m = re.search(r"@([^:/]+)", DB_URL)
        info["db_host"] = m.group(1) if m else "?"
    try:
        conn = _connect()
        info["connect_ok"] = True
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS planner_kv ("
                "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
                "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
        info["table_ok"] = True
        conn.close()
    except Exception:
        info["error"] = traceback.format_exc().splitlines()[-1]
    return info


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
