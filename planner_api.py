# =====================================================================
# Ryan Retro Creator Planner — sync API (FastAPI + Postgres via psycopg2)
#
# Storage: your Railway Postgres (private; survives deploys).
# NOT the volume — app.py serves the volume publicly at /data,
# so planner data there would be world-readable.
#
# Setup:
# 1. Save this file next to app.py as: planner_api.py
#    (uses psycopg2-binary, which is already in requirements.txt)
# 2. app.py: add near your other routes:
#
#      from planner_api import router as planner_router
#      app.include_router(planner_router, prefix="/api/planner")
#
#    and a page route ABOVE the /{product_id} catch-all:
#
#      @app.get("/planner", response_class=HTMLResponse)
#      async def planner_page():
#          return FileResponse(os.path.join(STATIC_DIR, "planner.html"))
#
# 3. Railway (ryanretro service -> Variables):
#      PLANNER_TOKEN = long random string (your devices' password)
#      DATABASE_URL  = add as a *reference* to Postgres.DATABASE_URL
#                      (New Variable -> Add Reference -> Postgres -> DATABASE_URL)
# =====================================================================

import os
import re
import threading

import psycopg2
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter()

TOKEN = os.getenv("PLANNER_TOKEN", "")
DB_URL = os.getenv("DATABASE_URL", "")

SAFE_ID = re.compile(r"^[a-zA-Z0-9]+$")
_table_ready = False
_table_lock = threading.Lock()


def _conn():
    if not DB_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")
    try:
        conn = psycopg2.connect(DB_URL, connect_timeout=8)
        conn.autocommit = True
    except Exception:
        raise HTTPException(status_code=500, detail="could not reach the database")
    global _table_ready
    if not _table_ready:
        with _table_lock:
            if not _table_ready:
                with conn.cursor() as cur:
                    cur.execute(
                        "CREATE TABLE IF NOT EXISTS planner_kv ("
                        "  key TEXT PRIMARY KEY,"
                        "  value TEXT NOT NULL,"
                        "  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                        ")"
                    )
                _table_ready = True
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
    finally:
        conn.close()


def _delete(key):
    conn = _conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM planner_kv WHERE key = %s", (key,))
    finally:
        conn.close()


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
    _put("state", body.value)
    return {"ok": True}


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
    _delete("img:" + img_id)  # deleting a missing key is fine
    return {"ok": True}
