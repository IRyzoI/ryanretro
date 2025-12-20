import os
import csv
import re
import time
import json
import shutil
import hashlib
import threading
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
import markdown
from fastapi import FastAPI, HTTPException, Response, Request, Header
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------------------
# FastAPI app + CORS
# --------------------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True, # Set to True to support the new Auth headers if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Config & Paths
# --------------------------------------------------------------------------------------
YT_API_KEY = os.getenv("YT_API_KEY")
DEFAULT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID") or "UCh9GxjM-FNuSWv7xqn3UKVw"
UA = {"User-Agent": "RyanRetro/1.0"}
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "admin123")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 24 * 3600
CACHE_VERSION = "v5"

REPO_DIR = os.path.dirname(__file__)
DEFAULT_DATA_DIR = os.path.join(REPO_DIR, "data")      # Repo-synced CSVs
DATA_DIR = os.getenv("DATA_DIR", "/data")             # Persistent volume (e.g. Railway)
os.makedirs(DATA_DIR, exist_ok=True)

# Chat & Password Storage
CHAT_FILE = os.path.join(DATA_DIR, "chat_history.json")
GAME_PASS_FILE = os.path.join(DATA_DIR, "game_password.txt")

PIPED_BASES = [
    "https://pipedapi.kavin.rocks/api/v1",
    "https://piped.video/api/v1",
    "https://piped.projectsegfau.lt/api/v1",
]

_locks: Dict[str, threading.Lock] = {}

# --------------------------------------------------------------------------------------
# Utility Helpers
# --------------------------------------------------------------------------------------
def _lock_for(key: str) -> threading.Lock:
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]

def _cache_file(key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def _cache_read(key: str, ttl: int = CACHE_TTL_SECONDS):
    path = _cache_file(key)
    try:
        st = os.stat(path)
        if time.time() - st.st_mtime <= ttl:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def _cache_read_any_age(key: str):
    path = _cache_file(key)
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _cache_write(key: str, data):
    path = _cache_file(key)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, path)

# --------------------------------------------------------------------------------------
# Chat & Password Logic
# --------------------------------------------------------------------------------------
def _get_game_password() -> str:
    if not os.path.exists(GAME_PASS_FILE):
        with open(GAME_PASS_FILE, "w", encoding="utf-8") as f:
            f.write("retro")
        return "retro"
    with open(GAME_PASS_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()

def _set_game_password(new_pass: str):
    with open(GAME_PASS_FILE, "w", encoding="utf-8") as f:
        f.write(new_pass.strip())

def _load_chat() -> List[Dict]:
    if not os.path.exists(CHAT_FILE): return []
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception: return []

def _save_chat(messages: List[Dict]):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

# --------------------------------------------------------------------------------------
# YouTube Data Fetchers (Core Logic)
# --------------------------------------------------------------------------------------
def _make_matcher(q: str, aliases: str):
    phrases = [p.strip().lower() for p in [q] + (aliases or "").split(",") if p.strip()]
    generic_tokens, disamb_tokens = set(), set()
    
    def _tokenize(s: str):
        for t in re.findall(r"[a-z0-9]+", s.lower()):
            if t.isdigit() or len(t) >= 2: yield t

    for ph in phrases:
        for t in _tokenize(ph):
            if any(ch.isdigit() for ch in t): disamb_tokens.add(t)
            else: generic_tokens.add(t)

    def match(title: str) -> bool:
        tl = (title or "").lower()
        if any(ph in tl for ph in phrases): return True
        gen_hits = sum(1 for t in generic_tokens if t in tl)
        if gen_hits >= 2:
            return (not disamb_tokens) or any(t in tl for t in disamb_tokens)
        return False
    return match

def _yt_api_list_uploads(channel_id: str, pages: int = 12) -> List[Dict[str, str]]:
    if not YT_API_KEY: return []
    with httpx.Client(timeout=10.0, headers=UA) as client:
        r = client.get("https://www.googleapis.com/youtube/v3/channels", 
                       params={"part": "contentDetails", "id": channel_id, "key": YT_API_KEY})
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items: return []
        uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        out, token, page_count = [], None, 0
        while page_count < max(1, pages):
            params = {"part": "snippet", "playlistId": uploads_id, "maxResults": 50, "key": YT_API_KEY}
            if token: params["pageToken"] = token
            rr = client.get("https://www.googleapis.com/youtube/v3/playlistItems", params=params)
            rr.raise_for_status()
            data = rr.json()
            for it in data.get("items", []):
                sn = it.get("snippet", {})
                vid = sn.get("resourceId", {}).get("videoId")
                if vid: out.append({"title": sn.get("title", ""), "videoId": vid, "publishedAt": sn.get("publishedAt", "")})
            token = data.get("nextPageToken")
            page_count += 1
            if not token: break
        return out

def _yt_api_search(channel_id: str, query: str, limit: int) -> List[Dict[str, str]]:
    if not YT_API_KEY: return []
    out, page_token, pages = [], None, 0
    with httpx.Client(timeout=10.0, headers=UA) as client:
        while len(out) < max(limit, 50) and pages < 5:
            params = {"part": "snippet", "channelId": channel_id, "q": query, "order": "date", "type": "video", "maxResults": 50, "key": YT_API_KEY}
            if page_token: params["pageToken"] = page_token
            r = client.get("https://www.googleapis.com/youtube/v3/search", params=params)
            r.raise_for_status()
            data = r.json()
            for it in data.get("items", []):
                vid = it.get("id", {}).get("videoId")
                if vid: out.append({"title": it.get("snippet", {}).get("title", ""), "videoId": vid})
            page_token = data.get("nextPageToken")
            if not page_token: break
            pages += 1
    return out

def _piped_get(path: str, params: Dict[str, Any] | None = None):
    import random
    bases = list(PIPED_BASES)
    random.shuffle(bases)
    for base in bases:
        try:
            with httpx.Client(timeout=6.0, headers=UA) as client:
                r = client.get(base + path, params=params)
                r.raise_for_status()
                return r.json()
        except Exception: continue
    return None

def _piped_channel_videos(channel_id: str, pages: int = 6) -> List[Dict[str, str]]:
    out, nextpage = [], None
    for _ in range(max(1, pages)):
        p = {"nextpage": nextpage} if nextpage else None
        data = _piped_get(f"/channel/{channel_id}/videos", params=p) or _piped_get(f"/channel/{channel_id}", params=p)
        if not data: break
        streams = data.get("videos") or data.get("relatedStreams") or []
        for s in streams:
            vid = s.get("videoId") or s.get("id")
            if vid: out.append({"title": s.get("title", ""), "videoId": vid})
        nextpage = data.get("nextpage")
        if not nextpage: break
    return out

def _piped_search_channel(channel_id: str, query: str, pages: int = 2) -> List[Dict[str, str]]:
    out, seen, nextpage = [], set(), None
    for _ in range(max(1, pages)):
        params = {"q": query, "channelId": channel_id, "region": "US"}
        if nextpage: params["nextpage"] = nextpage
        data = _piped_get("/search", params=params)
        if not data: break
        items = data.get("items") if isinstance(data, dict) else data
        if not items: break
        for s in items:
            vid = s.get("videoId") or s.get("id")
            if vid and vid not in seen:
                out.append({"title": s.get("title") or s.get("name") or "", "videoId": vid})
                seen.add(vid)
        if isinstance(data, dict):
            nextpage = data.get("nextpage")
            if not nextpage: break
        else: break
    return out

def _rss_latest(channel_id: str) -> List[Dict[str, str]]:
    rss = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    with httpx.Client(timeout=6.0, headers=UA) as client:
        r = client.get(rss)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
        out = []
        for entry in root.findall("atom:entry", ns):
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            vid = entry.findtext("yt:videoId", default="", namespaces=ns)
            pub = entry.findtext("atom:published", namespaces=ns) or entry.findtext("atom:updated", namespaces=ns)
            if vid: out.append({"title": title, "videoId": vid, "publishedAt": pub or ""})
        return out

def _fetch_videos(channel_id: str, q: str, aliases: str, limit: int) -> List[Dict[str, str]]:
    matcher = _make_matcher(q, aliases)
    results = []
    phrases = [p.strip() for p in [q] + (aliases or "").split(",") if p.strip()]

    # 1. Try API uploads
    try: uploads = _yt_api_list_uploads(channel_id) if YT_API_KEY else []
    except Exception: uploads = []
    results.extend([it for it in uploads if matcher(it["title"])])

    # 2. Try API Search
    if len(results) < limit and YT_API_KEY:
        try:
            for ph in phrases:
                hits = _yt_api_search(channel_id, ph, limit=100)
                results.extend([it for it in hits if matcher(it["title"])])
                if len(results) >= limit * 2: break
        except Exception: pass

    # 3. Fallback Piped
    if len(results) < limit:
        try:
            items = _piped_channel_videos(channel_id)
            results.extend([it for it in items if matcher(it["title"])])
            if len(results) < limit:
                for ph in phrases:
                    hits = _piped_search_channel(channel_id, ph)
                    results.extend([it for it in hits if matcher(it["title"])])
        except Exception: pass

    # 4. Final Fallback RSS
    if not results:
        try: results.extend([it for it in _rss_latest(channel_id) if matcher(it["title"])])
        except Exception: pass

    out, seen = [], set()
    for it in results:
        vid = it.get("videoId")
        if vid and vid not in seen:
            out.append({"title": it.get("title", ""), "videoId": vid})
            seen.add(vid)
    return out[:limit]

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
class ChatMessage(BaseModel):
    user: str = Field(..., min_length=1, max_length=20)
    text: str = Field(..., min_length=1, max_length=500)

class PasswordUpdate(BaseModel):
    password: str

class CompatSubmission(BaseModel):
    device: str = Field(..., min_length=1)
    chipset: str = Field(..., min_length=1)
    system: str  = Field(..., min_length=1)
    game: Optional[str] = None
    performance: Optional[str] = None
    driver: Optional[str] = None
    emulator: Optional[str] = None
    resolution: Optional[str] = None
    rom_region: Optional[str] = Field(None, alias="rom region")
    winlator_version: Optional[str] = Field(None, alias="winlator version")
    dx_wrapper: Optional[str] = Field(None, alias="dx wrapper")
    game_resolution: Optional[str] = Field(None, alias="game resolution")
    dxvk_version: Optional[str] = Field(None, alias="dxvk version")
    vkd3d_version: Optional[str] = Field(None, alias="vkd3d version")
    precompiled_shaders: Optional[str] = Field(None, alias="pre-compiled shaders")
    game_title_id: Optional[str] = Field(None, alias="game title id")
    notes: Optional[str] = None

class DealSubmission(BaseModel):
    item_name: str = Field(..., min_length=1)
    price: float
    currency: str = "USD"
    store_name: str
    link: str
    notes: Optional[str] = None

# --------------------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------------------

# --- YouTube ---
@app.get("/api/youtube/latest")
def youtube_latest(channel_id: str, q: str = "", aliases: str = "", limit: int = 18, force: int = 0):
    search_phrases = [p.strip().lower() for p in ([q] + aliases.split(',')) if p.strip()]
    key = f"{CACHE_VERSION}:yt-api:{channel_id}|q={q.strip().lower()}"
    def _filter(items):
        return [v for v in (items or []) if any(p in (v.get("title", "").lower()) for p in search_phrases)]
    if not force:
        if cached := _cache_read(key): return _filter(cached)[:limit]
    with _lock_for(key):
        try:
            fresh = _fetch_videos(channel_id, q, aliases, limit)
            _cache_write(key, fresh)
            return _filter(fresh)[:limit]
        except Exception:
            if stale := _cache_read_any_age(key): return _filter(stale)[:limit]
            raise

@app.get("/api/latest-videos")
def latest_videos(response: Response, channel_id: str = DEFAULT_CHANNEL_ID, limit: int = 3, force: int = 0):
    key = f"{CACHE_VERSION}:home-latest:{channel_id}"
    def _shape(items):
        data = [{"id": it["videoId"], "title": it.get("title", ""), "thumb": f"https://i.ytimg.com/vi/{it['videoId']}/hqdefault.jpg", "publishedAt": it.get("publishedAt", "")} for it in (items or [])][:limit]
        response.headers["Cache-Control"] = "no-store" if force else f"public, max-age={CACHE_TTL_SECONDS}"
        return data
    if not force:
        if cached := _cache_read(key): return _shape(cached)
    with _lock_for(key):
        try:
            items = _yt_api_list_uploads(channel_id, pages=3)
            _cache_write(key, items)
            return _shape(items)
        except Exception:
            try:
                items = _rss_latest(channel_id)
                _cache_write(key, items)
                return _shape(items)
            except Exception:
                return _shape(_cache_read_any_age(key))

# --- Chat & Auth ---
@app.get("/api/chat/messages")
def get_messages(x_auth: str = Header(None)):
    if not (x_auth == ADMIN_TOKEN or x_auth == _get_game_password()):
        raise HTTPException(status_code=401, detail="Invalid password")
    return {"messages": _load_chat()[-100:], "is_admin": (x_auth == ADMIN_TOKEN)}

@app.post("/api/chat/send")
def send_message(msg: ChatMessage, x_auth: str = Header(None)):
    is_admin = (x_auth == ADMIN_TOKEN)
    if not (is_admin or x_auth == _get_game_password()):
        raise HTTPException(status_code=401, detail="Invalid password")
    new_msg = {"id": str(uuid.uuid4()), "user": msg.user, "text": msg.text, "timestamp": datetime.utcnow().isoformat(), "is_admin": is_admin, "role_badge": "ADMIN" if is_admin else "PLAYER"}
    with _lock_for("chat_history"):
        chat = _load_chat()
        chat.append(new_msg)
        _save_chat(chat[-500:])
    return {"ok": True, "message": new_msg}

@app.delete("/api/chat/{msg_id}")
def delete_message(msg_id: str, x_auth: str = Header(None)):
    if x_auth != ADMIN_TOKEN: raise HTTPException(status_code=403)
    with _lock_for("chat_history"):
        chat = [m for m in _load_chat() if m["id"] != msg_id]
        _save_chat(chat)
    return {"ok": True}

@app.post("/api/admin/game-password")
def update_game_password(body: PasswordUpdate, x_auth: str = Header(None)):
    if x_auth != ADMIN_TOKEN: raise HTTPException(status_code=403)
    _set_game_password(body.password)
    return {"ok": True}

# --- Compatibility CSVs ---
SYSTEM_CSVS = {"switch": "switch.csv", "ps2": "ps2.csv", "ps3": "ps3.csv", "psvita": "psvita.csv", "winlator": "winlator.csv", "gamehub": "gamehub.csv", "wiiu": "wiiu.csv"}
REQUIRED_COLS = {"device", "chipset", "system", "date added"}

@app.get("/api/compatibility/{system_name}")
def get_compatibility_list(system_name: str):
    filename = f"{system_name}.csv"
    path = next((p for p in [os.path.join(DATA_DIR, filename), os.path.join(DEFAULT_DATA_DIR, filename)] if os.path.exists(p)), None)
    if not path: raise HTTPException(status_code=404)
    with open(path, mode="r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {"system": system_name, "count": len(rows), "rows": rows}

@app.post("/api/compat/submit")
def post_compat(sub: CompatSubmission):
    system = sub.system.strip().lower()
    if system not in SYSTEM_CSVS: raise HTTPException(status_code=400)
    file_path = os.path.join(DATA_DIR, SYSTEM_CSVS[system])
    row = sub.model_dump(by_alias=True, exclude_none=True)
    row["date added"] = datetime.utcnow().strftime("%Y/%m/%d")
    
    fieldnames = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            fieldnames = list(csv.DictReader(f).fieldnames or [])
    
    if not fieldnames:
        fieldnames = ["device", "chipset", "system", "game", "performance", "driver", "emulator", "resolution", "rom region", "winlator version", "dx wrapper", "game resolution", "dxvk version", "vkd3d version", "pre-compiled shaders", "game title id", "notes", "date added"]
    
    for k in row.keys():
        if k not in fieldnames: fieldnames.append(k)

    with _lock_for(file_path):
        write_h = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_h: writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return {"ok": True}

@app.get("/api/deals")
def get_deals():
    path = os.path.join(DATA_DIR, "deals.csv")
    if not os.path.exists(path): return {"rows": []}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {"rows": rows[::-1]}

# --------------------------------------------------------------------------------------
# Page Routes
# --------------------------------------------------------------------------------------
@app.get("/", response_class=FileResponse)
def root(): return FileResponse(os.path.join(REPO_DIR, "static", "index.html"))

@app.get("/handheld", response_class=FileResponse)
def handheld_page(): return FileResponse(os.path.join(REPO_DIR, "static", "handheld.html"))

@app.get("/benchmarks", response_class=FileResponse)
def benchmarks_page(): return FileResponse(os.path.join(REPO_DIR, "static", "benchmarks.html"))

@app.get("/store", response_class=FileResponse)
def store_page(): return FileResponse(os.path.join(REPO_DIR, "static", "shop.html"))

@app.get("/gameoftheweek", response_class=FileResponse)
def gotw_page(): return FileResponse(os.path.join(REPO_DIR, "static", "gameoftheweek.html"))

@app.get("/guides/{slug}", response_class=HTMLResponse)
def serve_guide(slug: str):
    md_path = os.path.join("guides", f"{slug}.md")
    if not os.path.exists(md_path): return HTMLResponse("Not Found", 404)
    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    
    html = markdown.markdown(md_content, extensions=["fenced_code", "tables"])
    # (Optional: Re-insert your CSS/Iframe wrap logic here as per the original serve_guide)
    return HTMLResponse(f"<html><body>{html}</body></html>")

# --------------------------------------------------------------------------------------
# Admin & Seeding
# --------------------------------------------------------------------------------------
@app.get("/admin/seed-data")
def admin_seed_data(token: str = ""):
    if token != os.getenv("SEED_TOKEN"): raise HTTPException(status_code=403)
    copied = []
    for name in os.listdir(DEFAULT_DATA_DIR):
        if name.endswith(".csv"):
            shutil.copyfile(os.path.join(DEFAULT_DATA_DIR, name), os.path.join(DATA_DIR, name))
            copied.append(name)
    return {"ok": True, "copied": copied}

# --------------------------------------------------------------------------------------
# Mounts & Main
# --------------------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=os.path.join(REPO_DIR, "static")), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
