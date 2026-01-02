import os
import csv
import re
import time
import json
import shutil
import hashlib
import threading
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
import markdown
from fastapi import FastAPI, HTTPException, Response, Request, Header
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
YT_API_KEY = os.getenv("YT_API_KEY")
DEFAULT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID") or "UCh9GxjM-FNuSWv7xqn3UKVw"
UA = {"User-Agent": "RyanRetro/1.0"}

# This token works for both Debug tools and Admin
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 24 * 3600
CACHE_VERSION = "v5"

# Piped mirrors
PIPED_BASES = [
    "https://pipedapi.kavin.rocks/api/v1",
    "https://piped.video/api/v1",
    "https://piped.projectsegfau.lt/api/v1",
]

_locks: Dict[str, threading.Lock] = {}

# --------------------------------------------------------------------------------------
# Storage Paths (Smart Logic for Mac vs Railway)
# --------------------------------------------------------------------------------------
REPO_DIR = os.path.dirname(__file__)
DEFAULT_DATA_DIR = os.path.join(REPO_DIR, "data")

# If DATA_DIR is set in env (Railway), use it. Otherwise use local repo folder (Mac).
DATA_DIR = os.getenv("DATA_DIR", os.path.join(REPO_DIR, "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# --------------------------------------------------------------------------------------
# Caching Logic
# --------------------------------------------------------------------------------------
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
    except (FileNotFoundError, Exception):
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

def _lock_for(key: str) -> threading.Lock:
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]

# --------------------------------------------------------------------------------------
# YouTube Helpers
# --------------------------------------------------------------------------------------
def _make_matcher(q: str, aliases: str):
    import re as _re
    phrases = []
    if q: phrases.append(q.strip().lower())
    for ph in (aliases or "").split(","):
        ph = ph.strip().lower()
        if ph: phrases.append(ph)

    # If no query/aliases provided, return a matcher that accepts everything.
    if not phrases:
        return lambda title: True

    generic_tokens, disamb_tokens = set(), set()
    def _tokenize(s):
        for t in _re.findall(r"[a-z0-9]+", s.lower()):
            if t.isdigit(): yield t
            elif len(t) >= 2: yield t

    for ph in phrases:
        for t in _tokenize(ph):
            if any(ch.isdigit() for ch in t): disamb_tokens.add(t)
            else: generic_tokens.add(t)

    def match(title: str) -> bool:
        tl = (title or "").lower()
        if any(ph and ph in tl for ph in phrases): return True
        gen_hits = sum(1 for t in generic_tokens if t in tl)
        if gen_hits >= 2:
            return (not disamb_tokens) or any(t in tl for t in disamb_tokens)
        return False
    return match

def _yt_api_list_uploads(channel_id: str, pages: int = 12) -> List[Dict[str, str]]:
    if not YT_API_KEY: return []
    with httpx.Client(timeout=10.0, headers=UA) as client:
        try:
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
                    sn = it.get("snippet") or {}
                    vid = sn.get("resourceId", {}).get("videoId")
                    if vid:
                        out.append({"title": sn.get("title", ""), "videoId": vid, "publishedAt": sn.get("publishedAt", "")})
                token = data.get("nextPageToken")
                page_count += 1
                if not token: break
            return out
        except Exception as e:
            print(f"[yt] api error: {e}")
            return []

def _yt_api_search(channel_id: str, query: str, limit: int) -> List[Dict[str, str]]:
    if not YT_API_KEY: return []
    out, pages, page_token = [], 0, None
    with httpx.Client(timeout=10.0, headers=UA) as client:
        while len(out) < max(limit, 50) and pages < 5:
            params = {"part": "snippet", "channelId": channel_id, "q": query, "order": "date", "type": "video", "maxResults": 50, "key": YT_API_KEY}
            if page_token: params["pageToken"] = page_token
            try:
                r = client.get("https://www.googleapis.com/youtube/v3/search", params=params)
                r.raise_for_status()
                data = r.json()
                for it in data.get("items", []):
                    vid = it.get("id", {}).get("videoId")
                    if vid: out.append({"title": it.get("snippet", {}).get("title", ""), "videoId": vid})
                page_token = data.get("nextPageToken")
                if not page_token: break
                pages += 1
            except Exception:
                break
    return out

def _piped_get(path: str, params=None):
    import random
    random.shuffle(PIPED_BASES)
    for base in PIPED_BASES:
        try:
            with httpx.Client(timeout=6.0, headers=UA) as client:
                r = client.get(base + path, params=params)
            r.raise_for_status()
            return r.json()
        except: continue
    return None

def _piped_channel_videos(channel_id: str, pages: int = 6):
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

def _piped_search_channel(channel_id: str, query: str, pages: int = 2):
    out, seen, nextpage = [], set(), None
    for _ in range(max(1, pages)):
        p = {"q": query, "channelId": channel_id, "region": "US"}
        if nextpage: p["nextpage"] = nextpage
        data = _piped_get("/search", params=p)
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

def _rss_latest(channel_id: str):
    try:
        with httpx.Client(timeout=6.0, headers=UA) as client:
            r = client.get(f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}")
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
        out = []
        for entry in root.findall("atom:entry", ns):
            out.append({
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip(),
                "videoId": entry.findtext("yt:videoId", default="", namespaces=ns),
                "publishedAt": entry.findtext("atom:published", default="", namespaces=ns) or ""
            })
        return out
    except Exception as e:
        print(f"[rss] error: {e}")
        return []

def _fetch_videos(channel_id: str, q: str, aliases: str, limit: int):
    matcher = _make_matcher(q, aliases)
    results = []
    
    # 1. API Uploads
    try:
        if YT_API_KEY:
            uploads = _yt_api_list_uploads(channel_id)
            results.extend([it for it in uploads if matcher(it["title"])])
    except: pass

    # 2. API Search
    if len(results) < limit and YT_API_KEY:
        try:
            phrases = [p.strip() for p in ([q] + aliases.split(",")) if p.strip()]
            for ph in phrases:
                hits = _yt_api_search(channel_id, ph, 100)
                results.extend([it for it in hits if matcher(it["title"])])
                if len(results) >= limit * 2: break
        except: pass

    # 3. Piped / RSS Fallback
    if len(results) < limit:
        try:
            items = _piped_channel_videos(channel_id)
            results.extend([it for it in items if matcher(it["title"])])
        except: pass
        if len(results) == 0:
            items = _rss_latest(channel_id)
            results.extend([it for it in items if matcher(it["title"])])

    # Dedupe
    out, seen = [], set()
    for it in results:
        vid = it.get("videoId")
        if vid and vid not in seen:
            out.append(it)
            seen.add(vid)
    return out[:limit]

# --------------------------------------------------------------------------------------
# YouTube Endpoints
# --------------------------------------------------------------------------------------
@app.get("/api/youtube/latest")
def youtube_latest(channel_id: str, q: str = "", aliases: str = "", limit: int = 18, force: int = 0):
    key = f"{CACHE_VERSION}:yt-api:{channel_id}|q={q.strip().lower()}"
    
    search_phrases = [p.strip().lower() for p in ([q] + aliases.split(',')) if p.strip()]
    
    # If no search phrases, return everything. Don't filter out results.
    def _filter(items):
        if not items: return []
        if not search_phrases: return items
        return [v for v in items if any(p in v.get("title","").lower() for p in search_phrases)]

    if not force:
        cached = _cache_read(key)
        if cached: return _filter(cached)[:limit]

    with _lock_for(key):
        if not force:
            cached = _cache_read(key)
            if cached: return _filter(cached)[:limit]
        try:
            fresh = _fetch_videos(channel_id, q, aliases, limit)
            _cache_write(key, fresh)
            return _filter(fresh)[:limit]
        except Exception:
            stale = _cache_read_any_age(key)
            if stale: return _filter(stale)[:limit]
            raise

@app.get("/api/latest-videos")
def latest_videos(response: Response, channel_id: str = DEFAULT_CHANNEL_ID, limit: int = 3, force: int = 0):
    key = f"{CACHE_VERSION}:home-latest:{channel_id}"
    
    def _shape(items):
        data = [{
            "id": it["videoId"],
            "title": it.get("title", ""),
            "thumb": f"https://i.ytimg.com/vi/{it['videoId']}/hqdefault.jpg",
            "publishedAt": it.get("publishedAt", "")
        } for it in (items or [])][:limit]
        response.headers["Cache-Control"] = "no-store" if force else f"public, max-age={CACHE_TTL_SECONDS}"
        return data

    if not force:
        cached = _cache_read(key)
        if cached: return _shape(cached)

    with _lock_for(key):
        if not force:
            cached = _cache_read(key)
            if cached: return _shape(cached)
        try:
            items = _yt_api_list_uploads(channel_id, pages=3)
            _cache_write(key, items)
            return _shape(items)
        except:
            items = _rss_latest(channel_id)
            _cache_write(key, items)
            return _shape(items)

# --------------------------------------------------------------------------------------
# Compatibility Data
# --------------------------------------------------------------------------------------
SYSTEM_CSVS = {
    "switch": "switch.csv",
    "ps2": "ps2.csv",
    "ps3": "ps3.csv",
    "psvita": "psvita.csv",
    "winlator": "winlator.csv",
    "gamehub": "gamehub.csv",
    "wiiu": "wiiu.csv",
}
REQUIRED_COLS = {"device", "chipset", "system", "date added"}

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

def _find_csv_path(system_name: str) -> str | None:
    filename = f"{system_name}.csv"
    p1 = os.path.join(DATA_DIR, filename)
    if os.path.exists(p1): return p1
    p2 = os.path.join(DEFAULT_DATA_DIR, filename)
    if os.path.exists(p2): return p2
    return None

def _list_csvs(folder: str) -> List[str]:
    if not os.path.isdir(folder): return []
    return sorted([n for n in os.listdir(folder) if n.lower().endswith(".csv")])

# Seeding Logic
if not any(name.lower().endswith(".csv") for name in os.listdir(DATA_DIR) or []):
    if os.path.isdir(DEFAULT_DATA_DIR):
        for name in _list_csvs(DEFAULT_DATA_DIR):
            shutil.copyfile(os.path.join(DEFAULT_DATA_DIR, name), os.path.join(DATA_DIR, name))

@app.get("/api/compatibility/{system_name}")
def get_compatibility_list(system_name: str):
    path = _find_csv_path(system_name)
    if not path: raise HTTPException(status_code=404, detail="System not found")
    rows = []
    with open(path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader: rows.append(row)
    return {"system": system_name, "count": len(rows), "rows": rows}

@app.post("/api/compat/submit")
def post_compat(sub: CompatSubmission):
    system = (sub.system or "").strip().lower()
    if system not in SYSTEM_CSVS:
        raise HTTPException(status_code=400, detail=f"Unknown system '{sub.system}'")
    
    file_path = os.path.join(DATA_DIR, SYSTEM_CSVS[system])
    row = sub.model_dump(by_alias=True, exclude_none=True)
    
    # Normalize aliases
    if "rom_region" in row: row["rom region"] = row.pop("rom_region")
    if "winlator_version" in row: row["winlator version"] = row.pop("winlator_version")
    if "dx_wrapper" in row: row["dx wrapper"] = row.pop("dx_wrapper")
    
    row["system"] = system
    row["date added"] = datetime.utcnow().strftime("%Y/%m/%d")

    fieldnames = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            fieldnames = list(csv.DictReader(f).fieldnames or [])
    
    if not fieldnames:
        fieldnames = ["device", "chipset", "system", "game", "performance", "driver", "emulator", "notes", "date added"]
        fieldnames += [k for k in row.keys() if k not in fieldnames]
    else:
        new_cols = [k for k in row.keys() if k not in fieldnames]
        if new_cols: fieldnames += new_cols

    with _lock_for(file_path):
        write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header: writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in fieldnames})
            
    return {"ok": True}

# --------------------------------------------------------------------------------------
# Admin / Debug
# --------------------------------------------------------------------------------------
@app.get("/api/debug/where")
def debug_where(request: Request):
    if request.query_params.get("token") != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    return {"repo_dir": REPO_DIR, "data_dir": DATA_DIR, "files": os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []}

@app.get("/admin/seed-data")
def admin_seed_data(request: Request, force: int = 0, token: str = ""):
    need = os.getenv("SEED_TOKEN", "")
    if not need or token != need:
        # Fallback to ADMIN_TOKEN if SEED_TOKEN not set
        if not ADMIN_TOKEN or token != ADMIN_TOKEN:
            raise HTTPException(status_code=403, detail="Forbidden")
            
    copied = []
    if os.path.isdir(DEFAULT_DATA_DIR):
        for name in _list_csvs(DEFAULT_DATA_DIR):
            src, dst = os.path.join(DEFAULT_DATA_DIR, name), os.path.join(DATA_DIR, name)
            if force or not os.path.exists(dst):
                shutil.copyfile(src, dst)
                copied.append(name)
    return {"ok": True, "copied": copied}

# --------------------------------------------------------------------------------------
# Page Routes & Static
# --------------------------------------------------------------------------------------

# 1. Initialize Templates (Do this before defining routes that use it)
templates = Jinja2Templates(directory="templates")

# 2. Define Main Routes (Using Templates)
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "active_page": "home"
    })

@app.get("/handheld", response_class=HTMLResponse)
async def handheld_page(request: Request):
    return templates.TemplateResponse("handheld.html", {
        "request": request, 
        "active_page": "handheld"
    })

@app.get("/patrons", response_class=HTMLResponse)
async def patrons_page(request: Request):
    return templates.TemplateResponse("patrons.html", {
        "request": request, 
        "active_page": "patrons"
    })

@app.get("/gameoftheweek", response_class=HTMLResponse)
@app.get("/gotw", response_class=HTMLResponse)
async def gotw_page(request: Request):
    return templates.TemplateResponse("gotw.html", {
        "request": request, 
        "active_page": "gotw"
    })

@app.get("/benchmarks", response_class=HTMLResponse)
async def benchmarks_page(request: Request):
    return templates.TemplateResponse("benchmarks.html", {
        "request": request, 
        "active_page": "benchmarks"
    })

@app.get("/reviews/retroid-pocket-g2", response_class=HTMLResponse)
async def rpg2_review_page(request: Request):
    return templates.TemplateResponse("rpg2-review.html", {
        "request": request,
        "active_page": "handheld"
    })

@app.get("/store", response_class=HTMLResponse)
async def store_page(request: Request):
    return templates.TemplateResponse("store.html", {
        "request": request, 
        "active_page": "store"
    })

@app.get("/ranking", response_class=FileResponse)
def ranking_page(): 
    return FileResponse(os.path.join(REPO_DIR, "static", "ranking.html"))

# 4. Helper Routes (Guides)
@app.get("/guides/{slug}", response_class=HTMLResponse)
def serve_guide(slug: str):
    md_path = os.path.join("guides", f"{slug}.md")
    if not os.path.exists(md_path):
        return HTMLResponse("<h1>Guide not found</h1>", status_code=404)
    with open(md_path, "r", encoding="utf-8") as f: content = f.read()
    html = markdown.markdown(content, extensions=["fenced_code", "tables"])
    # Simple wrapper for Markdown content
    return f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"><title>{slug}</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>body {{ background: #111827; color: white; }} a {{ color: #facc15; }}</style>
    </head><body class="p-8"><article class="prose prose-invert mx-auto">{html}</article></body></html>
    """

# 5. Mount Static Files
app.mount("/static", StaticFiles(directory=os.path.join(REPO_DIR, "static")), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# --------------------------------------------------------------------------------------
# Pretty URL Routing (Dynamic Handheld/Accessory IDs)
# --------------------------------------------------------------------------------------
# This set matches the IDs in your store.html script
KNOWN_IDS = {
    "anbernic_rg477v", "trimui_brick_hammer", "trimui_brick", "rp6",
    "steam_deck", "rp5", "rp_mini2", "rp_classic", "rp_flip2", "rpg2",
    "miyoo_mini_v4", "miyoo_mini_plus", "odin3", "odin2_portal", "ayn_thor",
    "anker_power", "ugreen_power", "ugreen_dock", "rp_ds_addon", 
    "gamesir_g8p", "sd_card"
}

# 6. The "Catch-All" Route (MUST BE LAST)
@app.get("/{product_id}", response_class=HTMLResponse)
async def serve_pretty_product_url(request: Request, product_id: str):
    # Check if the URL matches a known product ID
    if product_id in KNOWN_IDS:
        return templates.TemplateResponse("store.html", {
            "request": request, 
            "active_page": "store"
        })
    
    # If not found, raise 404
    raise HTTPException(status_code=404, detail="Page not found")

@app.get("/test-store", response_class=HTMLResponse)
async def test_store(request: Request):
    return templates.TemplateResponse("store.html", {
        "request": request, 
        "active_page": "store"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
