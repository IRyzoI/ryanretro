import os
import csv
import re
import time
import json
import base64
import shutil
import hashlib
import threading
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from datetime import datetime

import httpx
import markdown
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Creator Planner sync API (stores data in Postgres)
from planner_api import router as planner_router
app.include_router(planner_router, prefix="/api/planner")

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
STATIC_DIR = os.path.join(REPO_DIR, "static")

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

def _find_csv_path(system_name: str) -> Optional[str]:
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
# Page Routes & Static (SERVED AS DIRECT FILES)
# --------------------------------------------------------------------------------------

# 1. Mount Static Files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

# 2. Main Page Routes (Returning FileResponse)
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/handheld", response_class=HTMLResponse)
async def handheld_page():
    return FileResponse(os.path.join(STATIC_DIR, "handheld.html"))

@app.get("/patrons", response_class=HTMLResponse)
async def patrons_page():
    return FileResponse(os.path.join(STATIC_DIR, "patrons.html"))

@app.get("/gameoftheweek", response_class=HTMLResponse)
@app.get("/gotw", response_class=HTMLResponse)
async def gotw_page():
    return FileResponse(os.path.join(STATIC_DIR, "gotw.html"))

@app.get("/benchmarks", response_class=HTMLResponse)
async def benchmarks_page():
    return FileResponse(os.path.join(STATIC_DIR, "benchmarks.html"))

@app.get("/reviews/retroid-pocket-g2", response_class=HTMLResponse)
async def rpg2_review_page():
    # Maps to the specific HTML file for this review
    return FileResponse(os.path.join(STATIC_DIR, "rpg2review.html"))

@app.get("/store", response_class=HTMLResponse)
async def store_page():
    # Pointing to shop.html as per your file structure
    return FileResponse(os.path.join(STATIC_DIR, "shop.html"))

@app.get("/ranking", response_class=FileResponse)
def ranking_page(): 
    return FileResponse(os.path.join(STATIC_DIR, "ranking.html"))

@app.get("/planner", response_class=HTMLResponse)
async def planner_page():
    return FileResponse(os.path.join(STATIC_DIR, "planner.html"))

@app.get("/start", response_class=HTMLResponse)
async def start_page():
    return FileResponse(os.path.join(STATIC_DIR, "start.html"))

@app.get("/thumbnail", response_class=FileResponse)
def ranking_page(): 
    return FileResponse(os.path.join(STATIC_DIR, "thumbnail.html"))

@app.get("/reviews", response_class=FileResponse)
def ranking_page():
    return FileResponse(os.path.join(STATIC_DIR, "reviews.html"))

@app.get("/news", response_class=HTMLResponse)
async def news_page():
    return FileResponse(os.path.join(STATIC_DIR, "news.html"))

@app.get("/deals", response_class=HTMLResponse)
async def deals_page():
    return FileResponse(os.path.join(STATIC_DIR, "deals.html"))

@app.post("/api/guide/{slug}/save")
async def save_guide(slug: str, request: Request):
    data = await request.json()
    key = data.get("key", "")
    content = data.get("content", "")

    edit_key = os.getenv("GUIDE_EDIT_KEY", "ryanretro")
    if key != edit_key:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not re.match(r'^[a-z0-9-]+$', slug):
        return JSONResponse({"error": "Invalid slug"}, status_code=400)

    guide_path = os.path.join(REPO_DIR, "guides", f"{slug}.html")
    if not os.path.exists(guide_path):
        return JSONResponse({"error": "Guide not found"}, status_code=404)

    with open(guide_path, "r", encoding="utf-8") as f:
        html = f.read()

    start_marker = "<!-- EDITABLE-START -->"
    end_marker   = "<!-- EDITABLE-END -->"
    start_idx = html.find(start_marker)
    end_idx   = html.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        return JSONResponse({"error": "Editable markers not found"}, status_code=400)

    new_html = (
        html[:start_idx + len(start_marker)] +
        "\n  <div id=\"guide-cards\">\n" +
        content +
        "\n  </div><!-- /guide-cards -->\n  " +
        html[end_idx:]
    )

    with open(guide_path, "w", encoding="utf-8") as f:
        f.write(new_html)

    return JSONResponse({"ok": True})

# 3. Helper Routes (Guides)
@app.get("/guides/{slug}", response_class=HTMLResponse)
def serve_guide(slug: str):
    md_path = os.path.join("guides", f"{slug}.md")
    if not os.path.exists(md_path):
        # Fall back to a self-contained HTML guide if one exists.
        html_path = os.path.join("guides", f"{slug}.html")
        if os.path.exists(html_path):
            return FileResponse(html_path)
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

# 3b. Articles (new clean post-template style; stored in /articles)
@app.get("/articles/{slug}", response_class=HTMLResponse)
def serve_article(slug: str):
    html_path = os.path.join(REPO_DIR, "articles", f"{slug}.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return HTMLResponse("<h1>Article not found</h1>", status_code=404)

@app.post("/api/article/{slug}/save")
async def save_article(slug: str, request: Request):
    data = await request.json()
    key = data.get("key", "")
    content = data.get("content", "")

    edit_key = os.getenv("GUIDE_EDIT_KEY", "ryanretro")
    if key != edit_key:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    if not re.match(r'^[a-z0-9-]+$', slug):
        return JSONResponse({"error": "Invalid slug"}, status_code=400)

    article_path = os.path.join(REPO_DIR, "articles", f"{slug}.html")
    if not os.path.exists(article_path):
        return JSONResponse({"error": "Article not found"}, status_code=404)

    with open(article_path, "r", encoding="utf-8") as f:
        html = f.read()

    start_marker = "<!-- EDITABLE-START -->"
    end_marker   = "<!-- EDITABLE-END -->"
    start_idx = html.find(start_marker)
    end_idx   = html.find(end_marker)

    if start_idx == -1 or end_idx == -1:
        return JSONResponse({"error": "Editable markers not found"}, status_code=400)

    new_html = (
        html[:start_idx + len(start_marker)] +
        "\n  <div id=\"guide-cards\" class=\"post-body\">\n" +
        content +
        "\n  </div><!-- /guide-cards -->\n  " +
        html[end_idx:]
    )

    with open(article_path, "w", encoding="utf-8") as f:
        f.write(new_html)

    return JSONResponse({"ok": True})

# Image upload for the in-page article editor (saves into static/images/)
ALLOWED_IMG_EXT = {"png", "jpg", "jpeg", "gif", "webp", "avif", "svg"}

def _norm_img_ext(ext: str) -> str:
    ext = (ext or "").lower().lstrip(".").split("?")[0]
    if ext == "jpeg": return "jpg"
    if ext in ("svg+xml", "svg xml"): return "svg"
    return ext

@app.post("/api/upload-image")
async def upload_image(request: Request):
    data = await request.json()
    key      = data.get("key", "")
    name     = data.get("name", "")
    data_url = data.get("dataUrl", "")
    src_url  = data.get("url", "")

    edit_key = os.getenv("GUIDE_EDIT_KEY", "ryanretro")
    if key != edit_key:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    if not slug:
        return JSONResponse({"error": "Please give the image a name"}, status_code=400)

    raw, ext = None, None
    try:
        if data_url:
            m = re.match(r"data:(image/[\w.+-]+);base64,(.*)$", data_url, re.S)
            if not m:
                return JSONResponse({"error": "Invalid image data"}, status_code=400)
            ext = _norm_img_ext(m.group(1).split("/", 1)[1])
            raw = base64.b64decode(m.group(2))
        elif src_url:
            with httpx.Client(timeout=15.0, headers=UA, follow_redirects=True) as client:
                r = client.get(src_url)
                r.raise_for_status()
                raw = r.content
                ctype = r.headers.get("content-type", "").split(";")[0].strip()
                if ctype.startswith("image/"):
                    ext = _norm_img_ext(ctype.split("/", 1)[1])
                else:
                    ext = _norm_img_ext(os.path.splitext(src_url.split("?")[0])[1])
        else:
            return JSONResponse({"error": "Provide a file or an image URL"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": f"Could not read image: {e}"}, status_code=400)

    if ext not in ALLOWED_IMG_EXT:
        return JSONResponse({"error": f"Unsupported image type: .{ext or '?'}"}, status_code=400)
    if not raw:
        return JSONResponse({"error": "Image was empty"}, status_code=400)
    if len(raw) > 15 * 1024 * 1024:
        return JSONResponse({"error": "Image too large (max 15MB)"}, status_code=400)

    images_dir = os.path.join(STATIC_DIR, "images")
    os.makedirs(images_dir, exist_ok=True)
    filename = f"{slug}.{ext}"
    with open(os.path.join(images_dir, filename), "wb") as f:
        f.write(raw)

    return JSONResponse({"ok": True, "file": filename, "path": f"../static/images/{filename}"})

# --------------------------------------------------------------------------------------
# Box Art Grabber
#   Primary  : IGDB (Twitch) — official covers at their NATIVE aspect ratio (all platforms)
#   Extra    : SteamGridDB    — alternate capsule covers to cycle through
#   Fallback : Steam          — official PC library art
# --------------------------------------------------------------------------------------
SGDB_API_KEY = os.getenv("SGDB_API_KEY") or os.getenv("STEAMGRIDDB_API_KEY", "")
SGDB_BASE = "https://www.steamgriddb.com/api/v2"
# Portrait "cover/box art" dimension buckets on SteamGridDB.
SGDB_COVER_DIMS = "600x900,342x482,660x930"

# IGDB via Twitch OAuth (client-credentials). Create a free app at dev.twitch.tv.
IGDB_CLIENT_ID = os.getenv("IGDB_CLIENT_ID", "")
IGDB_CLIENT_SECRET = os.getenv("IGDB_CLIENT_SECRET", "")
_IGDB_TOKEN: Dict[str, Any] = {"token": None, "exp": 0}

# Domains the image proxy is allowed to fetch from (for ZIP downloads).
BOXART_ALLOWED_HOSTS = (
    "igdb.com", "images.igdb.com",
    "libretro.com", "thumbnails.libretro.com",
    "steamgriddb.com", "cdn2.steamgriddb.com", "cdn.steamgriddb.com",
    "steamstatic.com", "akamaihd.net", "steamcdn-a.akamaihd.net",
)

# IGDB platform name -> libretro-thumbnails system folder (real box scans, native ratios).
IGDB_TO_LIBRETRO = {
    "Game Boy Advance": "Nintendo - Game Boy Advance",
    "Game Boy Color": "Nintendo - Game Boy Color",
    "Game Boy": "Nintendo - Game Boy",
    "Nintendo 64": "Nintendo - Nintendo 64",
    "Super Nintendo Entertainment System": "Nintendo - Super Nintendo Entertainment System",
    "Super Famicom": "Nintendo - Super Nintendo Entertainment System",
    "Nintendo Entertainment System": "Nintendo - Nintendo Entertainment System",
    "Family Computer (FAMICOM)": "Nintendo - Nintendo Entertainment System",
    "Nintendo GameCube": "Nintendo - GameCube",
    "Nintendo DS": "Nintendo - Nintendo DS",
    "Nintendo 3DS": "Nintendo - Nintendo 3DS",
    "Wii": "Nintendo - Wii",
    "Wii U": "Nintendo - Wii U",
    "Virtual Boy": "Nintendo - Virtual Boy",
    "Sega Mega Drive/Genesis": "Sega - Mega Drive - Genesis",
    "Sega Master System/Mark III": "Sega - Master System - Mark III",
    "Sega Game Gear": "Sega - Game Gear",
    "Sega Saturn": "Sega - Saturn",
    "Dreamcast": "Sega - Dreamcast",
    "Sega CD": "Sega - Mega-CD - Sega CD",
    "Sega 32X": "Sega - 32X",
    "PlayStation": "Sony - PlayStation",
    "PlayStation 2": "Sony - PlayStation 2",
    "PlayStation Portable": "Sony - PlayStation Portable",
    "Atari 2600": "Atari - 2600",
    "Atari 7800": "Atari - 7800",
    "Atari Lynx": "Atari - Lynx",
    "TurboGrafx-16/PC Engine": "NEC - PC Engine - TurboGrafx 16",
    "Neo Geo Pocket Color": "SNK - Neo Geo Pocket Color",
    "WonderSwan Color": "Bandai - WonderSwan Color",
    "Xbox": "Microsoft - Xbox",
}

# On exact-name collisions across systems (e.g. an original + a later remake), prefer the
# more iconic/original hardware. Higher wins ties; this never overrides title/region scoring.
LIBRETRO_PRIORITY = {
    "Nintendo - Nintendo 64": 5, "Nintendo - Super Nintendo Entertainment System": 5,
    "Nintendo - Nintendo Entertainment System": 5, "Nintendo - GameCube": 5,
    "Sega - Mega Drive - Genesis": 5, "Sony - PlayStation": 5, "Sony - PlayStation 2": 5,
    "Sega - Saturn": 5, "Sega - Dreamcast": 5, "Microsoft - Xbox": 5,
    "Nintendo - Game Boy": 4, "Nintendo - Game Boy Color": 4, "Nintendo - Game Boy Advance": 4,
    "Sega - Master System - Mark III": 4, "NEC - PC Engine - TurboGrafx 16": 4,
    "Nintendo - Wii": 2, "Nintendo - Nintendo DS": 2, "Nintendo - Nintendo 3DS": 2,
    "Sony - PlayStation Portable": 2, "Nintendo - Wii U": 1,
}
def _sys_priority(system: str) -> int:
    return LIBRETRO_PRIORITY.get(system, 3)

# Short, friendly console label shown on each cover badge.
_SYS_LABEL = {
    "Nintendo - Super Nintendo Entertainment System": "SNES",
    "Nintendo - Nintendo Entertainment System": "NES",
    "Nintendo - Nintendo 64": "Nintendo 64", "Nintendo - GameCube": "GameCube",
    "Nintendo - Game Boy Advance": "Game Boy Advance", "Nintendo - Game Boy Color": "Game Boy Color",
    "Nintendo - Game Boy": "Game Boy", "Nintendo - Nintendo DS": "Nintendo DS",
    "Nintendo - Nintendo 3DS": "Nintendo 3DS", "Nintendo - Wii": "Wii",
    "Nintendo - Wii U": "Wii U", "Nintendo - Virtual Boy": "Virtual Boy",
    "Sega - Mega Drive - Genesis": "Genesis", "Sega - Master System - Mark III": "Master System",
    "Sega - Game Gear": "Game Gear", "Sega - Saturn": "Saturn", "Sega - Dreamcast": "Dreamcast",
    "Sega - Mega-CD - Sega CD": "Sega CD", "Sega - 32X": "32X",
    "Sony - PlayStation": "PlayStation", "Sony - PlayStation 2": "PlayStation 2",
    "Sony - PlayStation Portable": "PSP", "Microsoft - Xbox": "Xbox",
    "Atari - 2600": "Atari 2600", "Atari - 7800": "Atari 7800", "Atari - Lynx": "Lynx",
    "NEC - PC Engine - TurboGrafx 16": "TurboGrafx-16",
    "SNK - Neo Geo Pocket Color": "Neo Geo Pocket", "Bandai - WonderSwan Color": "WonderSwan",
}
def _system_label(system: str) -> str:
    return _SYS_LABEL.get(system, system.split(" - ")[-1])

def _sgdb_headers():
    return {"Authorization": f"Bearer {SGDB_API_KEY}", **UA}

def _sgdb_search(term: str) -> List[Dict[str, Any]]:
    """Fuzzy-search SteamGridDB for a game name -> list of {id, name}."""
    if not SGDB_API_KEY:
        return []
    from urllib.parse import quote
    try:
        with httpx.Client(timeout=12.0, headers=_sgdb_headers()) as c:
            r = c.get(f"{SGDB_BASE}/search/autocomplete/{quote(term.strip())}")
            r.raise_for_status()
            data = r.json().get("data", []) or []
        return [{"id": g.get("id"), "name": g.get("name", "")} for g in data if g.get("id")]
    except Exception as e:
        print(f"[boxart] sgdb search error: {e}")
        return []

def _sgdb_covers(game_id: int, limit: int = 12) -> List[Dict[str, Any]]:
    """Fetch portrait covers for a SteamGridDB game, best (largest + most upvoted) first."""
    if not SGDB_API_KEY:
        return []
    try:
        params = {"dimensions": SGDB_COVER_DIMS, "types": "static", "nsfw": "false"}
        with httpx.Client(timeout=12.0, headers=_sgdb_headers()) as c:
            r = c.get(f"{SGDB_BASE}/grids/game/{game_id}", params=params)
            r.raise_for_status()
            grids = r.json().get("data", []) or []
    except Exception as e:
        print(f"[boxart] sgdb covers error: {e}")
        return []
    out, seen = [], set()
    for g in grids:
        url = g.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append({
            "url": url,
            "thumb": g.get("thumb") or url,
            "w": g.get("width") or 0,
            "h": g.get("height") or 0,
            "score": g.get("upvotes", 0) or 0,
            "source": "steamgriddb",
        })
    # Largest resolution first, then most upvoted.
    out.sort(key=lambda x: (x["w"] * x["h"], x["score"]), reverse=True)
    return out[:limit]

def _steam_search(term: str, limit: int = 3) -> List[Dict[str, Any]]:
    try:
        with httpx.Client(timeout=10.0, headers=UA, follow_redirects=True) as c:
            r = c.get("https://store.steampowered.com/api/storesearch/",
                      params={"term": term.strip(), "cc": "us", "l": "en"})
            r.raise_for_status()
            items = r.json().get("items", []) or []
        return [{"appid": it.get("id"), "name": it.get("name", "")} for it in items[:limit] if it.get("id")]
    except Exception as e:
        print(f"[boxart] steam search error: {e}")
        return []

def _steam_cover(appid: int) -> Optional[Dict[str, Any]]:
    """Steam's official 1200x1800 library portrait, if the game has one."""
    url = f"https://steamcdn-a.akamaihd.net/steam/apps/{appid}/library_600x900_2x.jpg"
    try:
        with httpx.Client(timeout=8.0, headers=UA, follow_redirects=True) as c:
            r = c.head(url)
            if r.status_code != 200:
                return None
    except Exception:
        return None
    return {"url": url, "thumb": url, "w": 1200, "h": 1800, "score": 0, "source": "steam"}

# ---- IGDB (native-ratio official covers) ----
def _igdb_token() -> Optional[str]:
    """A valid Twitch app token, cached in memory + on disk (tokens last ~60 days)."""
    if not (IGDB_CLIENT_ID and IGDB_CLIENT_SECRET):
        return None
    now = time.time()
    if _IGDB_TOKEN.get("token") and _IGDB_TOKEN.get("exp", 0) - 120 > now:
        return _IGDB_TOKEN["token"]
    disk = _cache_read_any_age("igdb_token:v1")
    if disk and disk.get("exp", 0) - 120 > now:
        _IGDB_TOKEN.update(disk)
        return disk["token"]
    try:
        with httpx.Client(timeout=12.0) as c:
            r = c.post("https://id.twitch.tv/oauth2/token", params={
                "client_id": IGDB_CLIENT_ID, "client_secret": IGDB_CLIENT_SECRET,
                "grant_type": "client_credentials"})
            r.raise_for_status()
            d = r.json()
        tok = {"token": d["access_token"], "exp": now + int(d.get("expires_in", 5000000))}
        _IGDB_TOKEN.update(tok)
        _cache_write("igdb_token:v1", tok)
        return tok["token"]
    except Exception as e:
        print(f"[boxart] igdb token error: {e}")
        return None

def _igdb_query(body: str) -> List[Dict[str, Any]]:
    tok = _igdb_token()
    if not tok:
        return []
    headers = {"Client-ID": IGDB_CLIENT_ID, "Authorization": f"Bearer {tok}", **UA}
    try:
        with httpx.Client(timeout=12.0, headers=headers) as c:
            r = c.post("https://api.igdb.com/v4/games", content=body)
            r.raise_for_status()
            return r.json() or []
    except Exception as e:
        print(f"[boxart] igdb query error: {e}")
        return []

def _igdb_cover_obj(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a native-ratio cover from an IGDB game. t_1080p/t_720p preserve aspect ratio."""
    cov = g.get("cover") or {}
    iid = cov.get("image_id")
    if not iid:
        return None
    base = "https://images.igdb.com/igdb/image/upload"
    return {
        "url": f"{base}/t_1080p/{iid}.jpg",
        "thumb": f"{base}/t_720p/{iid}.jpg",
        "w": cov.get("width") or 0, "h": cov.get("height") or 0,
        "score": 0, "source": "igdb",
    }

_IGDB_FIELDS = ("name,cover.image_id,cover.width,cover.height,"
                "platforms.name,alternative_names.name,category,first_release_date")

def _igdb_search(term: str) -> List[Dict[str, Any]]:
    safe = term.replace('"', '\\"').strip()
    body = f'search "{safe}"; fields {_IGDB_FIELDS}; where cover != null; limit 15;'
    return _igdb_query(body)

def _igdb_game(gid: int) -> List[Dict[str, Any]]:
    body = f'fields {_IGDB_FIELDS}; where id = {int(gid)};'
    return _igdb_query(body)

# ---- libretro thumbnails (real box scans at native physical aspect ratios) ----
def _norm_title(s: str) -> str:
    """Normalize a title for fuzzy matching: drop region/rev tags, punctuation, leading 'the'."""
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"\([^)]*\)", " ", s)      # (USA), (Rev 1), ...
    s = re.sub(r"\[[^\]]*\]", " ", s)     # [!], [b], ...
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    # Drop "the" everywhere (titles flip it: "The Legend of Zelda" vs "Legend of Zelda, The").
    s = " ".join(t for t in s.split() if t != "the")
    return s

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")        # optional: lifts the 60/hr rate limit

def _libretro_system_files(system: str) -> List[str]:
    """All Named_Boxarts base filenames for a system (cached 7 days via GitHub tree API).
       Per-system locked so concurrent lookups fetch each tree only once."""
    key = f"libretro_files:{system}"
    cached = _cache_read(key, ttl=7 * 86400)
    if cached is not None:
        return cached
    with _lock_for(key):
        cached = _cache_read(key, ttl=7 * 86400)   # another thread may have just filled it
        if cached is not None:
            return cached
        from urllib.parse import quote
        repo = system.replace(" ", "_")
        url = f"https://api.github.com/repos/libretro-thumbnails/{quote(repo)}/git/trees/HEAD?recursive=1"
        headers = dict(UA)
        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
        try:
            with httpx.Client(timeout=25.0, headers=headers, follow_redirects=True) as c:
                r = c.get(url)
                r.raise_for_status()
                tree = r.json().get("tree", []) or []
            files = [t["path"][14:-4] for t in tree
                     if t.get("path", "").startswith("Named_Boxarts/")
                     and t["path"].lower().endswith(".png")]
            _cache_write(key, files)
            return files
        except Exception as e:
            print(f"[boxart] libretro tree error for {system}: {e}")
            stale = _cache_read_any_age(key)
            return stale if stale is not None else []

def _warm_libretro_cache():
    """Pre-fetch every system's box-art listing once (cached to disk) so the first user
       lookups don't race GitHub's rate limit. No-op when caches are already warm."""
    for system in sorted(set(IGDB_TO_LIBRETRO.values())):
        try:
            _libretro_system_files(system)
        except Exception:
            pass

threading.Thread(target=_warm_libretro_cache, daemon=True).start()

_LIBRETRO_BAD_TAGS = ("virtual console", "demo", "beta", "proto", "sample", "promo",
                      "kiosk", "(rev", "aftermarket", "unl", "pirate", "enhanced", "hack",
                      "mini", "classic", "collection", "all-stars", "anniversary", "switch online")

def _best_libretro_file(files: List[str], names: List[str]) -> tuple:
    """Best (filename, score) for the given candidate names. Title match dominates; region +
       'clean retail' cleanliness break ties (USA retail > JP / Virtual Console / revisions)."""
    targets = [frozenset(_norm_title(n).split()) for n in names]
    targets = [t for t in targets if t]
    # Collapsed (space-free) forms so "yugioh" matches "yu gi oh", "megaman" matches "mega man".
    ctargets = [c for c in (_norm_title(n).replace(" ", "") for n in names) if c]
    if not targets:
        return (None, 0)
    best, best_score = None, 0
    for f in files:
        nf = _norm_title(f)
        ft = frozenset(nf.split())
        cf = nf.replace(" ", "")
        if not ft:
            continue
        base = 0
        for tt in targets:
            if tt == ft:
                base = max(base, 1000)
            elif tt <= ft:                                   # file has all title words (+ "version" etc.)
                base = max(base, 820 - len(ft - tt) * 8)
            else:
                inter = tt & ft
                if len(inter) >= max(2, round(len(tt) * 0.8)) and len(tt - ft) <= 1:
                    base = max(base, 520 + len(inter) * 8 - len(ft - tt) * 4)
        for ct in ctargets:                                  # spacing/punctuation-insensitive match
            if ct == cf:
                base = max(base, 1000)
            elif len(ct) >= 8 and (cf.startswith(ct) or ct.startswith(cf)):
                base = max(base, 760)
        if base <= 0:
            continue
        low = f.lower()
        for i, reg in enumerate(["(usa", "(world", "(europe", "(japan"]):  # region preference
            if reg in low:
                base += 12 - i * 3
                break
        base -= sum(6 for t in _LIBRETRO_BAD_TAGS if t in low)            # non-retail penalty
        base -= max(0, f.count("(") - 1) * 2                              # prefer fewer tags
        if base > best_score:
            best_score, best = base, f
    return (best, best_score)

def _libretro_cover_obj(system: str, filename: str, name: str) -> Dict[str, Any]:
    from urllib.parse import quote
    url = (f"https://thumbnails.libretro.com/{quote(system)}"
           f"/Named_Boxarts/{quote(filename)}.png")
    return {"url": url, "thumb": url, "w": 0, "h": 0, "score": 0, "source": "libretro",
            "system": _system_label(system), "name": name}

def _pretty_libretro_name(cover: Dict[str, Any]) -> str:
    """A display title from a libretro filename: drop region/tags, flip 'Zelda, The' -> 'The Zelda'."""
    from urllib.parse import unquote
    fn = unquote(cover["url"].split("/Named_Boxarts/")[-1])[:-4]   # strip .png
    fn = re.sub(r"\s*\(.*$", "", fn).strip()                       # drop "(USA) ..." onward
    m = re.match(r"^(.*),\s*(The|A|An)$", fn)
    if m:
        fn = f"{m.group(2)} {m.group(1)}"
    return fn

def _match_in_system(system: str, names: List[str]) -> tuple:
    files = _libretro_system_files(system)
    if not files:
        return (None, 0)
    return _best_libretro_file(files, names)

def _libretro_covers(names: List[str], platform_names: List[str], limit: int = 8) -> List[Dict[str, Any]]:
    """Real box scans for the game across EVERY system it appears on (so the user can pick
       per-system), best match first. Trusts IGDB's platforms, then scans all systems for
       exact-title hits (IGDB mistags many retro originals' platforms)."""
    names = [n for n in names if n]
    if not names:
        return []
    igdb_systems = [IGDB_TO_LIBRETRO[p] for p in platform_names if p in IGDB_TO_LIBRETRO]
    hits, seen = [], set()  # hits: (score, priority, system, file)

    # 1. IGDB-declared platforms — trusted, so accept decent (>=700) matches.
    for system in igdb_systems:
        if system in seen:
            continue
        f, s = _match_in_system(system, names)
        if f and s >= 700:
            hits.append((s, _sys_priority(system), system, f))
            seen.add(system)

    # 2. All other systems — accept only exact-title (>=1000) matches to stay safe.
    for system in set(IGDB_TO_LIBRETRO.values()):
        if system in seen:
            continue
        f, s = _match_in_system(system, names)
        if f and s >= 1000:
            hits.append((s, _sys_priority(system), system, f))
            seen.add(system)

    hits.sort(key=lambda h: (h[0], h[1]), reverse=True)   # best match, then original-hardware priority
    return [_libretro_cover_obj(system, f, names[0]) for _s, _p, system, f in hits[:limit]]

# ---- picking the canonical IGDB game (avoid remakes / ROM hacks / wrong platform) ----
def _name_score(nq: str, nm: str) -> int:
    if not nm:
        return -999
    if nm == nq:
        return 100
    if nq.replace(" ", "") == nm.replace(" ", ""):   # spacing-only diff: "yugioh" == "yu gi oh"
        return 95
    tq, tn = set(nq.split()), set(nm.split())
    if tq and tq <= tn:
        return 60 - len(tn - tq) * 6        # candidate has all query words + a few extra
    if tn and tn <= tq:
        return 40 - len(tq - tn) * 6
    inter = tq & tn
    if len(inter) >= max(1, round(len(tq) * 0.6)):
        return 20 + len(inter) * 4 - len(tq - tn) * 4
    return -999

# IGDB category: 0 main_game, 8 remake, 9 remaster, 10 expanded, 11 port, 5 mod/ROM-hack.
_CAT_BONUS = {0: 400, 10: 90, 8: 80, 9: 80, 11: 60}

def _pick_igdb(games: List[Dict[str, Any]], q: str) -> Dict[str, Any]:
    nq = _norm_title(q)
    def rank(g):
        ns = _name_score(nq, _norm_title(g.get("name", "")))
        if ns <= -900:
            return (-1e9, 0)
        cat = g.get("category", 0)
        cb = -5000 if cat == 5 else _CAT_BONUS.get(cat, -200)
        frd = g.get("first_release_date") or 4102444800   # default: far future
        return (ns * 10 + cb, -frd)                        # name+category first, earliest release breaks ties
    return max(games, key=rank)

def _collect_platforms(games: List[Dict[str, Any]], chosen: Dict[str, Any]) -> List[str]:
    """Union platforms across every result that shares the chosen game's name (handles split entries)."""
    nchosen = _norm_title(chosen.get("name", ""))
    plats, seen = [], set()
    for g in games:
        if _norm_title(g.get("name", "")) == nchosen:
            for p in (g.get("platforms") or []):
                nm = p.get("name", "")
                if nm and nm not in seen:
                    seen.add(nm)
                    plats.append(nm)
    return plats

def _boxart_lookup(q: str, pc: bool = False) -> Dict[str, Any]:
    """pc=True: these are PC games — skip console box scans and serve the PC cover
       (Steam store capsule first, then IGDB cover, then SteamGridDB)."""
    q = (q or "").strip()
    matched, covers, alts = None, [], []

    # 1. IGDB identifies the game; for console games libretro adds the native-ratio box scan.
    igdb_games = _igdb_search(q)
    igdb_names: List[str] = []
    if igdb_games:
        g0 = _pick_igdb(igdb_games, q)
        matched = {"source": "igdb", "id": g0["id"], "name": g0.get("name", "")}
        igdb_names = [g0.get("name", "")] + [a.get("name", "") for a in (g0.get("alternative_names") or [])]
        if not pc:
            covers.extend(_libretro_covers(igdb_names, _collect_platforms(igdb_games, g0)))
        co = _igdb_cover_obj(g0)
        if co:
            covers.append(co)
        alts += [{"source": "igdb", "id": g["id"], "name": g.get("name", "")}
                 for g in igdb_games if g.get("id") != g0.get("id")][:5]

    # 1b. (console mode only) No scan yet — search libretro across all systems with the raw
    #     query too. Collapsed matching bridges "yugioh" -> "Yu-Gi-Oh!".
    if not pc and not any(c.get("source") == "libretro" for c in covers):
        libs = _libretro_covers(igdb_names + [q], [])
        if libs:
            covers[:0] = libs
            if matched is None:
                matched = {"source": "libretro", "id": 0, "name": _pretty_libretro_name(libs[0])}

    # 2. SteamGridDB — extra capsule covers to cycle through (and a match if IGDB missed).
    sgdb_games = _sgdb_search(q)
    if sgdb_games:
        if matched is None:
            matched = {"source": "steamgriddb", "id": sgdb_games[0]["id"], "name": sgdb_games[0]["name"]}
        covers += _sgdb_covers(sgdb_games[0]["id"])
        if not alts:
            alts = [{"source": "steamgriddb", "id": g["id"], "name": g["name"]} for g in sgdb_games[1:6]]

    # 3. Steam official store art. In PC mode it's the canonical cover -> put it first; otherwise
    #    only a last-resort fallback when nothing else matched.
    if pc or not covers:
        steam_hits = _steam_search(q, limit=1)
        if steam_hits:
            sc = _steam_cover(steam_hits[0]["appid"])
            if sc:
                sc["name"] = steam_hits[0]["name"]
                if pc:
                    covers.insert(0, sc)
                else:
                    covers.append(sc)
                if matched is None:
                    matched = {"source": "steam", "id": steam_hits[0]["appid"], "name": steam_hits[0]["name"]}

    return {"query": q, "matched": matched, "covers": covers, "alts": alts}

@app.get("/api/boxart")
def boxart_lookup(q: str, force: int = 0, platform: str = ""):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Missing game name")
    pc = platform.lower() == "pc"
    key = f"boxart:v6:{'pc:' if pc else ''}{q.strip().lower()}"
    if not force:
        cached = _cache_read(key)
        if cached:
            return cached
    with _lock_for(key):
        if not force:
            cached = _cache_read(key)
            if cached:
                return cached
        result = _boxart_lookup(q, pc=pc)
        if not (IGDB_CLIENT_ID and IGDB_CLIENT_SECRET):
            result["warning"] = ("IGDB credentials not set — add IGDB_CLIENT_ID / "
                                 "IGDB_CLIENT_SECRET for native-ratio box art.")
        if result.get("covers"):
            _cache_write(key, result)
        return result

@app.get("/api/boxart/covers")
def boxart_covers(source: str, id: int, platform: str = ""):
    """Covers for a specific game the user picked as an override."""
    pc = platform.lower() == "pc"
    if source == "igdb":
        games = _igdb_game(id)
        out: List[Dict[str, Any]] = []
        if games:
            g = games[0]
            names = [g.get("name", "")] + [a.get("name", "") for a in (g.get("alternative_names") or [])]
            plats = [p.get("name", "") for p in (g.get("platforms") or [])]
            if not pc:
                out.extend(_libretro_covers(names, plats))
            co = _igdb_cover_obj(g)
            if co:
                out.append(co)
            if pc:
                sh = _steam_search(g.get("name", ""), limit=1)
                if sh:
                    sc = _steam_cover(sh[0]["appid"])
                    if sc:
                        sc["name"] = sh[0]["name"]
                        out.insert(0, sc)
        return {"covers": out}
    if source == "steamgriddb":
        return {"covers": _sgdb_covers(id)}
    if source == "steam":
        sc = _steam_cover(id)
        return {"covers": [sc] if sc else []}
    raise HTTPException(status_code=400, detail="Unknown source")

@app.get("/api/boxart/proxy")
def boxart_proxy(url: str):
    """Stream an allowed image through our origin so the page can ZIP it (CORS-safe)."""
    from urllib.parse import urlparse
    host = (urlparse(url).hostname or "").lower()
    if not any(host == d or host.endswith("." + d) for d in BOXART_ALLOWED_HOSTS):
        raise HTTPException(status_code=400, detail="Host not allowed")
    try:
        with httpx.Client(timeout=20.0, headers=UA, follow_redirects=True) as c:
            r = c.get(url)
            r.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Fetch failed: {e}")
    ctype = r.headers.get("content-type", "image/jpeg")
    return Response(content=r.content, media_type=ctype,
                    headers={"Cache-Control": "public, max-age=86400"})

@app.get("/boxart", response_class=HTMLResponse)
async def boxart_page():
    return FileResponse(os.path.join(STATIC_DIR, "boxart.html"))

@app.get("/animate", response_class=HTMLResponse)
async def animate_page():
    return FileResponse(os.path.join(STATIC_DIR, "animate.html"))

# --------------------------------------------------------------------------------------
# Pretty URL Routing (Dynamic Handheld/Accessory IDs)
# --------------------------------------------------------------------------------------
# This set matches the IDs in your shop.html script
KNOWN_IDS = {
    "anbernic_rg477v", "trimui_brick_hammer", "trimui_brick", "rp6",
    "steam_deck", "rp5", "rp_mini2", "rp_classic", "rp_flip2", "rpg2",
    "miyoo_mini_v4", "miyoo_mini_plus", "odin3", "odin2_portal", "ayn_thor",
    "anker_power", "ugreen_power", "ugreen_dock", "rp_ds_addon", 
    "gamesir_g8p", "sd_card"
}

# 4. The "Catch-All" Route (MUST BE LAST)
@app.get("/{product_id}", response_class=HTMLResponse)
async def serve_pretty_product_url(product_id: str):
    # Check if the URL matches a known product ID
    if product_id in KNOWN_IDS:
        # Return the shop page; the JS there can handle highlighting if needed, 
        # or it just serves as a valid entry point.
        return FileResponse(os.path.join(STATIC_DIR, "shop.html"))
    
    # If not found, raise 404
    raise HTTPException(status_code=404, detail="Page not found")

@app.get("/test-store", response_class=HTMLResponse)
async def test_store():
    return FileResponse(os.path.join(STATIC_DIR, "shop.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
