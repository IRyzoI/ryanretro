import os
import csv
import re
import time
import json
import hashlib
import threading
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

import httpx
import markdown
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field  # consolidated import
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# --------------------------------------------------------------------------------------
# FastAPI app + CORS
# --------------------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
YT_API_KEY = os.getenv("YT_API_KEY")  # set this in your environment
# Default channel for homepage latest videos (set YT_CHANNEL_ID in .env, or hardcode here)
DEFAULT_CHANNEL_ID = os.getenv("YT_CHANNEL_ID") or "UCh9GxjM-FNuSWv7xqn3UKVw"

UA = {"User-Agent": "RyanRetro/1.0"}

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 24 * 3600
CACHE_VERSION = "v5"  # bump to invalidate old caches when logic changes

# Piped mirrors (fast fallback)
PIPED_BASES = [
    "https://pipedapi.kavin.rocks/api/v1",
    "https://piped.video/api/v1",
    "https://piped.projectsegfau.lt/api/v1",
]

_locks: Dict[str, threading.Lock] = {}

def _cache_file(key: str) -> str:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.json")

def _cache_read(key: str, ttl: int = CACHE_TTL_SECONDS):
    path = _cache_file(key)
    try:
        st = os.stat(path)
        if time.time() - st.st_mtime <= ttl:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
    except FileNotFoundError:
        pass
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

def _lock_for(key: str) -> threading.Lock:
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]

# --------------------------------------------------------------------------------------
# Matcher (prevents RP5/Flip 2 collisions via disambiguators)
# --------------------------------------------------------------------------------------
def _make_matcher(q: str, aliases: str):
    """
    Match if title contains any full phrase (q or alias) OR
    (≥2 generic tokens) AND (at least 1 disambiguator token if any exist).

    Disambiguator tokens are those that contain digits (e.g., '5', 'rp5', 'flip2', '8g2').
    NOTE: pure-digit tokens like "2" or "5" are kept even if length == 1.
    """
    import re as _re

    phrases: List[str] = []
    if q:
        phrases.append(q.strip().lower())
    for ph in (aliases or "").split(","):
        ph = ph.strip().lower()
        if ph:
            phrases.append(ph)

    generic_tokens: set[str] = set()
    disamb_tokens: set[str] = set()

    def _tokenize(s: str):
        # Keep numeric tokens even if a single character (e.g., "2", "5")
        for t in _re.findall(r"[a-z0-9]+", s.lower()):
            if t.isdigit():          # "2", "5", "8"
                yield t
            elif len(t) >= 2:        # drop single-letter alphabetic noise
                yield t

    for ph in phrases:
        for t in _tokenize(ph):
            if t.isdigit() or any(ch.isdigit() for ch in t):
                disamb_tokens.add(t)
            else:
                generic_tokens.add(t)

    def match(title: str) -> bool:
        tl = (title or "").lower()
        # 1) full phrase hit (most permissive)
        if any(ph and ph in tl for ph in phrases):
            return True
        # 2) token fallback: require ≥2 generic hits AND at least one disambiguator (if any exist)
        gen_hits = sum(1 for t in generic_tokens if t in tl)
        if gen_hits >= 2:
            return (not disamb_tokens) or any(t in tl for t in disamb_tokens)
        return False

    return match


# --------------------------------------------------------------------------------------
# YouTube Data API (primary)
# --------------------------------------------------------------------------------------
def _yt_api_list_uploads(channel_id: str, pages: int = 12) -> List[Dict[str, str]]:
    """
    Fetch recent uploads via the channel's 'uploads' playlist.
    pages=12 -> up to ~600 videos. Cheap and reliable.
    """
    if not YT_API_KEY:
        return []

    with httpx.Client(timeout=10.0, headers=UA) as client:
        # Get uploads playlist id
        r = client.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"part": "contentDetails", "id": channel_id, "key": YT_API_KEY},
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            print("[yt] channels.list returned no items.")
            return []
        uploads_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]

        out: List[Dict[str, str]] = []
        token = None
        page_count = 0
        while page_count < max(1, pages):
            params = {
                "part": "snippet",
                "playlistId": uploads_id,
                "maxResults": 50,
                "key": YT_API_KEY,
            }
            if token:
                params["pageToken"] = token
            rr = client.get("https://www.googleapis.com/youtube/v3/playlistItems", params=params)
            rr.raise_for_status()
            data = rr.json()
            for it in data.get("items", []):
                sn = it.get("snippet") or {}
                res_id = (sn.get("resourceId") or {})
                vid = res_id.get("videoId")
                title = sn.get("title") or ""
                if vid and title:
                    out.append({
                        "title": title,
                        "videoId": vid,
                        "publishedAt": sn.get("publishedAt") or ""
                    })
            token = data.get("nextPageToken")
            page_count += 1
            if not token:
                break
        print(f"[yt] uploads fetched: {len(out)}")
        return out

def _yt_api_search(channel_id: str, query: str, limit: int) -> List[Dict[str, str]]:
    """Search a channel using YouTube Data API v3 (requires YT_API_KEY)."""
    if not YT_API_KEY:
        return []
    url = "https://www.googleapis.com/youtube/v3/search"
    out: List[Dict[str, str]] = []
    page_token: Optional[str] = None
    pages = 0

    with httpx.Client(timeout=10.0, headers=UA) as client:
        while len(out) < max(limit, 50) and pages < 5:
            params = {
                "part": "snippet",
                "channelId": channel_id,
                "q": query or "",
                "order": "date",
                "type": "video",
                "maxResults": 50,
                "key": YT_API_KEY,
            }
            if page_token:
                params["pageToken"] = page_token
            r = client.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            items = data.get("items", [])
            for it in items:
                id_obj = it.get("id") or {}
                vid = id_obj.get("videoId")
                title = ((it.get("snippet") or {}).get("title")) or ""
                if vid and title:
                    out.append({"title": title, "videoId": vid})
            page_token = data.get("nextPageToken")
            if not page_token:
                break
            pages += 1
    print(f"[yt] search('{query}') hits: {len(out)}")
    return out

# --------------------------------------------------------------------------------------
# Piped fallback
# --------------------------------------------------------------------------------------
def _piped_get(path: str, params: Dict[str, Any] | None = None):
    import random as _random
    _random.shuffle(PIPED_BASES)
    for base in PIPED_BASES:
        try:
            with httpx.Client(timeout=6.0, headers=UA) as client:
                r = client.get(base + path, params=params)
            r.raise_for_status()
            return r.json()
        except Exception:
            continue
    return None

def _piped_channel_videos(channel_id: str, pages: int = 6) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    nextpage = None
    for _ in range(max(1, pages)):
        params = {"nextpage": nextpage} if nextpage else None
        data = _piped_get(f"/channel/{channel_id}/videos", params=params) or _piped_get(
            f"/channel/{channel_id}", params=params
        )
        if not data:
            break
        streams = data.get("videos") or data.get("relatedStreams") or []
        for s in streams:
            vid = s.get("videoId") or s.get("id")
            title = s.get("title") or ""
            if vid and title:
                out.append({"title": title, "videoId": vid})
        nextpage = data.get("nextpage")
        if not nextpage:
            break
    print(f"[piped] uploads fetched: {len(out)}")
    return out

def _piped_search_channel(channel_id: str, query: str, pages: int = 2) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen: set[str] = set()
    nextpage = None
    for _ in range(max(1, pages)):
        params: Dict[str, Any] = {"q": query, "channelId": channel_id, "region": "US"}
        if nextpage:
            params["nextpage"] = nextpage
        data = _piped_get("/search", params=params)
        if not data:
            break
        items = data.get("items") if isinstance(data, dict) else data
        if not items:
            break
        for s in items:
            vid = s.get("videoId") or s.get("id")
            title = s.get("title") or s.get("name") or ""
            if vid and title and vid not in seen:
                out.append({"title": title, "videoId": vid})
                seen.add(vid)
        if isinstance(data, dict):
            nextpage = data.get("nextpage")
            if not nextpage:
                break
        else:
            break
    print(f"[piped] search('{query}') hits: {len(out)}")
    return out

# --------------------------------------------------------------------------------------
# RSS fallback
# --------------------------------------------------------------------------------------
def _rss_latest(channel_id: str) -> List[Dict[str, str]]:
    rss = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    with httpx.Client(timeout=6.0, headers=UA) as client:
        r = client.get(rss)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
    out: List[Dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        vid = entry.findtext("yt:videoId", default="", namespaces=ns)
        published = (entry.findtext("atom:published", default="", namespaces=ns)
                     or entry.findtext("atom:updated", default="", namespaces=ns)
                     or "")
        if vid and title:
            out.append({"title": title, "videoId": vid, "publishedAt": published})
    print(f"[rss] latest fetched: {len(out)}")
    return out

# --------------------------------------------------------------------------------------
# Core fetch combining API + fallbacks (for device pages)
# --------------------------------------------------------------------------------------
def _fetch_videos(channel_id: str, q: str, aliases: str, limit: int) -> List[Dict[str, str]]:
    matcher = _make_matcher(q, aliases)
    results: List[Dict[str, str]] = []

    # Build phrases list (main q + distinct aliases)
    phrases: List[str] = []
    if q and q.strip():
        phrases.append(q.strip())
    seen_lower = set(p.lower() for p in phrases)
    for ph in (aliases or "").split(","):
        ph = ph.strip()
        if ph and ph.lower() not in seen_lower:
            phrases.append(ph)
            seen_lower.add(ph.lower())

    # 1) Primary: list uploads and filter locally (deep & cheap)
    try:
        uploads = _yt_api_list_uploads(channel_id, pages=12) if YT_API_KEY else []
    except Exception as e:
        print(f"[yt] uploads error: {e}")
        uploads = []
    if uploads:
        filtered = [it for it in uploads if matcher(it["title"])]
        print(f"[yt] uploads matched: {len(filtered)}")
        results.extend(filtered)

    # 2) If not enough, API search per phrase (helps find older/oddly-titled matches)
    if len(results) < limit and YT_API_KEY:
        try:
            for ph in phrases:
                hits = _yt_api_search(channel_id, ph, limit=100)
                filtered = [it for it in hits if matcher(it["title"])]
                print(f"[yt] search('{ph}') matched: {len(filtered)}")
                results.extend(filtered)
                if len(results) >= limit * 2:
                    break
        except Exception as e:
            print(f"[yt] search error: {e}")

    # 3) Piped fallback
    if len(results) < limit:
        try:
            items = _piped_channel_videos(channel_id, pages=6)
            filtered = [it for it in items if matcher(it["title"])]
            print(f"[piped] uploads matched: {len(filtered)}")
            results.extend(filtered)
            if len(results) < limit:
                for ph in phrases:
                    hits = _piped_search_channel(channel_id, ph, pages=2)
                    filtered = [it for it in hits if matcher(it["title"])]
                    print(f"[piped] search('{ph}') matched: {len(filtered)}")
                    results.extend(filtered)
                    if len(results) >= limit * 2:
                        break
        except Exception as e:
            print(f"[piped] error: {e}")

    # 4) RSS last resort
    if len(results) == 0:
        try:
            items = _rss_latest(channel_id)
            filtered = [it for it in items if matcher(it["title"])]
            print(f"[rss] matched: {len(filtered)}")
            results.extend(filtered)
        except Exception as e:
            print(f"[rss] error: {e}")

    # De-dupe & trim
    out, seen = [], set()
    for it in results:
        vid = it.get("videoId")
        if vid and vid not in seen:
            out.append({"title": it.get("title", ""), "videoId": vid})
            seen.add(vid)

    print(f"[final] unique matched: {len(out)} (returning {min(len(out), limit)})")
    return out[:limit]

# ==============================================================================
# APIs
# ==============================================================================

# Device pages
@app.get("/api/youtube/latest")
def youtube_latest(channel_id: str, q: str = "", aliases: str = "", limit: int = 18, force: int = 0):
    # Exact phrases we’ll use to filter titles (local, strict)
    search_phrases = [p.strip().lower() for p in ([q] + aliases.split(',')) if p.strip()]

    # Versioned cache key (you can also include aliases if you want per-alias caches)
    key = f"{CACHE_VERSION}:yt-api:{channel_id}|q={q.strip().lower()}"

    def _filter(items: List[Dict[str, str]] | None) -> List[Dict[str, str]]:
        items = items or []
        return [v for v in items if any(p in (v.get("title", "").lower()) for p in search_phrases)]

    if not force:
        if cached := _cache_read(key):
            return _filter(cached)[:limit]

    with _lock_for(key):
        if not force:
            if cached := _cache_read(key):
                return _filter(cached)[:limit]
        try:
            fresh_videos = _fetch_videos(channel_id, q, aliases, limit)
            _cache_write(key, fresh_videos)
            return _filter(fresh_videos)[:limit]
        except Exception as e:
            print(f"[yt] latest error, serving stale if present: {e}")
            if stale := _cache_read_any_age(key):
                return _filter(stale)[:limit]
            raise

# Homepage “latest videos”
@app.get("/api/latest-videos")
def latest_videos(
    response: Response,
    channel_id: str = DEFAULT_CHANNEL_ID,
    limit: int = 3,   # show only 3 on homepage
    force: int = 0
):
    """
    Simple latest uploads for homepage.
    - Cached for 24h via file cache (+ Cache-Control header)
    - Force refresh with ?force=1
    - Output shape matches index.html expectations: id, title, thumb, publishedAt
    """
    if not channel_id:
        raise HTTPException(status_code=400, detail="No channel_id. Set YT_CHANNEL_ID in .env or pass ?channel_id=")

    key = f"{CACHE_VERSION}:home-latest:{channel_id}"

    def _shape(items: List[Dict[str, str]] | None):
        items = items or []
        data = [{
            "id": it["videoId"],
            "title": it.get("title", ""),
            "thumb": f"https://i.ytimg.com/vi/{it['videoId']}/hqdefault.jpg",
            "publishedAt": it.get("publishedAt", "")
        } for it in items][:limit]
        # cache headers
        if force:
            response.headers["Cache-Control"] = "no-store"
        else:
            response.headers["Cache-Control"] = f"public, max-age={CACHE_TTL_SECONDS}"
        return data

    if not force:
        if cached := _cache_read(key):
            return _shape(cached)

    with _lock_for(key):
        if not force:
            if cached := _cache_read(key):
                return _shape(cached)
        try:
            # Cheap, accurate: official uploads list
            items = _yt_api_list_uploads(channel_id, pages=3)
            _cache_write(key, items)
            return _shape(items)
        except Exception as e:
            print(f"[home latest] yt error: {e}")
            try:
                # Fallback: RSS feed (no API quota)
                items = _rss_latest(channel_id)
                _cache_write(key, items)
                return _shape(items)
            except Exception as e2:
                print(f"[home latest] rss error: {e2}")
                if stale := _cache_read_any_age(key):
                    return _shape(stale)
                raise

# --------------------------------------------------------------------------------------
# Static mounts
# --------------------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/data", StaticFiles(directory="data"), name="data")

DATA_DIR = "data"

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# --------------------------------------------------------------------------------------
# Compatibility CSV -> JSON API (read)
# --------------------------------------------------------------------------------------
class GameEntry(BaseModel):
    game: str
    performance: str | None = None
    driver: str | None = None
    emulator: str | None = None
    update_version: str | None = None
    notes: str | None = None
    device: str | None = None
    date_added: str | None = None

@app.get("/api/compatibility/{system_name}")
def get_compatibility_list(system_name: str):
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="List not found.")
    rows: List[Dict[str, Any]] = []
    with open(file_path, mode="r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return {"system": system_name, "count": len(rows), "rows": rows}

# --------------------------------------------------------------------------------------
# Submit compatibility entries (append to CSV)
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
    # common required
    device: str = Field(..., min_length=1)
    chipset: str = Field(..., min_length=1)
    system: str  = Field(..., min_length=1)

    # everything else is optional and varies by system
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

    class Config:
        allow_population_by_field_name = True  # accept alias or pythonic names

@app.post("/api/compat/submit")
def post_compat(sub: CompatSubmission):
    """
    Append a row to the appropriate system CSV.
    Required in payload: device, chipset, system
    Other fields (like game/performance/driver/emulator...) are optional per-system,
    and will be added if present.
    """
    system = (sub.system or "").strip().lower()
    if system not in SYSTEM_CSVS:
        raise HTTPException(status_code=400, detail=f"Unknown system '{sub.system}'")

    # Path to the CSV
    file_path = os.path.join(DATA_DIR, SYSTEM_CSVS[system])
    os.makedirs(DATA_DIR, exist_ok=True)

    # Flatten with alias names and strip Nones
    row = json.loads(sub.json(by_alias=True, exclude_none=True))

    # Normalize any pythonic keys to CSV header names (aliases already do most of this)
    if "rom_region" in row and "rom region" not in row:
        row["rom region"] = row.pop("rom_region")
    if "winlator_version" in row and "winlator version" not in row:
        row["winlator version"] = row.pop("winlator_version")
    if "dx_wrapper" in row and "dx wrapper" not in row:
        row["dx wrapper"] = row.pop("dx_wrapper")
    if "game_resolution" in row and "game resolution" not in row:
        row["game resolution"] = row.pop("game_resolution")
    if "dxvk_version" in row and "dxvk version" not in row:
        row["dxvk version"] = row.pop("dxvk_version")
    if "vkd3d_version" in row and "vkd3d version" not in row:
        row["vkd3d version"] = row.pop("vkd3d_version")
    if "precompiled_shaders" in row and "pre-compiled shaders" not in row:
        row["pre-compiled shaders"] = row.pop("precompiled_shaders")
    if "game_title_id" in row and "game title id" not in row:
        row["game title id"] = row.pop("game_title_id")

    # Always set/override these
    row["system"] = system
    row["date added"] = datetime.utcnow().strftime("%Y/%m/%d")

    # Read existing header order (if any)
    fieldnames: List[str] = []
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])

    # Ensure required columns exist (and include any new keys we’re adding)
    present = set(fieldnames)
    to_add = list((REQUIRED_COLS | set(row.keys())) - present)
    if not fieldnames:
        # New file: start with a sensible order
        fieldnames = ["device", "chipset", "system", "game", "performance", "driver",
                      "emulator", "resolution", "rom region", "winlator version",
                      "dx wrapper", "game resolution", "dxvk version", "vkd3d version",
                      "pre-compiled shaders", "game title id", "notes", "date added"]
    fieldnames += [c for c in to_add if c not in fieldnames]

    # Lock + write
    lock = _lock_for(file_path)
    with lock:
        write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0
        with open(file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            # Fill missing cells with ''
            full_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(full_row)

    return {"ok": True, "row": full_row}

# --------------------------------------------------------------------------------------
# Guides (Markdown or static HTML)
# --------------------------------------------------------------------------------------
@app.get("/guides/{slug}", response_class=HTMLResponse)
def serve_guide(slug: str):
    md_path = os.path.join("guides", f"{slug}.md")
    if os.path.exists(md_path):
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Drop a leading H1 (we add our own)
        lines = md_content.splitlines()
        if lines and lines[0].lstrip().startswith("#"):
            lines = lines[1:]
        md_content = "\n".join(lines)

        m = re.search(r"^\s*#[ \t]+(.+?)\s*$", md_content, flags=re.M)
        page_title = (m.group(1).strip() if m else slug.replace("-", " ").title())

        html = markdown.markdown(md_content, extensions=["fenced_code", "tables"])

        def wrap_iframe(match):
            iframe = match.group(0)
            iframe = re.sub(r'\s(width|height)="[^"]*"', "", iframe)
            return (
                '<div class="mx-auto max-w-3xl">'
                '  <div class="relative w-full aspect-video rounded-lg overflow-hidden pixel-border">'
                f'    {iframe}'
                "  </div>"
                "</div>"
            )

        html = re.sub(
            r'<iframe[^>]*src="https://www\.youtube\.com/embed/[^"]+"[^>]*></iframe>',
            wrap_iframe,
            html,
            flags=re.I,
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>{page_title} | Ryan Retro</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
  <script src="https://unpkg.com/aos@2.3.1/dist/aos.js" defer></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Press Start 2P&display=swap');
    .retro-font{{ font-family:'Press Start 2P', cursive; }}
    .pixel-border{{ border: 4px solid #000; box-shadow: 8px 8px 0 rgba(0,0,0,.2); }}
    .hero-header{{ position:relative; background:#111827; }}
    .hero-header::before{{ content:""; position:absolute; inset:0;
      background-image:linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
                       linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
      background-size:24px 24px; opacity:.35; pointer-events:none; }}
    .hero-header::after{{ content:""; position:absolute; inset:0;
      background: radial-gradient(80% 60% at 50% 30%, rgba(0,0,0,0) 0%, rgba(0,0,0,.35) 55%, rgba(0,0,0,.55) 100%); }}
    .aspect-video iframe{{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
    .prose :where(h1,h2,h3,h4){{ color:#e5e7eb; }}
    .prose a{{ color:#facc15; }}
  </style>
</head>
<body class="bg-gray-900 text-white">
  <header class="hero-header py-10 md:py-14">
    <div class="container mx-auto px-4 relative z-10 text-center">
      <a href="/static/index.html" class="inline-flex items-center justify-center mb-4">
        <img src="https://huggingface.co/spaces/IRyzoI/RyanRetro/resolve/main/images/ryan%20retro%20logo.png"
             alt="Ryan Retro" class="w-20 h-20 rounded-full pixel-border bg-gray-900">
      </a>
      <h1 class="retro-font text-2xl md:text-4xl text-yellow-400">{page_title}</h1>
    </div>
  </header>
  <main class="container mx-auto px-4 py-10">
    <article class="prose prose-invert max-w-3xl mx-auto">
      {html}
    </article>
    <div class="max-w-3xl mx-auto mt-10">
      <a href="/static/index.html#guides" class="inline-flex items-center text-yellow-400 hover:text-yellow-300 font-bold">← Back to Retro Guides</a>
    </div>
  </main>
  <footer class="bg-black py-8 mt-16">
    <div class="container mx-auto px-4 text-center text-gray-500 text-sm">
      © 2025 Ryan Retro. All rights reserved.
    </div>
  </footer>
</body>
</html>"""

    html_path = os.path.join("static", "guides", f"{slug}.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")

    idx_path = os.path.join("static", "guides", slug, "index.html")
    if os.path.exists(idx_path):
        return FileResponse(idx_path, media_type="text/html")

    return HTMLResponse("<h1>Guide not found</h1>", status_code=404)
