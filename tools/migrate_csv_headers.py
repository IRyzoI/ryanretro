# tools/migrate_csv_headers.py
import os, csv

DATA_DIR = "data"
META_LEFT  = ["device","chipset","system"]
META_RIGHT = ["date added"]

def normalize(s): return (s or "").strip()

def desired_header(existing):
    ex = [normalize(h) for h in existing]
    # Keep your current display columns in the middle, but ensure meta columns exist.
    middle = [h for h in ex if h not in set(META_LEFT + META_RIGHT)]
    return META_LEFT + middle + META_RIGHT

for name in os.listdir(DATA_DIR):
    if not name.endswith(".csv"): continue
    path = os.path.join(DATA_DIR, name)

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        hdr = next(r, [])
        rows = list(r)

    if not hdr:
        print(f"[skip empty] {name}")
        continue

    new_hdr = desired_header(hdr)
    if hdr == new_hdr:
        print(f"[ok] {name} already normalized")
        continue

    # Map old header -> dict, then write new header with defaults
    old_hdr = [normalize(h) for h in hdr]
    dict_rows = []
    for row in rows:
        d = {old_hdr[i]: row[i] if i < len(row) else "" for i in range(len(old_hdr))}
        # ensure meta keys exist
        for k in META_LEFT + META_RIGHT:
            d.setdefault(k, "")
        dict_rows.append(d)

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=new_hdr)
        w.writeheader()
        for d in dict_rows:
            w.writerow({k: d.get(k, "") for k in new_hdr})

    print(f"[migrated] {name} -> {new_hdr}")
