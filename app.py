import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

# --- Configuration ---
DATA_FILE = "compatibility_list.csv"

# --- FastAPI App Initialization ---
app = FastAPI()

# Mount the 'static' directory to serve index.html, css, js files
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Data Model for New Entries ---
class GameEntry(BaseModel):
    name: str
    performance: str
    driver: str
    emulator: str
    update_version: str
    notes: str

# --- Backend Functions ---
def load_data():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=['name', 'performance', 'driver', 'emulator', 'update', 'notes', 'date']).to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)

def save_entry(entry: GameEntry):
    df = load_data()
    today_date = pd.Timestamp.now().strftime('%Y/%m/%d')
    new_row = {
        'name': entry.name,
        'performance': entry.performance,
        'driver': entry.driver,
        'emulator': entry.emulator,
        'update': entry.update_version,
        'notes': entry.notes,
        'date': today_date
    }
    new_df = pd.DataFrame([new_row])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(DATA_FILE, index=False)

# --- API Endpoints ---
@app.get("/api/games")
def get_games():
    df = load_data()
    return df.to_dict(orient="records")

@app.post("/api/games")
async def add_game(entry: GameEntry):
    save_entry(entry)
    return {"status": "success", "data": entry.dict()}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # This serves your main HTML file
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)