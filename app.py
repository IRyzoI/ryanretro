import csv
import os
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
DATA_DIR = "data"

class GameEntry(BaseModel):
    game: str
    performance: str
    driver: str
    emulator: str
    update_version: str
    notes: str
    device: str
    
@app.get("/api/compatibility/{system_name}")
def get_compatibility_list(system_name: str):
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"List not found.")
    
    games_list = []
    try:
        with open(file_path, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            print(f"Headers found in {system_name}.csv: {reader.fieldnames}")
            for row in reader:
                games_list.append(row)
        return games_list
    except Exception as e:
        print(f"CRITICAL ERROR reading CSV file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error processing CSV file.")

@app.post("/api/compatibility/{system_name}")
async def add_game_entry(system_name: str, entry: GameEntry):
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    headers = ['game', 'performance', 'driver', 'emulator', 'update', 'notes', 'date', 'device', 'system']
    
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        today_date = datetime.now().strftime('%Y-%m-%d')
        new_row = {
            'game': entry.game,
            'performance': entry.performance,
            'driver': entry.driver,
            'emulator': entry.emulator,
            'update': entry.update_version,
            'notes': entry.notes,
            'date': today_date,
            'device': entry.device,
            'system': system_name
        }
        writer.writerow(new_row)
    return {"status": "success", "data": entry.dict()}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)