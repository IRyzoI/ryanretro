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
        with open(file_path, mode='r', encoding='utf-8-sig') as csvfile: # Use utf-8-sig to handle BOM
            reader = csv.DictReader(csvfile)
            for row in reader:
                games_list.append(row)
        return games_list
    except Exception as e:
        print(f"CRITICAL ERROR reading CSV file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error processing CSV file.")

@app.post("/api/compatibility/{system_name}")
async def add_game_entry(system_name: str, entry: GameEntry):
    # This print statement is for debugging. It will show up in your logs.
    print(f"Received new entry for {system_name}: {entry.dict()}")
    
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    headers = ['game', 'performance', 'driver', 'emulator', 'update', 'notes', 'date added', 'device', 'system']
    
    # Create file with headers if it doesn't exist
    file_exists = os.path.exists(file_path)
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            if not file_exists:
                writer.writeheader() # Add headers only if the file is new

            today_date = datetime.now().strftime('%Y-%m-%d')
            new_row = {
                'game': entry.game,
                'performance': entry.performance,
                'driver': entry.driver,
                'emulator': entry.emulator,
                'update': entry.update_version,
                'notes': entry.notes,
                'date added': today_date,
                'device': entry.device,
                'system': system_name
            }
            writer.writerow(new_row)
            
        return {"status": "success", "data": entry.dict()}
    except Exception as e:
        print(f"CRITICAL ERROR writing to CSV file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error saving entry.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)