import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
DATA_DIR = "data"

class GameEntry(BaseModel):
    name: str
    performance: str
    driver: str
    emulator: str
    update_version: str
    notes: str
    device: str
    # 'system' will be handled by the backend, not submitted from the form
    
@app.get("/api/compatibility/{system_name}")
def get_compatibility_list(system_name: str):
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Compatibility list for {system_name} not found.")
    
    df = pd.read_csv(file_path)
    
    # NEW LINE: This is the magic fix. It replaces all empty/NaN values with an empty string.
    df = df.fillna('')
    
    return df.to_dict(orient="records")

@app.post("/api/compatibility/{system_name}")
async def add_game_entry(system_name: str, entry: GameEntry):
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Compatibility list for {system_name} not found.")
    
    df = pd.read_csv(file_path)
    today_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    new_row = {
        'name': entry.name,
        'performance': entry.performance,
        'driver': entry.driver,
        'emulator': entry.emulator,
        'update': entry.update_version,
        'notes': entry.notes,
        'date': today_date,
        'device': entry.device,
        'system': system_name
    }
    
    new_df = pd.DataFrame([new_row])
    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_csv(file_path, index=False)
    
    return {"status": "success", "data": entry.dict()}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    # Assumes your main page is index.html
    # If you renamed it, change "index.html" here.
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)