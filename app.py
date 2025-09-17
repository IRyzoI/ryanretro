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
    
@app.get("/api/compatibility/{system_name}")
def get_compatibility_list(system_name: str):
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Compatibility list for {system_name} not found.")
    
    # Read the CSV, but this time we will FORCE the column names.
    df = pd.read_csv(file_path)

    # --- THIS IS THE NEW, BULLETPROOF CODE ---
    # Define the exact headers our JavaScript expects.
    expected_headers = ['game', 'performance', 'driver', 'emulator', 'update', 'notes', 'date', 'device', 'system']
    
    # Check if the number of columns matches to avoid errors.
    if len(df.columns) == len(expected_headers):
        df.columns = expected_headers
    else:
        # If the columns don't match, we know the CSV is malformed.
        print(f"ERROR: Column count mismatch in {file_path}. Expected {len(expected_headers)}, but found {len(df.columns)}.")
        # Return an empty list to prevent a crash.
        return []
    # --- END OF NEW CODE ---

    df = df.fillna('')
    return df.to_dict(orient="records")

@app.post("/api/compatibility/{system_name}")
async def add_game_entry(system_name: str, entry: GameEntry):
    # ... (The rest of your app.py code remains the same) ...
    file_path = os.path.join(DATA_DIR, f"{system_name}.csv")
    if not os.path.exists(file_path):
        # Create the file with headers if it doesn't exist
        pd.DataFrame(columns=['game', 'performance', 'driver', 'emulator', 'update', 'notes', 'date', 'device', 'system']).to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    today_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    
    new_row = {
        'game': entry.name,
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
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)