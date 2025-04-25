# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from automation import run_automation

class ECRequest(BaseModel):
    search_type: str
    zone: str
    district: str
    ec_start_date: str
    ec_end_date: str
    sro: str
    village: str
    survey_number: str
    subdivision_number: str

app = FastAPI()

@app.post("/generate-ec")
def generate_ec(data: ECRequest):
    download_success = run_automation(data)
    return {"pdf_downloaded": download_success}

