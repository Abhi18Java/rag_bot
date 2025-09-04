from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import data_ingestion
import generation
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    result = data_ingestion.ingest_pdf(file_path)
    return result

@app.post("/query")
async def query(query: str):
    result = generation.generate_response(query)
    return result