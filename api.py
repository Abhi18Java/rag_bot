import logging
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import data_ingestion
import generation
import os
from model import QueryRequest, QueryResponse

app = FastAPI()

@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    logging.info(f"Received file: {file.filename}")
    os.makedirs("temp", exist_ok=True)
    file_path = f"temp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    result = data_ingestion.ingest_pdf(file_path)
    logging.info(f"Ingestion result: {result}")
    return result

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    logging.info(f"Received query: {request.query}")
    result = generation.generate_response(request.query)
    logging.info(f"Query result: {result}")
    return QueryResponse(**result)