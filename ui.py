#D:\AI_Project\rag_app\ui.py
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"  # Uvicorn default port

st.title("RAG Application")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    with st.spinner("Ingesting PDF..."):
        response = requests.post(f"{BACKEND_URL}/ingest", files={"file": uploaded_file})
        if response.status_code == 200:
            st.success(f"PDF ingested: {response.json()['chunks_ingested']} chunks")
            

query = st.text_input("Ask a question about the PDFs")
if query:
    with st.spinner("Generating response..."):
        response = requests.post(f"{BACKEND_URL}/query", json={"query": query})
        if response.status_code == 200:
            result = response.json()
            st.write("Response:", result["response"])
            st.write("Sources:", result["sources"])