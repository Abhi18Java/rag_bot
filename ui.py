import streamlit as st
import requests
import logging

BACKEND_URL = "http://localhost:8000"  # Uvicorn default port
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

st.title("RAG Application")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    logging.info(f"Uploading file: {uploaded_file.name}")
    with st.spinner("Ingesting PDF..."):
        try:
            response = requests.post(f"{BACKEND_URL}/ingest", files={"file": uploaded_file})
            result = response.json()
            logging.info(f"Ingestion response: {result}")
            if result.get("status") == "success":
                st.success(f"PDF ingested: {result['chunks_ingested']} chunks")
            else:
                st.error(f"Ingestion failed: {result.get('message', 'Unknown error')}")
                logging.error(f"Ingestion failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error during ingestion: {e}")
            logging.error(f"Exception during ingestion: {e}")

query = st.text_input("Ask a question about the PDFs")
if query:
    logging.info(f"User query: {query}")
    with st.spinner("Generating response..."):
        try:
            response = requests.post(f"{BACKEND_URL}/query", json={"query": query})
            if response.status_code == 200:
                result = response.json()
                logging.info(f"Query response: {result}")
                st.write("Response:", result["response"])
                st.write("Sources:", result["sources"])
            else:
                st.error("Failed to get response from backend.")
                logging.error(f"Backend query failed: {response.text}")
        except Exception as e:
            st.error(f"Error during query: {e}")
            logging.error(f"Exception during query: {e}")