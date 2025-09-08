import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import config


def ingest_pdf(file_path: str):
    logging.info(f"Starting ingestion for PDF: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from PDF.")
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(chunks)} chunks.")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",   # Gemini embedding model
            google_api_key=config.GEMINI_API_KEY
        )
        logging.info("Initialized Gemini embeddings.")

        vectorstore = FAISS.from_documents(chunks, embeddings)
        logging.info("Created FAISS vectorstore.")

        vectorstore.save_local("faiss_index")
        logging.info("Saved FAISS index locally.")

        return {"status": "success", "chunks_ingested": len(chunks)}
    except Exception as e:
        logging.error(f"Error during PDF ingestion: {e}")
        return {"status": "error", "message": str(e)}
