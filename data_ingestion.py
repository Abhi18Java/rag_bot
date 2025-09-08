# D:\AI_Project\rag_app\data_ingestion.py
import logging
import asyncio
#from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import config


def upload_pdf(file_path: str):
    logging.info(f"Starting ingestion for PDF: {file_path}")
    try:
        # Ensure an event loop exists
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logging.info(f"Loaded {len(documents)} documents from PDF.")
    
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(chunks)} chunks.")

        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,   # OpenAI embedding model
            openai_api_key=config.OPENAI_API_KEY
        )
        logging.info("Initialized OpenAI embeddings.")

        vectorstore = FAISS.from_documents(chunks, embeddings)
        logging.info("Created FAISS vectorstore.")

        vectorstore.save_local("faiss_index")
        logging.info("Saved FAISS index locally.")

        return {"status": "success", "chunks_ingested": len(chunks)}
    except Exception as e:
        logging.error(f"Error during PDF ingestion: {e}")
        return {"status": "error", "message": str(e)}
