from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import config

def ingest_pdf(file_path: str):
    # Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Chunk Text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    
    # Embeddings
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    
    # Push to FAISS (local vector DB)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # (Optional) Save FAISS index to disk for persistence
    vectorstore.save_local("faiss_index")

    return {"status": "success", "chunks_ingested": len(chunks)}