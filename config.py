import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI embedding model
# EMBEDDING_MODEL = "models/embedding-001"  # Gemini embedding model
LLM_MODEL = "gpt-3.5-turbo-instruct"  # Or any compatible model   // models/gemini-1.5-pro-latest
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200