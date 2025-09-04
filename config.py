import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
# PINECONE_ENV = os.getenv("PINECONE_ENV")
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI embedding model
LLM_MODEL = "gpt-4o-mini"  # Or any compatible model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200