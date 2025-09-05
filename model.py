# model.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query: str

class SourceMetadata(BaseModel):
    # You can expand this as needed
    source: Optional[str] = None
    # Add more fields if your doc.metadata contains more info

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
