from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import config

def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL, google_api_key=config.GEMINI_API_KEY)
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever(search_kwargs={"k": 5})  # Top 5 results

def retrieve_docs(query: str):
    retriever = get_retriever()
    docs = retriever.get_relevant_documents(query)
    return docs