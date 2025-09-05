from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
import retrieval
import config

def generate_response(query: str):
    llm = GoogleGenerativeAI(model=config.LLM_MODEL, google_api_key=config.GEMINI_API_KEY)
    docs = retrieval.retrieve_docs(query)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retrieval.get_retriever())
    response = qa_chain.run(query)
    return {"response": response, "sources": [doc.metadata for doc in docs]}