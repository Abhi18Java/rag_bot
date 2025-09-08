from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import OpenAI
import retrieval
import config

def generate_response(query: str):
    llm = OpenAI(model=config.LLM_MODEL, openai_api_key=config.OPENAI_API_KEY)
    docs = retrieval.retrieve_docs(query)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retrieval.get_retriever())
    response = qa_chain.run(query)
    return {"response": response, "sources": [doc.metadata for doc in docs]}