# D:\AI_Project\rag_app\generation.py
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import config


def get_conversation_chain(query: str):
    # Load FAISS vectorstore
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        openai_api_key=config.OPENAI_API_KEY
    )
    vectorstore = FAISS.load_local(
        "faiss_index", embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # ✅ Add memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    llm = ChatOpenAI(
        model_name=config.LLM_MODEL,   # e.g. "gpt-4o-mini" or "gpt-3.5-turbo"
        openai_api_key=config.OPENAI_API_KEY
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return conversation_chain.run(query)
