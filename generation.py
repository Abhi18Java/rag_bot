from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
import config
import logging as log

# --- Initialize vectorstore ---
embeddings = OpenAIEmbeddings(
    model=config.EMBEDDING_MODEL,
    openai_api_key=config.OPENAI_API_KEY
)
vectorstore = FAISS.load_local(
    "faiss_index", embeddings, allow_dangerous_deserialization=True
)
docs = list(vectorstore.docstore._dict.values())

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # For Semantic Search
bm25_retriever = BM25Retriever.from_documents(docs)  # For Similarity/Keyword Search
bm25_retriever.k = 5

retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)

# --- LLM ---
llm = ChatOpenAI(
    model_name=config.LLM_MODEL,
    openai_api_key=config.OPENAI_API_KEY
)

# --- Memory per user ---
user_memories = {}

# --- Custom prompt that includes chat_history ---
CUSTOM_CHAT_PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""You are a helpful assistant.

You have two sources of information:
1. Chat History → personal details about the user (e.g., name, nickname, preferences).
2. Retrieved Context → information from documents.

Rules:
- If the user statement provides new personal details (e.g., "My name is Sandeep", "Call me Sandy"), 
  acknowledge it and treat it as stored information for future turns.
- If the question is about the user (e.g., "Who am I?"), use Chat History if available. 
  • If details exist, answer directly.
  • If no details exist, politely say you don’t know.
- If the question is about documents (e.g., "Summarize this document"), ignore Chat History 
  and only use Retrieved Context.
- Never mix Chat History and Retrieved Context in the same answer.
- Always respond conversationally, not mechanically.
- If you don’t know the answer based on the above rules, say you don’t know and suggest next steps eg. "Can you provide more details?".
- Always suggest next steps to keep the conversation going.
Chat History:
{chat_history}

Retrieved Context:
{context}

User Statement or Question:
{question}

Answer:"""
)


def get_conversation_chain(query: str, user_id: str):
    log.info(f"Getting conversation chain for user: {user_id}")
    if user_id not in user_memories:
        user_memories[user_id] = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",  
            return_messages=True
        )
    memory = user_memories[user_id]

    # Retrieve context
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([d.page_content for d in docs])

    # Build chain with memory-aware prompt
    qa_chain = LLMChain(
        llm=llm,
        prompt=CUSTOM_CHAT_PROMPT,
        memory=memory
    )

    response = qa_chain.run({"context": context, "question": query})
    return response
