import streamlit as st
import requests
import logging

BACKEND_URL = "http://localhost:8000"  # Uvicorn default port
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

st.set_page_config(page_title="RAG Application", layout="wide")
st.title("📄 RAG Application")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# File uploader section
# -------------------------
uploaded_file = st.file_uploader("📤 Upload a PDF file", type=["pdf"])
if uploaded_file:
    logging.info(f"Uploading file: {uploaded_file.name}")
    with st.spinner("Uploading PDF..."):
        try:
            response = requests.post(
                f"{BACKEND_URL}/upload", files={"file": uploaded_file})
            result = response.json()
            logging.info(f"Upload response: {result}")
            if result.get("status") == "success":
                st.success(
                    f"✅ PDF ingested: {result['chunks_ingested']} chunks")
            else:
                st.error(
                    f"❌ Uploading failed: {result.get('message', 'Unknown error')}")
                logging.error(
                    f"Uploading failed: {result.get('message', 'Unknown error')}")
        except Exception as e:
            st.error(f"⚠️ Error during Uploading: {e}")
            logging.error(f"Exception during Uploading: {e}")

st.markdown("---")

# -------------------------
# Chat display area
# -------------------------
st.subheader("💬 Chat with your documents")

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"** You:** {msg['content']}")
        else:
            st.markdown(f"** AI:** {msg['content']}")

# -------------------------
# Chat input area (bottom)
# -------------------------
st.markdown("---")
col1, col2 = st.columns([8, 1])


def send_message():
    user_msg = st.session_state.user_input.strip()
    if user_msg:
        logging.info(f"User query: {user_msg}")
        st.session_state.messages.append({"role": "user", "content": user_msg})

        with st.spinner("🤔 Generating response..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query", json={"query": user_msg})
                if response.status_code == 200:
                    result = response.json()
                    logging.info(f"Query response: {result}")

                    bot_reply = result.get(
                        "response", "⚠️ No response generated.")
                    sources = result.get("sources", [])

                    answer = bot_reply
                    if sources:
                        answer += f"\n\n📚 **Sources:** {', '.join(sources)}"

                    st.session_state.messages.append(
                        {"role": "bot", "content": answer})
                else:
                    error_msg = "❌ Failed to get response from backend."
                    st.error(error_msg)
                    logging.error(f"Backend query failed: {response.text}")
                    st.session_state.messages.append(
                        {"role": "bot", "content": error_msg})
            except Exception as e:
                error_msg = f"⚠️ Error during query: {e}"
                st.error(error_msg)
                logging.error(f"Exception during query: {e}")
                st.session_state.messages.append(
                    {"role": "bot", "content": error_msg})

    # ✅ Clear input box
    st.session_state.user_input = ""


with col1:
    st.text_input(
        "Type your question here...",
        key="user_input",
        label_visibility="collapsed",
        on_change=send_message,  # runs when Enter is pressed
    )

with col2:
    st.button("Send", on_click=send_message)  # runs when button clicked
