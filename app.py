# from sentence_transformers import SentenceTransformer
# from groq import Groq
# from core.prompts import ANSWER_PROMPT
# from groq import Groq
# from core.prompts import VALIDATOR_PROMPT
# import faiss
# import numpy as np
# import PyPDF2

import streamlit as st
import os
from dotenv import load_dotenv
from utils.pdf_loader import load_pdf
from utils.text_loader import load_text
from utils.helpers import chunk_text
from core.embeddings import EmbeddingModel
from core.vectorstore import VectorStore
from core.memory import MemoryManager
from agents.router_agent import RouterAgent
from agents.retrieval_agent import RetrievalAgent
from agents.calculator_agent import CalculatorAgent
from agents.validator_agent import ValidatorAgent

load_dotenv()

st.set_page_config(page_title="AI Document Assistant", layout="wide")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = EmbeddingModel()
if "has_asked_question" not in st.session_state:
    st.session_state.has_asked_question = False

api_key = os.getenv("GROQ_API_KEY")

MODEL_OPTIONS = {
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Qwen 3 32B": "qwen/qwen3-32b",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "GPT OSS 120B": "openai/gpt-oss-120b",
}

st.sidebar.title("Model Settings")
selected_model_name = st.sidebar.selectbox("Select LLM Model", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[selected_model_name]

st.sidebar.title("Document Upload")
uploaded_files = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"], accept_multiple_files=True, key="uploaded_files")

if st.sidebar.button("Process Documents"):
    if uploaded_files:
        with st.spinner("Processing documents... this may take a minute for large files"):
            total_files = len(uploaded_files)
            for idx, file in enumerate(uploaded_files, start=1):
                st.sidebar.info(f"Processing {file.name} ({idx}/{total_files})")
                if file.name.endswith(".pdf"):
                    text = load_pdf(file)
                else:
                    text = load_text(file)

                chunks = [chunk for chunk in chunk_text(text) if chunk.strip()]
                if len(chunks) > 500:
                    st.sidebar.warning(
                        "Large document detected — indexing may take several minutes."
                    )
                embeddings = st.session_state.embedding_model.get_embeddings(chunks)

                if st.session_state.vector_store is None:
                    st.session_state.vector_store = VectorStore(len(embeddings[0]))

                st.session_state.vector_store.add_texts(chunks, embeddings, file.name)
        st.sidebar.success("Documents processed!")

st.title("AI Document Assistant")

if not st.session_state.has_asked_question:
    st.info(
        "**How to use this platform**\n\n"
        "1. Upload one or more PDF/TXT files in the sidebar.\n"
        "2. Click **Process Documents** to build the document index.\n"
        "3. Ask any question in the chat box below.\n"
        "4. The assistant will reply with answers from your uploaded documents.\n"
        "5. Use follow-up questions to narrow the results or verify details.\n\n"
    )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_col, action_col = st.columns([5, 1])
with action_col:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []

with chat_col:
    prompt = st.chat_input("Ask something about your documents...")

if prompt:
    if prompt.lower() in ["exit", "quit"]:
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.has_asked_question = False
        st.rerun()
    
    st.session_state.has_asked_question = True
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store is None:
        st.error("Please upload and process documents first.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                memory = MemoryManager(api_key, model_id)
                standalone_query = memory.condense_query(prompt, st.session_state.chat_history[:-1])
                
                router = RouterAgent(api_key, model_id)
                intent = router.route(standalone_query, st.session_state.chat_history[:-1])
                
                if intent == "REJECT":
                    answer = "I could not find this information in the provided documents"
                else:
                    query_embedding = st.session_state.embedding_model.get_embeddings([standalone_query])[0]
                    context = st.session_state.vector_store.search(query_embedding)
                    
                    if intent == "CALCULATE":
                        agent = CalculatorAgent(api_key, model_id)
                        answer = agent.calculate(standalone_query, context)
                    else:
                        agent = RetrievalAgent(api_key, model_id)
                        answer = agent.generate_answer(standalone_query, context)
                    
                    validator = ValidatorAgent(api_key, model_id)
                    if not validator.validate(answer, context):
                        answer = "I could not find this information in the provided documents"

                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
