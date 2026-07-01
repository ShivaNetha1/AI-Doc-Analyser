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

st.markdown(
    """
    <style>
    .main .block-container {
        padding-bottom: 7rem;
    }

    /* Style the file uploader to look compact and chat-friendly */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 0.75rem 1rem;
        margin-bottom: 1rem;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: rgba(99, 102, 241, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = EmbeddingModel()
    if st.session_state.embedding_model.model is None:
        st.error("⚠️ Embedding model failed to load. Please restart the app.")
        st.stop()
if "has_asked_question" not in st.session_state:
    st.session_state.has_asked_question = False
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

api_key = os.getenv("GROQ_API_KEY")

# ── Sidebar — Model Settings only ─────────────────────────────
MODEL_OPTIONS = {
    "Qwen 3 32B": "qwen/qwen3-32b",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "GPT OSS 120B": "openai/gpt-oss-120b",
}

st.sidebar.title("Model Settings")
selected_model_name = st.sidebar.selectbox("Select LLM Model", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[selected_model_name]

if st.sidebar.button("Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.session_state.has_asked_question = False
    st.session_state.vector_store = None
    st.session_state.processed_files = set()
    st.rerun()

# ── Main area ──────────────────────────────────────────────────
st.title("AI Document Assistant")

# Instructions (shown until user interacts)
if not st.session_state.has_asked_question and not st.session_state.processed_files:
    st.info(
        "**How to use this platform**\n\n"
        "1. Upload one or more PDF/TXT files using the uploader below.\n"
        "2. Documents are processed automatically — no extra clicks needed.\n"
        "3. Ask any question in the chat box.\n"
        "4. The assistant will reply with answers from your uploaded documents.\n"
        "5. Use follow-up questions to narrow the results or verify details.\n\n"
    )

# File uploader in the main area
uploaded_files = st.file_uploader(
    "📎 Drop your PDF or TXT files here",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    key="uploaded_files",
)

# ── Auto-process new files ─────────────────────────────────────
if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]

    if new_files:
        status_lines = []
        with st.spinner(f"Processing {len(new_files)} document(s)..."):
            for idx, file in enumerate(new_files, start=1):
                if file.name.endswith(".pdf"):
                    text = load_pdf(file)
                else:
                    text = load_text(file)

                chunks = [chunk for chunk in chunk_text(text) if chunk.strip()]
                if not chunks:
                    status_lines.append(f"⚠️ **{file.name}** — no readable text, skipped.")
                    continue

                embeddings = st.session_state.embedding_model.get_embeddings(chunks)

                if len(embeddings) == 0:
                    status_lines.append(f"⚠️ **{file.name}** — could not generate embeddings, skipped.")
                    continue

                if st.session_state.vector_store is None:
                    st.session_state.vector_store = VectorStore(len(embeddings[0]))

                st.session_state.vector_store.add_texts(chunks, embeddings, file.name)
                st.session_state.processed_files.add(file.name)
                status_lines.append(f"✅ **{file.name}** — {len(chunks)} chunks indexed.")

        # Add processing summary as an assistant chat bubble
        summary = "📄 **Documents processed**\n\n" + "\n".join(status_lines)
        st.session_state.chat_history.append({"role": "assistant", "content": summary})
        st.rerun()

# ── Display chat history ───────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ─────────────────────────────────────────────────
prompt = st.chat_input("Ask something about your documents...")

if prompt:
    if prompt.lower() in ["exit", "quit"]:
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.has_asked_question = False
        st.session_state.processed_files = set()
        st.rerun()

    st.session_state.has_asked_question = True
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vector_store is None:
        answer = "Please upload documents first using the uploader above."
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()
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
                st.rerun()
