import streamlit as st
from sentence_transformers import SentenceTransformer


@st.cache_resource(show_spinner="Loading embedding model...")
def _load_model(model_name: str):
    """Load the SentenceTransformer model once and cache across reruns."""
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        st.error(
            f"Failed to load embedding model '{model_name}': {e}\n\n"
            "This is usually a transient infrastructure issue. "
            "Try restarting the app."
        )
        return None


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = _load_model(model_name)

    def get_embeddings(self, texts):
        if self.model is None:
            raise RuntimeError(
                "Embedding model is not loaded. Please restart the app."
            )
        return self.model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64,
            convert_to_numpy=True,
        )
