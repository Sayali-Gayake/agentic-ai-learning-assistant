
from pathlib import Path

import streamlit as st
from langchain_community.vectorstores import FAISS
from app_embeddings import load_embeddings

INDEX_DIR = Path(__file__).parent / "faiss_index"


@st.cache_resource(show_spinner="Loading saved PDF knowledge base...")
def load_saved_knowledge_base():
    """Load the FAISS index created by our own knowledge_base.py."""

    index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        return None

    # Only load an index created by your own trusted project code.
    return FAISS.load_local(
        str(INDEX_DIR),
        load_embeddings(),
        allow_dangerous_deserialization=True,
    )


def search_all_knowledge(question, uploaded_vector_store=None, k=4):
    """Search permanent PDFs and, when available, uploaded PDFs."""

    stores = []

    saved_store = load_saved_knowledge_base()

    if saved_store is not None:
        stores.append(saved_store)

    if uploaded_vector_store is not None:
        stores.append(uploaded_vector_store)

    candidates = []

    for store in stores:
        matches = store.similarity_search_with_score(question, k=k)
        candidates.extend(matches)

    # FAISS normally returns L2 distances here; lower is better.
    candidates.sort(key=lambda item: float(item[1]))

    results = []
    seen = set()

    for document, score in candidates:
        key = (
            document.metadata.get("source", "Unknown"),
            document.metadata.get("page", "Unknown"),
            document.page_content[:160],
        )

        if key in seen:
            continue

        seen.add(key)
        results.append(document)

        if len(results) >= k:
            break

    return results
