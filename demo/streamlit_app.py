"""Minimal chat UI proving the end-to-end pipeline: question -> grounded, cited answer.

Usage: streamlit run demo/streamlit_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from rag.config import load_config
from rag.embeddings import get_embedder
from rag.generation import generate_answer
from rag.retrievers import get_retriever
from rag.vectorstore import load_vectorstore

load_dotenv()
st.set_page_config(page_title="RAG Mid-Sem Demo")
st.title("RAG over SQuAD — Mid-Sem Demo")


@st.cache_resource
def load_pipeline():
    cfg = load_config()
    embedder = get_embedder(cfg)
    store = load_vectorstore(embedder, cfg)
    retriever = get_retriever(cfg["retriever"]["strategy"], cfg, vectorstore=store)
    return cfg, retriever


cfg, retriever = load_pipeline()
st.caption(
    f"strategy={cfg['retriever']['strategy']} · top_k={cfg['retriever']['top_k']} · "
    f"chunk_size={cfg['chunking']['chunk_size']} · llm={cfg['generation']['provider']}/{cfg['generation']['model']}"
)

question = st.text_input("Ask a question about Nikola Tesla, the Amazon rainforest, Black Death or French & Indian War:")
if question:
    with st.spinner("Retrieving and generating..."):
        chunks = retriever.retrieve(question, top_k=cfg["retriever"]["top_k"])
        result = generate_answer(question, chunks, cfg)

    st.markdown(f"**Answer:** {result['answer']}")
    if chunks:
        st.markdown("**Cited passages:**")
        for i, c in enumerate(chunks):
            with st.expander(f"[{i + 1}] {c.title} (score={c.score:.3f})"):
                st.write(c.text)
