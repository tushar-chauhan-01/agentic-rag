"""
Retriever module for Agentic RAG.

Provides document retrieval with similarity scores for transparency.
The retriever bridges stored knowledge (vector DB) and the LLM.
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from typing import List, Tuple
from langchain_core.documents import Document


def get_vectorstore():
    """Get the Chroma vectorstore instance."""
    import os

    # Use absolute repo-root path for consistency with ingestion
    repo_root = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(repo_root, "chroma_db")
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            "No documents loaded. Please upload a PDF first through the Streamlit interface."
        )

    embeddings = OpenAIEmbeddings()

    # Use langchain-chroma which handles ChromaDB properly
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )

    return vectorstore


def get_retriever(top_k=5):
    """Get a standard retriever (for simple use cases)."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def retrieve_with_scores(query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with similarity scores.

    Returns:
        List of (Document, score) tuples where score is the similarity score.
        Higher scores mean more similar/relevant.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return results


def retrieve_documents_only(query: str, top_k: int = 5) -> List[Document]:
    """Retrieve just the documents without scores (for agent tools)."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=top_k)
    return docs
