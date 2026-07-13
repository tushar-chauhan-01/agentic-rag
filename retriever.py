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


def _doc_filter(doc_names=None):
    """Build a Chroma metadata filter scoping search to the given documents.

    None/empty means no filter (search the whole knowledge base).
    """
    if not doc_names:
        return None
    names = list(doc_names)
    if len(names) == 1:
        return {"doc_name": names[0]}
    return {"doc_name": {"$in": names}}


def retrieve_with_scores(
    query: str, top_k: int = 5, doc_names=None
) -> List[Tuple[Document, float]]:
    """
    Retrieve documents with similarity scores.

    doc_names: optional list of document names to scope the search to.

    Returns:
        List of (Document, score) tuples where score is a relevance score
        normalized to [0, 1]. Higher scores mean more similar/relevant.
    """
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search_with_relevance_scores(
        query, k=top_k, filter=_doc_filter(doc_names)
    )
    return results


def retrieve_documents_only(
    query: str, top_k: int = 5, doc_names=None
) -> List[Document]:
    """Retrieve just the documents without scores (for agent tools)."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(
        query, k=top_k, filter=_doc_filter(doc_names)
    )
    return docs


def _flush_chroma_cache():
    """Flush ChromaDB's per-process client cache. The DB may have been
    rebuilt (by the ingestion subprocess or a CLI run) since this process
    last connected, and a cached client would keep serving the old,
    deleted database's contents."""
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def list_documents():
    """
    List every document in the persistent knowledge base.

    The vector DB outlives app sessions, so this provenance must come from
    chunk metadata (written at ingest time), not from session state.

    Returns:
        List of dicts with 'name', 'ingested_at', 'chunks' (one per
        document, sorted by name). Empty list if the store is empty or
        unavailable.
    """
    import os

    try:
        _flush_chroma_cache()
        collection = get_vectorstore()._collection
        if not collection.count():
            return []
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        docs = {}
        for meta in metadatas:
            meta = meta or {}
            name = meta.get("doc_name") or os.path.basename(
                str(meta.get("source", "document"))
            )
            entry = docs.setdefault(
                name, {"name": name, "ingested_at": meta.get("ingested_at"),
                       "chunks": 0}
            )
            entry["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["name"])
    except Exception:
        return []


def delete_document(doc_name: str):
    """Delete all chunks belonging to one document."""
    _flush_chroma_cache()
    collection = get_vectorstore()._collection
    collection.delete(where={"doc_name": doc_name})
