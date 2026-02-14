"""
This module contains placeholder functions to ingest documents. Replace with
parsing/embedding/storage logic as needed.

PDF → text → chunks → embeddings → store in Chroma
"""
from langchain_community.document_loaders import PyPDFLoader
# langchain's text-splitter package has been reorganized across releases.
# Try the common import paths in order so this module works across versions.
from importlib import import_module

RecursiveCharacterTextSplitter = None
for mod_name in (
    "langchain.text_splitter",
    "langchain.text_splitters",
    "langchain_text_splitters",
):
    try:
        mod = import_module(mod_name)
        RecursiveCharacterTextSplitter = getattr(mod, "RecursiveCharacterTextSplitter")
        break
    except Exception:
        RecursiveCharacterTextSplitter = None

if RecursiveCharacterTextSplitter is None:
    raise ImportError(
        "Could not import RecursiveCharacterTextSplitter from langchain; "
        "ensure langchain is installed and up-to-date (e.g. pip install -U langchain)."
    )
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

# Load environment variables from .env if present (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional; we'll fall back to environment variables
    pass

def clear_database():
    """Safely clear the ChromaDB database"""
    repo_root = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(repo_root, "chroma_db")

    if os.path.exists(persist_dir):
        import shutil
        shutil.rmtree(persist_dir)

    # Recreate with proper permissions
    os.makedirs(persist_dir, mode=0o777, exist_ok=True)
    os.chmod(persist_dir, 0o777)


def ingest_pdf(pdf_path):
    # Try to load the PDF; if parsing problems occur, attempt a simple
    # sanitization by rewriting the PDF with PyPDF2 and reloading.
    loader = PyPDFLoader(pdf_path)
    try:
        documents = loader.load()
    except Exception:
        # Lazy import to avoid extra dependency unless needed
        try:
            from PyPDF2 import PdfReader, PdfWriter

            reader = PdfReader(pdf_path)
            safe_path = pdf_path + ".sanitized.pdf"
            writer = PdfWriter()
            for p in reader.pages:
                writer.add_page(p)
            with open(safe_path, "wb") as f:
                writer.write(f)
            loader = PyPDFLoader(safe_path)
            documents = loader.load()
        except Exception:
            # Re-raise the original loader error if sanitization fails
            raise

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    # Ensure OPENAI_API_KEY is available
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set.\n"
            "Set it in your shell or add it to an .env file in the project root.\n"
            "Example (zsh): export OPENAI_API_KEY=\"<your_key>\"\n"
            "Or add to .env: OPENAI_API_KEY=<your_key>"
        )

    embeddings = OpenAIEmbeddings()

    # Use an absolute, repo-root relative path for Chroma persistence so the
    # DB is created consistently regardless of current working directory.
    repo_root = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(repo_root, "chroma_db")
    os.makedirs(persist_dir, mode=0o777, exist_ok=True)
    os.chmod(persist_dir, 0o777)

    # Simple ingestion without custom client (works in subprocess)
    vectorstore = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    # Explicitly close the connection to prevent locks
    try:
        if hasattr(vectorstore, '_client'):
            del vectorstore._client
        del vectorstore
    except Exception:
        pass

    # Chroma persists automatically in recent releases; explicit persist
    # calls are deprecated. We avoid calling `vectorstore.persist()` to
    # prevent deprecation warnings and potential locking issues.

    return "Ingestion complete"
