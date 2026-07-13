"""
Wrapper script to run ingestion in a subprocess.
This isolates the ingestion from Streamlit's execution context.
"""
import sys
from ingestion import ingest_pdf

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python ingest_wrapper.py <pdf_path> [display_name]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc_name = sys.argv[2] if len(sys.argv) == 3 else None
    result = ingest_pdf(pdf_path, doc_name=doc_name)
    print(result)
    sys.stdout.flush()  # Ensure output is written
    sys.exit(0)  # Clean exit
