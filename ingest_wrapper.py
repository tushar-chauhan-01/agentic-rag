"""
Wrapper script to run ingestion in a subprocess.
This isolates the ingestion from Streamlit's execution context.
"""
import sys
from ingestion import ingest_pdf

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_wrapper.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = ingest_pdf(pdf_path)
    print(result)
    sys.stdout.flush()  # Ensure output is written
    sys.exit(0)  # Clean exit
