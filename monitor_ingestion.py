"""
Monitor ChromaDB for ingestion activity
"""
import os
import time
from datetime import datetime

def get_chroma_info():
    chroma_dir = "chroma_db"

    # Get directory size
    total_size = 0
    file_count = 0
    files = []

    if os.path.exists(chroma_dir):
        for dirpath, dirnames, filenames in os.walk(chroma_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
                    file_count += 1
                    files.append(filename)

    return {
        "size": total_size,
        "file_count": file_count,
        "files": files
    }

def format_size(size_bytes):
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def monitor():
    print("=" * 70)
    print("MONITORING ChromaDB - Watching for PDF ingestion")
    print("=" * 70)
    print("\n📊 Initial State:")

    initial_info = get_chroma_info()
    print(f"   Size: {format_size(initial_info['size'])}")
    print(f"   Files: {initial_info['file_count']}")

    if initial_info['file_count'] > 0:
        print(f"   Files in DB: {', '.join(initial_info['files'][:5])}")

    print("\n👀 Watching for changes... (Press Ctrl+C to stop)")
    print("   Upload a PDF in the Streamlit UI at http://localhost:8503")
    print("-" * 70)

    last_info = initial_info

    try:
        while True:
            time.sleep(2)  # Check every 2 seconds
            current_info = get_chroma_info()

            # Check if something changed
            if (current_info['size'] != last_info['size'] or
                current_info['file_count'] != last_info['file_count']):

                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n✅ [{timestamp}] CHANGE DETECTED!")
                print(f"   Size: {format_size(last_info['size'])} → {format_size(current_info['size'])}")
                print(f"   Files: {last_info['file_count']} → {current_info['file_count']}")

                if current_info['file_count'] > 0:
                    print(f"   Files in DB: {', '.join(current_info['files'][:10])}")

                if current_info['size'] > 0:
                    print("\n🎉 PDF SUCCESSFULLY INGESTED INTO DATABASE!")
                    print("   The vector store now contains embedded document chunks")
                    print("   Ready to answer questions about the uploaded PDF")

                last_info = current_info

    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("MONITORING STOPPED")
        print("=" * 70)

        final_info = get_chroma_info()
        print(f"\n📊 Final State:")
        print(f"   Size: {format_size(final_info['size'])}")
        print(f"   Files: {final_info['file_count']}")

        if final_info['file_count'] > 0:
            print(f"\n   ✅ Database contains ingested data")
            print(f"   Files: {', '.join(final_info['files'])}")
        else:
            print(f"\n   ℹ️  Database is empty (no PDF uploaded)")

if __name__ == "__main__":
    monitor()
