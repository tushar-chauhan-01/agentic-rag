"""
Manually check ChromaDB contents
"""
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def check_database():
    print("=" * 70)
    print("CHECKING CHROMADB CONTENTS")
    print("=" * 70)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    persist_dir = os.path.join(repo_root, "chroma_db")

    print(f"\n📂 ChromaDB Directory: {persist_dir}")

    # Check if directory exists
    if not os.path.exists(persist_dir):
        print("❌ ChromaDB directory does not exist!")
        return

    # Check directory permissions
    import stat
    st = os.stat(persist_dir)
    permissions = oct(st.st_mode)[-3:]
    print(f"   Permissions: {permissions}")
    print(f"   Owner: {st.st_uid}")

    # List all files in the directory
    print(f"\n📄 Files in ChromaDB:")
    file_count = 0
    total_size = 0

    for root, dirs, files in os.walk(persist_dir):
        for file in files:
            filepath = os.path.join(root, file)
            size = os.path.getsize(filepath)
            total_size += size
            file_count += 1
            rel_path = os.path.relpath(filepath, persist_dir)
            print(f"   - {rel_path} ({size:,} bytes)")

    if file_count == 0:
        print("   ⚠️  No files found (database is empty)")
        return

    print(f"\n   Total: {file_count} files, {total_size:,} bytes")

    # Try to connect to ChromaDB
    print(f"\n🔌 Attempting to connect to ChromaDB...")
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
        )
        print("   ✅ Connection successful!")

        # Try to get collection info
        try:
            # Perform a test query
            results = vectorstore.similarity_search("test", k=1)
            print(f"\n📊 Database Stats:")
            print(f"   Documents found: {len(results)}")

            if len(results) > 0:
                print(f"\n   Sample document preview:")
                print(f"   {results[0].page_content[:200]}...")

            # Try to count total documents
            collection = vectorstore._collection
            count = collection.count()
            print(f"\n   Total documents in collection: {count}")

        except Exception as e:
            print(f"   ⚠️  Could not query database: {e}")

    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)

if __name__ == "__main__":
    check_database()
