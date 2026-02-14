"""
Automated test: Ingest PDF → Ask question → Clean up
"""
import os
import shutil
from ingestion import ingest_pdf
from agents import AgenticRAG
from dotenv import load_dotenv

load_dotenv()

def run_automated_test():
    print("=" * 70)
    print("AUTOMATED TEST: PDF Ingestion + Query")
    print("=" * 70)

    pdf_path = "Attention_is_all_you_need.pdf"

    # Step 1: Ingest PDF
    print(f"\n📄 Step 1: Ingesting '{pdf_path}'...")
    try:
        result = ingest_pdf(pdf_path)
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return

    # Step 2: Initialize Agent
    print("\n🤖 Step 2: Initializing AgenticRAG agent...")
    try:
        agent = AgenticRAG(model_name="gpt-3.5-turbo", temperature=0.7, top_k=5)
        print("✅ Agent initialized successfully")
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return

    # Step 3: Ask a question
    test_question = "What is the main contribution of this paper?"
    print(f"\n💬 Step 3: Asking question: '{test_question}'")
    print("-" * 70)

    try:
        response = agent.query(test_question)

        print("\n📝 ANSWER:")
        print(response["answer"])

        print("\n🧠 REASONING STEPS:")
        for i, step in enumerate(response.get("reasoning", []), 1):
            print(f"  {i}. {step}")

        print("\n📊 RETRIEVAL SCORES:")
        for doc_info in response.get("retrieved_docs", [])[:3]:  # Show top 3
            score = doc_info.get("score", "N/A")
            content = doc_info.get("content", "")[:100]  # First 100 chars
            print(f"  Score: {score:.4f} | Preview: {content}...")

        print("\n✅ Query completed successfully!")

    except Exception as e:
        print(f"❌ Query failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 4: Clean up
    print("\n🧹 Step 4: Cleaning up ChromaDB...")
    try:
        chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
            os.makedirs(chroma_dir, exist_ok=True)
            print("✅ Database cleaned successfully")
        else:
            print("ℹ️  No database to clean")
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)

if __name__ == "__main__":
    run_automated_test()
