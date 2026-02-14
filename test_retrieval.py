"""Quick test to verify retrieval works with the Transformer paper"""
from dotenv import load_dotenv
load_dotenv()

from retriever import get_retriever

print("Testing retrieval on 'Attention is All You Need' paper...\n")

retriever = get_retriever(top_k=3)
docs = retriever.invoke('What is the main contribution of this paper?')

print(f"✅ Retrieved {len(docs)} documents\n")
print("=" * 80)
print("📄 First retrieved chunk:")
print("=" * 80)
print(docs[0].page_content[:400])
print("\n...")
print("=" * 80)
print("\n✅ Retrieval is working! Ready to build the agentic chat interface.")
