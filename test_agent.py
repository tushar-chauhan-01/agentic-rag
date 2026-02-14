"""
Test script for the Agentic RAG system.
Tests the agent with a sample question.
"""

from dotenv import load_dotenv
load_dotenv()

from agents import AgenticRAG
import json

print("=" * 80)
print("TESTING AGENTIC RAG SYSTEM")
print("=" * 80)

# Initialize agent
print("\n1. Initializing AgenticRAG...")
agent = AgenticRAG(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    top_k=5,
    verbose=True  # Show agent thinking
)
print("✅ Agent initialized successfully")

# Test question
test_question = "What is the main contribution of this paper?"
print(f"\n2. Testing with question: '{test_question}'")
print("-" * 80)

# Query the agent
result = agent.query(test_question)

# Display results
print("\n3. RESULTS:")
print("=" * 80)
print("\n📝 Answer:")
print(result["answer"])

print("\n\n🧠 Reasoning Steps:")
for i, step in enumerate(result["reasoning_steps"], 1):
    print(f"\nStep {i}: {step['tool']}")
    print(f"  Input: {step['input'][:100]}...")
    print(f"  Output: {step['output'][:100]}...")

print("\n\n📊 Metadata:")
print(f"  Model: {result['model']}")
print(f"  Temperature: {result['temperature']}")
print(f"  Top-k: {result['top_k']}")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE - System is working!")
print("=" * 80)

print("\n💡 Next steps:")
print("  1. Run: streamlit run app.py")
print("  2. Open the localhost URL in your browser")
print("  3. Start chatting with the Transformer paper!")
