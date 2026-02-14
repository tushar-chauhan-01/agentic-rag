"""
Agentic RAG Research Copilot - Streamlit UI

A production-ready RAG system with:
- Agentic reasoning (LangChain ReAct Agent)
- Multiple tools (retriever, summarizer, memory)
- Temperature control
- Top-k tuning
- Retrieval score transparency
- Agent reasoning display
"""

import streamlit as st
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

from agents import AgenticRAG
from retriever import retrieve_with_scores
from ingestion import ingest_pdf, clear_database

# Page config
st.set_page_config(
    page_title="Agentic RAG Research Copilot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False
if "show_scores" not in st.session_state:
    st.session_state.show_scores = False
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "model" not in st.session_state:
    st.session_state.model = "claude-opus-4-6"
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# Title
st.title("🤖 Agentic RAG Research Copilot")
st.markdown("*Intelligent document Q&A with reasoning transparency*")

# Sidebar - Controls
with st.sidebar:
    st.header("⚙️ Configuration")

    # Document upload section
    st.subheader("📄 Document Management")

    # Check if document is already loaded (directory exists AND has files)
    import os
    doc_loaded = os.path.exists("./chroma_db") and len(os.listdir("./chroma_db")) > 0

    if doc_loaded:
        st.success("✅ Document loaded and ready!")
        st.info("💡 Upload a new PDF to replace the current document")
    else:
        st.warning("⚠️ No document loaded")
        st.info("👇 Upload a PDF to get started")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    # Check if this is a new file (prevent re-processing on rerun)
    file_changed = (uploaded_file is not None and
                    (st.session_state.last_uploaded_file != uploaded_file.name))

    if uploaded_file and file_changed:
        with st.spinner("🔄 Ingesting document... This may take 30-60 seconds"):
            # Write temp file with explicit permissions
            temp_path = "temp_upload.pdf"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            # Set file permissions
            os.chmod(temp_path, 0o666)

            try:
                # Close existing agent connection if any
                if st.session_state.agent:
                    st.session_state.agent = None

                st.info("Step 1: Clearing database...")
                # Clear old database if exists
                clear_database()

                # Small delay to ensure filesystem operations complete
                import time
                time.sleep(1.0)

                st.info(f"Step 2: Ingesting {temp_path}...")
                # Run ingestion in subprocess to avoid Streamlit context issues
                import subprocess
                import sys

                # Get absolute path to python and script
                python_path = sys.executable
                script_path = os.path.abspath("ingest_wrapper.py")
                pdf_path = os.path.abspath(temp_path)

                result = subprocess.run(
                    [python_path, script_path, pdf_path],
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minutes for ingestion
                    env=os.environ.copy(),  # Pass environment variables
                    cwd=os.path.dirname(script_path)  # Set working directory
                )
                if result.returncode != 0:
                    raise Exception(f"Ingestion failed: {result.stderr}")
                st.success(f"✅ {result.stdout.strip()}")

                st.info("Step 3: Verifying ingestion...")
                # Verify database has data
                import os
                if os.path.exists("./chroma_db") and len(os.listdir("./chroma_db")) > 0:
                    st.success(f"✅ Database populated")
                else:
                    st.error("❌ Database is empty after ingestion!")
                    st.stop()

                st.success(f"📄 Loaded: {uploaded_file.name}")
                st.info("💬 You can now start asking questions below!")

                # Mark file as processed
                st.session_state.last_uploaded_file = uploaded_file.name

                # Clear old messages (fresh start with new document)
                st.session_state.messages = []

                # Reinitialize agent with current settings (fresh agent, no old memory)
                st.session_state.agent = AgenticRAG(
                    model_name=st.session_state.model,
                    temperature=st.session_state.temperature,
                    top_k=st.session_state.top_k,
                    verbose=False
                )

                # Force rerun to update UI
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(f"Full traceback:\n{traceback.format_exc()}")

    st.divider()

    # Model selection
    st.subheader("🧠 Model Settings")
    model_options = ["claude-opus-4-6", "claude-sonnet-4-5-20250929", "gpt-4", "gpt-3.5-turbo"]

    # Find current model index
    try:
        current_index = model_options.index(st.session_state.model)
    except ValueError:
        current_index = 0  # Default to Opus if current model not in list

    model_option = st.selectbox(
        "Model",
        model_options,
        index=current_index,
        help="Opus 4.6: Most powerful, Sonnet 4.5: Balanced, GPT-4: OpenAI flagship, GPT-3.5: Fast & cheap"
    )

    # Temperature control
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Lower = more focused/deterministic, Higher = more creative/random"
    )

    # Top-k control
    top_k = st.slider(
        "Top-k Retrieval",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        step=1,
        help="Number of document chunks to retrieve. Higher = more context but potential noise"
    )

    # Update settings if changed
    if (temperature != st.session_state.temperature or
        top_k != st.session_state.top_k or
        model_option != st.session_state.model):

        st.session_state.temperature = temperature
        st.session_state.top_k = top_k
        st.session_state.model = model_option

        # Update agent if it exists
        if st.session_state.agent:
            st.session_state.agent.update_settings(
                temperature=temperature,
                top_k=top_k
            )
            # If model changed, reinitialize
            if model_option != st.session_state.agent.model_name:
                st.session_state.agent = AgenticRAG(
                    model_name=model_option,
                    temperature=temperature,
                    top_k=top_k,
                    verbose=False
                )

    st.divider()

    # Display options
    st.subheader("👁️ Display Options")
    show_reasoning = st.checkbox("Show Agent Reasoning", value=False)
    show_scores = st.checkbox("Show Retrieval Scores", value=False)

    st.session_state.show_reasoning = show_reasoning
    st.session_state.show_scores = show_scores

    st.divider()

    # Actions
    st.subheader("🔧 Actions")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.agent:
            st.session_state.agent.clear_memory()
        st.rerun()

    # Info section
    st.divider()
    st.subheader("ℹ️ About")
    st.markdown("""
    **Features:**
    - 🤖 Agentic reasoning with ReAct
    - 🔍 Multi-tool system
    - 📊 Retrieval transparency
    - 🎛️ Runtime controls
    - 💭 Conversation memory

    **Built with:**
    - LangChain
    - OpenAI
    - ChromaDB
    - Streamlit
    """)

# Check if documents are loaded
import os
if not os.path.exists("./chroma_db"):
    st.warning("⚠️ No documents loaded yet!")
    st.info("""
    👈 **Get Started:**
    1. Upload a PDF using the sidebar
    2. Wait for ingestion to complete
    3. Start asking questions!

    **Recommended:** Upload "Attention is All You Need" paper for best demo results.
    """)
    st.stop()

# Main chat interface
# Only initialize agent if database exists
if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
    if not st.session_state.agent:
        # Initialize agent with default settings
        try:
            st.session_state.agent = AgenticRAG(
                model_name=st.session_state.model,
                temperature=st.session_state.temperature,
                top_k=st.session_state.top_k,
                verbose=False
            )
        except Exception as e:
            st.error(f"Error initializing agent: {str(e)}")
            st.info("💡 Make sure you have set OPENAI_API_KEY in your .env file")
            st.stop()
else:
    # Clear agent if database doesn't exist
    st.session_state.agent = None

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show reasoning if available and enabled
        if (message["role"] == "assistant" and
            "reasoning" in message and
            st.session_state.show_reasoning and
            message["reasoning"]):

            with st.expander("🧠 Agent Reasoning Process", expanded=False):
                for i, step in enumerate(message["reasoning"], 1):
                    st.markdown(f"**Step {i}: {step['tool']}**")
                    st.text(f"Input: {step['input']}")
                    st.text(f"Output: {step['output']}")
                    st.divider()

        # Show retrieval scores if available
        if (message["role"] == "assistant" and
            "scores" in message and
            st.session_state.show_scores and
            message["scores"]):

            with st.expander("📊 Retrieved Documents & Scores", expanded=False):
                for i, (doc, score) in enumerate(message["scores"], 1):
                    similarity = f"{score:.3f}"
                    st.markdown(f"**Document {i}** - Similarity: `{similarity}`")
                    st.text(doc[:300] + "...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about the document..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from agent
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Query the agent
                result = st.session_state.agent.query(prompt)

                # Display answer
                answer = result["answer"]
                st.markdown(answer)

                # Get retrieval scores if requested
                retrieval_scores = []
                if st.session_state.show_scores:
                    try:
                        docs_with_scores = retrieve_with_scores(
                            prompt,
                            top_k=st.session_state.top_k
                        )
                        retrieval_scores = [
                            (doc.page_content, score)
                            for doc, score in docs_with_scores
                        ]
                    except Exception:
                        pass

                # Show reasoning
                reasoning_steps = result.get("reasoning_steps", [])
                if reasoning_steps and st.session_state.show_reasoning:
                    with st.expander("🧠 Agent Reasoning Process", expanded=False):
                        for i, step in enumerate(reasoning_steps, 1):
                            st.markdown(f"**Step {i}: {step['tool']}**")
                            st.text(f"Input: {step['input']}")
                            st.text(f"Output: {step['output']}")
                            st.divider()

                # Show retrieval scores
                if retrieval_scores and st.session_state.show_scores:
                    with st.expander("📊 Retrieved Documents & Scores", expanded=False):
                        for i, (doc, score) in enumerate(retrieval_scores, 1):
                            similarity = f"{score:.3f}"
                            st.markdown(f"**Document {i}** - Similarity: `{similarity}`")
                            st.text(doc[:300] + "...")
                            st.divider()

                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "reasoning": reasoning_steps,
                    "scores": retrieval_scores
                })

            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "reasoning": [],
                    "scores": []
                })

# Footer
st.divider()
st.caption(f"🔧 Model: {st.session_state.model} | 🌡️ Temperature: {st.session_state.temperature} | 🔍 Top-k: {st.session_state.top_k}")
