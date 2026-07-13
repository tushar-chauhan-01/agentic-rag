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

from agents import AgenticRAG, RELEVANCE_THRESHOLD
from retriever import retrieve_with_scores, list_documents, delete_document
from ingestion import ingest_pdf, clear_database


# ---------- UI helpers ----------

GUARDRAIL_PHRASES = (
    "do not contain", "does not contain", "doesn't contain",
    "no sufficiently relevant", "not contain this information",
)


def is_guardrail_refusal(answer: str) -> bool:
    """Detect answers where the agent refused because the documents lack the info."""
    a = answer.lower()
    return any(p in a for p in GUARDRAIL_PHRASES)


def render_answer_badge(answer: str):
    """Show a guardrail badge under refusals so users understand WHY."""
    if is_guardrail_refusal(answer):
        st.info(
            "🛡️ **Guardrail active** — no document chunk was relevant enough "
            "(threshold {:.2f}), so the assistant refused instead of guessing "
            "from its own knowledge.".format(st.session_state.relevance_threshold)
        )


def render_scores(scores, threshold):
    """Retrieved chunks with relevance bars, colored against the threshold."""
    st.caption(
        f"Relevance is normalized 0–1 (higher = better). Chunks below the "
        f"guardrail threshold ({threshold:.2f}) are not trusted for answers."
    )
    for i, (doc, score) in enumerate(scores, 1):
        passed = score >= threshold
        icon = "🟢" if passed else "🔴"
        verdict = "above threshold" if passed else "below threshold"
        st.markdown(f"{icon} **Chunk {i}** — relevance `{score:.3f}` ({verdict})")
        st.progress(min(max(score, 0.0), 1.0))
        st.text(doc[:300] + ("..." if len(doc) > 300 else ""))
        st.divider()


def render_reasoning(steps):
    """Agent tool calls, with cleaned-up inputs."""
    tool_icons = {"document_retriever": "🔍", "summarizer": "📝",
                  "conversation_memory": "💭"}
    for i, step in enumerate(steps, 1):
        icon = tool_icons.get(step["tool"], "🔧")
        # Inputs look like "{'__arg1': 'the query'}" — show just the query
        raw = step["input"]
        query = raw.split("'__arg1': ", 1)[-1].strip("{}' ") if "__arg1" in raw else raw
        st.markdown(f"{icon} **Step {i}: `{step['tool']}`**")
        st.caption(f"Search query: {query}")
    if not steps:
        st.caption("No tools used — the agent answered directly (e.g. chitchat).")

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
    st.session_state.show_scores = True
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "model" not in st.session_state:
    st.session_state.model = "claude-opus-4-6"
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "relevance_threshold" not in st.session_state:
    st.session_state.relevance_threshold = RELEVANCE_THRESHOLD

# Title
st.title("🤖 Agentic RAG Research Copilot")
st.markdown("*Intelligent document Q&A with reasoning transparency*")

# Sidebar - Controls
with st.sidebar:
    st.header("⚙️ Configuration")

    # Document upload section
    st.subheader("📄 Document Management")

    # The vector DB persists across app restarts and can hold MULTIPLE
    # documents; read what it holds from chunk metadata (written at ingest
    # time), not from session-scoped state.
    all_docs = list_documents()
    selected_docs = []

    if all_docs:
        st.caption(f"📚 {len(all_docs)} document(s) in knowledge base — "
                   "tick to include in search")
        for d in all_docs:
            col_check, col_del = st.columns([5, 1])
            with col_check:
                when = (f"Ingested {d['ingested_at']}"
                        if d.get("ingested_at") else "")
                checked = st.checkbox(
                    f"**{d['name']}** — {d['chunks']} chunks",
                    value=True,
                    key=f"docsel_{d['name']}",
                    help=when,
                )
            with col_del:
                if st.button("🗑️", key=f"docdel_{d['name']}",
                             help=f"Delete {d['name']}"):
                    delete_document(d["name"])
                    st.session_state.agent = None
                    st.rerun()
            if checked:
                selected_docs.append(d["name"])

        st.info("💡 Upload another PDF to add it to the knowledge base")

        if st.button("🗑️ Clear All Documents"):
            clear_database()
            st.session_state.agent = None
            st.session_state.messages = []
            st.session_state.doc_name = None
            st.session_state.last_uploaded_file = None
            st.rerun()
    else:
        st.warning("⚠️ No documents loaded")
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

                st.info(f"Step 1: Ingesting {uploaded_file.name} "
                        "(added to the knowledge base)...")
                # Run ingestion in subprocess to avoid Streamlit context issues
                import subprocess
                import sys

                # Get absolute path to python and script
                python_path = sys.executable
                script_path = os.path.abspath("ingest_wrapper.py")
                pdf_path = os.path.abspath(temp_path)

                result = subprocess.run(
                    [python_path, script_path, pdf_path, uploaded_file.name],
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minutes for ingestion
                    env=os.environ.copy(),  # Pass environment variables
                    cwd=os.path.dirname(script_path)  # Set working directory
                )
                if result.returncode != 0:
                    raise Exception(f"Ingestion failed: {result.stderr}")
                st.success(f"✅ {result.stdout.strip()}")

                st.info("Step 2: Verifying ingestion...")
                if any(d["name"] == uploaded_file.name for d in list_documents()):
                    st.success(f"📄 Added to knowledge base: {uploaded_file.name}")
                else:
                    st.error("❌ Document not found in database after ingestion!")
                    st.stop()

                # Mark file as processed
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.doc_name = uploaded_file.name

                # Drop the agent so it is rebuilt with the new document scope
                st.session_state.agent = None

                # Force rerun to update UI
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(f"Full traceback:\n{traceback.format_exc()}")

    st.divider()

    # Model selection
    st.subheader("🧠 Model Settings")
    model_options = ["claude-opus-4-6", "claude-sonnet-4-5-20250929", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]

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

    # Guardrail threshold control
    relevance_threshold = st.slider(
        "🛡️ Guardrail Threshold",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.relevance_threshold,
        step=0.05,
        help="Minimum retrieval relevance (0-1) required to answer. Below this, "
             "the assistant says the documents don't contain the information "
             "instead of guessing. 0 disables the guardrail."
    )

    # Update settings if changed
    if (temperature != st.session_state.temperature or
        top_k != st.session_state.top_k or
        relevance_threshold != st.session_state.relevance_threshold or
        model_option != st.session_state.model):

        st.session_state.temperature = temperature
        st.session_state.top_k = top_k
        st.session_state.relevance_threshold = relevance_threshold
        st.session_state.model = model_option

        # Update agent if it exists
        if st.session_state.agent:
            st.session_state.agent.update_settings(
                temperature=temperature,
                top_k=top_k,
                relevance_threshold=relevance_threshold
            )
            # If model changed, reinitialize
            if model_option != st.session_state.agent.model_name:
                st.session_state.agent = AgenticRAG(
                    model_name=model_option,
                    temperature=temperature,
                    top_k=top_k,
                    verbose=False,
                    relevance_threshold=relevance_threshold
                )

    st.divider()

    # Display options
    st.subheader("👁️ Display Options")
    show_reasoning = st.checkbox("Show Agent Reasoning",
                                 value=st.session_state.show_reasoning)
    show_scores = st.checkbox("Show Retrieval Scores",
                              value=st.session_state.show_scores)

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
    - 📚 Multi-document knowledge base
    - 🛡️ Relevance guardrail (anti-hallucination)
    - 📊 Retrieval transparency
    - 🎛️ Runtime controls
    - 💭 Conversation memory

    **Built with:**
    - LangGraph / LangChain
    - Claude & OpenAI models
    - ChromaDB
    - Streamlit
    """)

# Check if documents are loaded
if not all_docs:
    st.warning("⚠️ No documents loaded yet!")
    st.info("""
    👈 **Get Started:**
    1. Upload a PDF using the sidebar
    2. Wait for ingestion to complete
    3. Start asking questions!
    """)
    st.stop()

if not selected_docs:
    st.warning("⚠️ No documents selected — tick at least one document "
               "in the sidebar to search it.")
    st.stop()

# Main chat interface: (re)build the agent when missing or when the
# document selection changed since it was created.
agent = st.session_state.agent
if agent is None or set(agent.doc_filter or []) != set(selected_docs):
    try:
        new_agent = AgenticRAG(
            model_name=st.session_state.model,
            temperature=st.session_state.temperature,
            top_k=st.session_state.top_k,
            verbose=False,
            relevance_threshold=st.session_state.relevance_threshold,
            doc_filter=selected_docs
        )
        # Rescoping shouldn't wipe the conversation — carry memory over
        if agent is not None:
            new_agent.memory = agent.memory
        st.session_state.agent = new_agent
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.info("💡 Make sure you have set OPENAI_API_KEY in your .env file")
        st.stop()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_answer_badge(message["content"])

            # Show reasoning if available and enabled
            if ("reasoning" in message and st.session_state.show_reasoning
                    and message["reasoning"] is not None):
                with st.expander("🧠 Agent Reasoning Process", expanded=False):
                    render_reasoning(message["reasoning"])

            # Show retrieval scores if available
            if ("scores" in message and st.session_state.show_scores
                    and message["scores"]):
                with st.expander("📊 Retrieved Chunks & Relevance", expanded=False):
                    render_scores(message["scores"],
                                  st.session_state.relevance_threshold)

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
                render_answer_badge(answer)

                # Get retrieval scores if requested (only when the agent
                # actually retrieved — chitchat has no scores to show)
                reasoning_steps = result.get("reasoning_steps", [])
                did_retrieve = any(
                    s["tool"] == "document_retriever" for s in reasoning_steps
                )
                retrieval_scores = []
                if st.session_state.show_scores and did_retrieve:
                    try:
                        docs_with_scores = retrieve_with_scores(
                            prompt,
                            top_k=st.session_state.top_k,
                            doc_names=selected_docs
                        )
                        retrieval_scores = [
                            (doc.page_content, score)
                            for doc, score in docs_with_scores
                        ]
                    except Exception:
                        pass

                # Show reasoning
                if st.session_state.show_reasoning:
                    with st.expander("🧠 Agent Reasoning Process", expanded=False):
                        render_reasoning(reasoning_steps)

                # Show retrieval scores
                if retrieval_scores and st.session_state.show_scores:
                    with st.expander("📊 Retrieved Chunks & Relevance", expanded=False):
                        render_scores(retrieval_scores,
                                      st.session_state.relevance_threshold)

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
st.caption(
    f"🔧 Model: {st.session_state.model} | "
    f"🌡️ Temperature: {st.session_state.temperature} | "
    f"🔍 Top-k: {st.session_state.top_k} | "
    f"🛡️ Guardrail: {st.session_state.relevance_threshold:.2f} | "
    f"📚 Searching {len(selected_docs)}/{len(all_docs)} document(s)"
)
