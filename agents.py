"""
Agentic RAG orchestration with LangChain ReAct Agent.

Implements an agent with multiple tools:
- Retriever: Search document database
- Summarizer: Summarize retrieved content
- Memory: Access conversation history
"""

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool
from retriever import retrieve_documents_only
from memory import ConversationMemory
from typing import Dict, Any, Optional
import os


class AgenticRAG:
    """
    Agentic RAG system with reasoning capabilities.

    The agent can:
    - Decide when to retrieve from documents
    - Summarize content
    - Access conversation memory
    - Reason about the best approach to answer
    """

    def __init__(
        self,
        model_name: str = "claude-opus-4-6",
        temperature: float = 0.7,
        top_k: int = 5,
        verbose: bool = True
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.verbose = verbose

        # Initialize LLM based on model provider
        if model_name.startswith("claude"):
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=4096
            )
        elif model_name.startswith("gpt"):
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                max_tokens=2000
            )
        else:
            # Default to Anthropic
            self.llm = ChatAnthropic(
                model=model_name,
                temperature=temperature,
                max_tokens=4096
            )

        # Initialize memory
        self.memory = ConversationMemory()

        # Create tools
        self.tools = self._create_tools()

        # Create agent
        self.agent_executor = self._create_agent()

    def _create_tools(self):
        """Create the tools available to the agent."""

        # Tool 1: Document Retriever
        def retriever_func(query: str) -> str:
            """Retrieve relevant documents from the knowledge base."""
            try:
                docs = retrieve_documents_only(query, top_k=self.top_k)
                if not docs:
                    return "No relevant documents found."

                result_parts = [f"Found {len(docs)} relevant documents:\n"]
                for i, doc in enumerate(docs, 1):
                    content_preview = doc.page_content[:300].replace("\n", " ")
                    result_parts.append(f"\n[Document {i}]\n{content_preview}...")

                return "\n".join(result_parts)
            except Exception as e:
                return f"Error retrieving documents: {str(e)}"

        retriever_tool = Tool(
            name="document_retriever",
            func=retriever_func,
            description="""Search the document database for relevant information.
            Use this when you need to find specific information from the uploaded documents.
            Input should be a clear search query or question.
            Returns relevant document excerpts."""
        )

        # Tool 2: Summarizer
        def summarizer_func(text: str) -> str:
            """Summarize text content."""
            try:
                # Use the LLM to summarize
                from langchain_core.messages import HumanMessage
                summary_prompt = f"""Summarize the following text concisely in 2-3 sentences:

{text[:2000]}

Summary:"""
                # Handle both invoke and predict methods
                if hasattr(self.llm, 'invoke'):
                    result = self.llm.invoke([HumanMessage(content=summary_prompt)])
                    return result.content.strip()
                else:
                    return self.llm.predict(summary_prompt).strip()
            except Exception as e:
                return f"Error summarizing: {str(e)}"

        summarizer_tool = Tool(
            name="summarizer",
            func=summarizer_func,
            description="""Summarize long text into concise points.
            Use this when you need to condense retrieved information.
            Input should be the text to summarize.
            Returns a concise summary."""
        )

        # Tool 3: Memory Search
        def memory_func(query: str) -> str:
            """Search conversation history."""
            return self.memory.search_history(query)

        memory_tool = Tool(
            name="conversation_memory",
            func=memory_func,
            description="""Access previous conversation history.
            Use this when the user references something from earlier in the conversation,
            or when context from previous Q&As would be helpful.
            Input should be a keyword or topic to search for.
            Returns relevant past exchanges."""
        )

        return [retriever_tool, summarizer_tool, memory_tool]

    def _create_agent(self):
        """Create the ReAct agent with tools using LangGraph."""

        # System message for the agent
        from langchain_core.messages import SystemMessage

        system_message = """You are a helpful AI research assistant with access to a document database.

Important Instructions:
- ALWAYS use the document_retriever tool IMMEDIATELY when asked any question about documents
- DO NOT ask permission to search - just search automatically
- Answer directly from the retrieved documents - DO NOT use the summarizer tool
- Be concise, direct, and factual
- If information isn't in the documents, state that clearly
- Never ask "Would you like me to search?" - always search proactively"""

        # Create the ReAct agent using LangGraph
        agent_executor = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SystemMessage(content=system_message)
        )

        return agent_executor

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the agentic RAG system.

        Args:
            question: User's question

        Returns:
            Dict containing answer, reasoning steps, and metadata
        """
        # Get conversation context
        context = self.memory.get_recent_context(num_turns=2)

        # Add context to question if there is any
        question_with_context = question
        if context and context != "No previous conversation.":
            question_with_context = f"Context from previous conversation:\n{context}\n\nCurrent question: {question}"

        # Run the agent
        try:
            from langchain_core.messages import HumanMessage

            # Invoke agent with message
            result = self.agent_executor.invoke({
                "messages": [HumanMessage(content=question_with_context)]
            })

            # Extract answer from messages
            messages = result.get("messages", [])
            answer = "I couldn't generate an answer."

            if messages:
                # Get the last AI message
                for msg in reversed(messages):
                    if hasattr(msg, 'content') and msg.content and not isinstance(msg, HumanMessage):
                        answer = msg.content
                        break

            # Extract reasoning steps from messages
            reasoning_steps = []
            for msg in messages:
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        reasoning_steps.append({
                            "tool": tool_call.get("name", "unknown"),
                            "input": str(tool_call.get("args", {}))[:100],
                            "output": "Tool executed"
                        })

            # Update memory
            self.memory.add_user_message(question)
            self.memory.add_ai_message(answer)

            return {
                "answer": answer,
                "reasoning_steps": reasoning_steps,
                "model": self.model_name,
                "temperature": self.temperature,
                "top_k": self.top_k
            }

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            return {
                "answer": error_msg,
                "reasoning_steps": [],
                "error": str(e)
            }

    def update_settings(self, temperature: Optional[float] = None, top_k: Optional[int] = None):
        """Update agent settings."""
        if temperature is not None:
            self.temperature = temperature
            self.llm.temperature = temperature

        if top_k is not None:
            self.top_k = top_k

    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()

    def get_memory_summary(self) -> str:
        """Get conversation summary."""
        return self.memory.get_conversation_summary()
