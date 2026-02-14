"""
Conversation memory system for Agentic RAG.

Manages conversation history, context, and provides memory as a tool for the agent.
"""

from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.chat_history import BaseChatMessageHistory


class ConversationMemory:
    """Simple in-memory conversation storage."""

    def __init__(self):
        self.messages: List[BaseMessage] = []
        self.conversation_pairs: List[Dict[str, str]] = []

    def add_user_message(self, message: str):
        """Add a user message to history."""
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        """Add an AI response to history."""
        self.messages.append(AIMessage(content=message))

        # Also store as Q&A pair for easy access
        if len(self.messages) >= 2:
            last_human = None
            for msg in reversed(self.messages[:-1]):
                if isinstance(msg, HumanMessage):
                    last_human = msg.content
                    break

            if last_human:
                self.conversation_pairs.append({
                    "question": last_human,
                    "answer": message
                })

    def get_messages(self) -> List[BaseMessage]:
        """Get all messages."""
        return self.messages

    def get_recent_context(self, num_turns: int = 3) -> str:
        """
        Get recent conversation context as formatted string.

        Args:
            num_turns: Number of recent Q&A turns to include

        Returns:
            Formatted conversation history
        """
        if not self.conversation_pairs:
            return "No previous conversation."

        recent = self.conversation_pairs[-num_turns:]
        context_parts = []

        for i, pair in enumerate(recent, 1):
            context_parts.append(f"Turn {i}:")
            context_parts.append(f"Q: {pair['question']}")
            context_parts.append(f"A: {pair['answer'][:200]}...")  # Truncate long answers
            context_parts.append("")

        return "\n".join(context_parts)

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context."""
        if not self.conversation_pairs:
            return "No conversation history yet."

        num_turns = len(self.conversation_pairs)
        last_question = self.conversation_pairs[-1]["question"] if self.conversation_pairs else "None"

        return f"Conversation has {num_turns} turns. Last question: {last_question}"

    def search_history(self, query: str) -> str:
        """
        Search conversation history for relevant past Q&As.
        Simple keyword-based search.
        """
        if not self.conversation_pairs:
            return "No conversation history to search."

        query_lower = query.lower()
        relevant_pairs = []

        for pair in self.conversation_pairs:
            if (query_lower in pair["question"].lower() or
                query_lower in pair["answer"].lower()):
                relevant_pairs.append(pair)

        if not relevant_pairs:
            return f"No relevant history found for: {query}"

        # Return most recent relevant pairs
        result_parts = ["Found relevant conversation history:"]
        for pair in relevant_pairs[-3:]:  # Last 3 relevant
            result_parts.append(f"\nQ: {pair['question']}")
            result_parts.append(f"A: {pair['answer'][:150]}...")

        return "\n".join(result_parts)

    def clear(self):
        """Clear all conversation history."""
        self.messages = []
        self.conversation_pairs = []


def create_memory_tool_description() -> str:
    """Description for the memory tool."""
    return """Useful for accessing previous conversation history.
    Use this when the user asks about something mentioned earlier in the conversation,
    or when you need context from previous questions and answers.
    Input should be a search query or topic to look for in conversation history."""
