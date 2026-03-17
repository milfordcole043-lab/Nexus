"""LLM prompts for the memory agent."""

MEMORY_SYSTEM_PROMPT = (
    "You are Nexus, a personal knowledge assistant. "
    "Answer based ONLY on the provided context from the user's personal documents. "
    "If the context doesn't contain enough information, say so. "
    "Cite sources as [Source N]. Be concise."
)


def build_answer_prompt(question: str, context_block: str) -> str:
    """Build the user prompt for answer generation."""
    return (
        f"Context from your knowledge base:\n"
        f"---\n{context_block}\n---\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
