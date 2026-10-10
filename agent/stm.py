# ── stm.py — Short Term Memory: context monitoring and summarization ──
#
# Transport-agnostic version for LangGraph Platform deployment.
# All socketio.emit() calls removed — the FastAPI/UI layer handles
# progress events by subscribing to graph stream events instead.

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from agent.config import (
    CONTEXT_TPM_LIMIT,
    CONTEXT_WARN_THRESHOLD,
    CONTEXT_SUMMARIZE_RATIO,
    stm_llm,
)

STM_SUMMARIZATION_PROMPT = (
    "Summarize the conversation above. "
    "Focus on: topics discussed, key entities (people, places, organizations, technologies), "
    "important facts and data points, and the user's intent and interests. "
    "Keep it concise but information-dense. Format as flowing prose, not bullet points."
)

STM_EXTENSION_PROMPT = (
    "Extend the summary by incorporating the new conversation above. "
    "Preserve all important facts, entities, locations, dates, and decisions. "
    "Keep it concise but information-dense. Do not lose any factual details."
)


def count_message_tokens(messages: list) -> int:
    """Count total tokens across all messages using LangChain's approximation."""
    return count_tokens_approximately(messages)


def needs_summarization(messages: list) -> bool:
    """Return True if total message tokens exceed 80% of the TPM limit."""
    total = count_message_tokens(messages)
    threshold = CONTEXT_TPM_LIMIT * CONTEXT_WARN_THRESHOLD
    return total >= threshold


def get_context_usage_pct(messages: list) -> float:
    """Return current token usage as a percentage of the TPM limit (0.0 to 100.0)."""
    total = count_message_tokens(messages)
    return round((total / CONTEXT_TPM_LIMIT) * 100, 1)


def split_messages_for_summarization(messages: list) -> tuple[list, list]:
    """
    Split messages into (to_summarize, to_keep).

    - Total token budget = CONTEXT_TPM_LIMIT * CONTEXT_WARN_THRESHOLD (80% of limit)
    - Of that budget, CONTEXT_SUMMARIZE_RATIO (80%) should be summarized
    - Target summarize token budget = 80% * 80% = 64% of total limit
    - We include the partial message that crosses the boundary (ceil behaviour)
    - Remaining messages after the split boundary are kept fresh
    """
    summarize_token_budget = int(
        CONTEXT_TPM_LIMIT * CONTEXT_WARN_THRESHOLD * CONTEXT_SUMMARIZE_RATIO
    )

    accumulated = 0
    split_index = 0

    for i, msg in enumerate(messages):
        msg_tokens = count_tokens_approximately([msg])
        accumulated += msg_tokens
        if accumulated >= summarize_token_budget:
            split_index = i + 1
            break
    else:
        split_index = len(messages)

    to_summarize = messages[:split_index]
    to_keep = messages[split_index:]

    print(f"[STM] Split: {len(to_summarize)} msgs to summarize ({accumulated} tokens), "
          f"{len(to_keep)} msgs to keep")

    return to_summarize, to_keep


async def summarize_messages(
    messages: list,
    existing_summary: str | None,
) -> str:
    """
    Summarize a list of messages into a compact string.
    Pure async function — no socket/transport dependencies.
    Progress events are emitted by the FastAPI layer via graph stream events.
    """
    print(f"[STM] Starting summarization of {len(messages)} messages...")

    # Build conversation text for the summarizer
    convo_lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            convo_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            convo_lines.append(f"Assistant: {msg.content}")
        elif isinstance(msg, SystemMessage):
            convo_lines.append(f"System: {msg.content}")
    convo_text = "\n\n".join(convo_lines)

    MAX_CONVO_CHARS = 12000
    if len(convo_text) > MAX_CONVO_CHARS:
        convo_text = convo_text[:MAX_CONVO_CHARS] + "\n\n[... earlier content truncated ...]"

    # Build prompt
    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            f"New conversation:\n{convo_text}\n\n"
            + STM_EXTENSION_PROMPT
        )
    else:
        prompt = f"Conversation:\n{convo_text}\n\n" + STM_SUMMARIZATION_PROMPT

    print("[STM] Invoking summarizer LLM...")
    response = await stm_llm.ainvoke([HumanMessage(content=prompt)])

    summary = response.content.strip()
    print(f"[STM] Summary generated ({len(summary)} chars)")

    return summary


async def run_stm(
    messages: list,
    existing_summary: str | None,
) -> tuple[str | None, list, int]:
    """
    Main STM entry point. Called inside summarizer_node in the graph.

    Returns:
        - new_summary: updated summary string (or None if no summarization needed)
        - messages_to_keep: the fresh messages to retain in state
        - token_count: current token count after processing
    """
    token_count = count_message_tokens(messages)
    print(f"[STM] Current token count: {token_count} / {CONTEXT_TPM_LIMIT} "
          f"({round(token_count / CONTEXT_TPM_LIMIT * 100, 1)}%)")

    if not needs_summarization(messages):
        print("[STM] No summarization needed.")
        return None, messages, token_count

    print("[STM] Threshold reached — summarizing...")
    to_summarize, to_keep = split_messages_for_summarization(messages)

    new_summary = await summarize_messages(
        messages=to_summarize,
        existing_summary=existing_summary,
    )

    token_count_after = count_message_tokens(to_keep)
    print(f"[STM] After summarization: {token_count_after} tokens in active window")

    return new_summary, to_keep, token_count_after


def build_llm_context(
    messages: list,
    summary: str | None,
) -> list:
    """
    Build the message list to pass to any LLM node.
    If a summary exists, prepend it as a SystemMessage.
    Always returns: [SystemMessage(summary)] + messages  (if summary exists)
                 or: messages  (if no summary)
    """
    if not summary:
        return messages
    return [
        SystemMessage(content=f"[Conversation Summary]\n{summary}")
    ] + messages