# ── ltm.py — Long Term Memory: per-user persistent memory across chats ──
#
# Storage:  Per-user FAISS flat index (cosine similarity via L2-normalised vectors)
#           backed by a JSON file so memories survive server restarts.
# Embeddings: ibm-granite/granite-embedding-30m-english via HuggingFaceEmbeddings
# Extraction: Structured LLM call (MemoryDecision) at the END of every full turn,
#             after the agent has produced its final answer — so the extractor sees
#             the complete human message in full conversation context.
#
# Public API (imported by ontology_graph.py):
#   build_ltm_context(user_id, query) → str   (read — called in head_agent_node)
#   ltm_node(state, config)           → {}    (write — LangGraph node at END of turn)
#   LTMStore                                  (singleton — no longer exported as ltm_store)

import os
import uuid
import json
import threading
from typing import List

import faiss
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from agent.config import ltm_llm, MEMORY_EXTRACT_PROMPT, LTM_CONTEXT_PROMPT
from agent.schemas import MemoryDecision
from agent.state import ChatState


# ── Embedding model (module-level singleton, loaded once) ─────────────────────

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

_EMBED_MODEL_NAME = "ibm-granite/granite-embedding-30m-english"
_VECTOR_DIM = 384

_embed_model = HuggingFaceEmbeddings(model_name=_EMBED_MODEL_NAME)


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts → float32 L2-normalised array of shape (N, 384)."""
    vectors = _embed_model.embed_documents(texts)
    arr = np.array(vectors, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


def _embed_query(text: str) -> np.ndarray:
    """Embed a single query string → float32 L2-normalised array of shape (1, 384)."""
    vec = _embed_model.embed_query(text)
    arr = np.array([vec], dtype="float32")
    faiss.normalize_L2(arr)
    return arr


# ── Per-user FAISS store ───────────────────────────────────────────────────────

LTM_DATA_FILE = "ltm_memories.json"   # flat JSON: { user_id: [ {id, text}, … ] }


class LTMStore:
    """
    Per-user in-memory FAISS index backed by a JSON persistence file.

    Layout in memory:
        _memories  : { user_id: [ {"id": str, "text": str}, … ] }
        _indexes   : { user_id: faiss.IndexFlatIP }   (inner-product on L2-norm = cosine)

    The JSON file is read once at startup; every write flushes only the affected
    user's list back to disk (full file rewrite — acceptable for small memory sets).
    """

    def __init__(self):
        self._memories: dict[str, list] = {}   # user_id → list of {id, text}
        self._indexes:  dict[str, faiss.IndexFlatIP] = {}
        self._lock = threading.Lock()
        self._load_from_disk()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_from_disk(self):
        """Load all memories from JSON on startup and rebuild FAISS indexes."""
        if not os.path.exists(LTM_DATA_FILE):
            print("[LTM] No existing memory file found — starting fresh.")
            return
        try:
            with open(LTM_DATA_FILE, "r", encoding="utf-8") as f:
                data: dict = json.load(f)
            for user_id, items in data.items():
                if not items:
                    continue
                self._memories[user_id] = items
                self._rebuild_index(user_id)
            total = sum(len(v) for v in self._memories.values())
            print(f"[LTM] Loaded {total} memories for {len(self._memories)} users from disk.")
        except Exception as e:
            print(f"[LTM] Failed to load memory file: {e} — starting fresh.")

    def _save_to_disk(self):
        """Flush the full memory dict to JSON (called after every write)."""
        try:
            with open(LTM_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(self._memories, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[LTM] Failed to save memory file: {e}")

    # ── Index management ──────────────────────────────────────────────────────

    def _rebuild_index(self, user_id: str):
        """Rebuild FAISS IndexFlatIP from scratch for a user's current memories."""
        items = self._memories.get(user_id, [])
        if not items:
            self._indexes.pop(user_id, None)
            return
        texts = [item["text"] for item in items]
        vectors = _embed_texts(texts)
        index = faiss.IndexFlatIP(_VECTOR_DIM)
        index.add(vectors)
        self._indexes[user_id] = index

    # ── Write ─────────────────────────────────────────────────────────────────

    def put(self, user_id: str, text: str) -> str:
        """Store a new memory for a user. Returns the assigned memory id."""
        mem_id = str(uuid.uuid4())
        with self._lock:
            if user_id not in self._memories:
                self._memories[user_id] = []
            self._memories[user_id].append({"id": mem_id, "text": text})
            self._rebuild_index(user_id)
            self._save_to_disk()
        return mem_id

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(self, user_id: str, query: str, top_k: int = 5) -> List[str]:
        """
        Semantic search: return up to top_k memory texts most similar to query.
        Returns plain strings (not dicts) so callers can use them directly.
        """
        with self._lock:
            index = self._indexes.get(user_id)
            items = self._memories.get(user_id, [])
        if index is None or not items:
            return []
        query_vec = _embed_query(query)
        k = min(top_k, len(items))
        _, indices = index.search(query_vec, k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(items):
                results.append(items[idx]["text"])
        return results

    def get_all(self, user_id: str) -> List[str]:
        """Return all memory texts for a user (used for duplicate checking)."""
        with self._lock:
            return [item["text"] for item in self._memories.get(user_id, [])]

    def count(self, user_id: str) -> int:
        with self._lock:
            return len(self._memories.get(user_id, []))


# Module-level singleton — used internally; NOT exported as ltm_store
_ltm_store = LTMStore()


# ── LLM extractor ─────────────────────────────────────────────────────────────

_memory_extractor = ltm_llm.with_structured_output(MemoryDecision)


# ── Public read API ───────────────────────────────────────────────────────────

def get_user_memories(user_id: str, query: str, top_k: int = 5) -> str:
    """
    Semantic search for memories most relevant to the current query.
    Returns a formatted string ready to inject into a prompt, or "" if none.
    """
    results = _ltm_store.search(user_id, query, top_k=top_k)
    if not results:
        return ""
    return "\n".join(f"- {text}" for text in results)


def build_ltm_context(user_id: str, query: str) -> str:
    """
    Build the LTM context string to inject into head_agent_node's system prompt.
    Returns empty string if no memories exist for this user.
    """
    memories = get_user_memories(user_id, query)
    if not memories:
        return ""
    return LTM_CONTEXT_PROMPT.format(memories=memories)


# ── LangGraph node (write path) ───────────────────────────────────────────────

async def ltm_node(state: ChatState) -> dict:
    """
    LangGraph node — runs at the END of every full turn (after aggregator_node
    for pipeline queries, after head_agent_node for direct answers).

    Extracts user-specific facts from the latest human message and stores
    any new ones that aren't already in memory.

    Always returns {} so it passes state through unchanged.
    """
    user_id = state.get("user_id", "")
    if not user_id:
        print("[LTM] No user_id in state — skipping.")
        return {}

    # Get latest human message
    messages = state.get("messages", [])
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
        None,
    )
    if not last_human:
        return {}

    # Load existing memories as plain text for duplicate detection
    existing_texts = _ltm_store.get_all(user_id)
    existing_text = (
        "\n".join(f"- {t}" for t in existing_texts) if existing_texts else "(none yet)"
    )

    # Extract new memories via structured LLM call
    try:
        decision: MemoryDecision = await _memory_extractor.ainvoke([
            SystemMessage(content=MEMORY_EXTRACT_PROMPT.format(existing=existing_text)),
            HumanMessage(content=last_human),
        ])

        if not decision.should_write:
            print(f"[LTM] No new memories to store for user {user_id}")
            return {}

        new_count = 0
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                _ltm_store.put(user_id, mem.text.strip())
                new_count += 1

        if new_count:
            print(
                f"[LTM] Stored {new_count} new memories for user {user_id} "
                f"(total: {_ltm_store.count(user_id)})"
            )

    except Exception as e:
        print(f"[LTM] Memory extraction failed: {e}")

    return {}