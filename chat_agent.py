import uuid
import time
import json
import os
import threading
from langchain_core.messages import HumanMessage, AIMessage
from agent.knowledge_graph import kg_registry, KnowledgeGraph

CHAT_DATA_FILE = "chat_data.json"


class ChatAgent:
    """
    Manages per-user sessions, chats, and KG instances.
    Each user has their own isolated chats and KnowledgeGraph.
    Socket connections (sid) are mapped to user_id on login/connect.
    """

    def __init__(self):
        # { user_id: { "thread_id": str, "chats": [{id, title, timestamp}] } }
        self._lock = threading.Lock()
        self._user_data: dict = self._load()
        # { sid: user_id } — maps active socket connection to user
        self._sid_to_user: dict = {}

    def _load(self) -> dict:
        """Load user data from disk on startup."""
        if os.path.exists(CHAT_DATA_FILE):
            try:
                with open(CHAT_DATA_FILE, "r") as f:
                    data = json.load(f)
                    print(f"[ChatAgent] Loaded {len(data)} users from {CHAT_DATA_FILE}")
                    return data
            except Exception as e:
                print(f"[ChatAgent] Failed to load {CHAT_DATA_FILE}: {e}")
        return {}

    def _save(self):
        """Persist user data to disk."""
        try:
            with open(CHAT_DATA_FILE, "w") as f:
                json.dump(self._user_data, f, indent=2)
        except Exception as e:
            print(f"[ChatAgent] Failed to save {CHAT_DATA_FILE}: {e}")

    # ── Auth binding ──

    def bind_user(self, sid: str, user_id: str):
        """Bind a socket connection to a user. Called after successful auth."""
        with self._lock:
            self._sid_to_user[sid] = user_id
            # Create user data if first time or if empty
            if user_id not in self._user_data or not self._user_data[user_id].get("chats"):
                thread_id = self._new_thread_id()
                self._user_data[user_id] = {
                    "thread_id": thread_id,
                    "chats": [{"id": thread_id, "title": "New Chat", "timestamp": time.time()}]
                }
                self._save()

    def unbind_user(self, sid: str):
        """Remove socket binding on disconnect."""
        with self._lock:
            self._sid_to_user.pop(sid, None)

    def get_user_id(self, sid: str) -> str | None:
        """Get user_id for a socket. Returns None if not authenticated."""
        return self._sid_to_user.get(sid)

    # ── KG access ──

    def get_kg(self, sid: str) -> KnowledgeGraph | None:
        """Get the KG for the user associated with this socket."""
        user_id = self.get_user_id(sid)
        if not user_id:
            return None
        return kg_registry.get(user_id)

    def get_kg_by_user(self, user_id: str) -> KnowledgeGraph:
        return kg_registry.get(user_id)

    # ── Session management ──

    def _get_user_data(self, sid: str) -> dict | None:
        user_id = self.get_user_id(sid)
        if not user_id or user_id not in self._user_data:
            return None
        return self._user_data[user_id]

    # ── Chat management ──

    def new_chat(self, sid: str) -> dict | None:
        """Create a new chat, switch to it, return updated info."""
        with self._lock:
            data = self._get_user_data(sid)
            if not data:
                return None
            thread_id = self._new_thread_id()
            data["chats"].append({"id": thread_id, "title": "New Chat", "timestamp": time.time()})
            data["thread_id"] = thread_id
            self._save()
            return {"thread_id": thread_id, "chats": data["chats"]}

    def switch_chat(self, sid: str, thread_id: str) -> dict | None:
        """Switch active chat. Returns None if thread_id doesn't belong to this user."""
        with self._lock:
            data = self._get_user_data(sid)
            if not data:
                return None
            if not any(c["id"] == thread_id for c in data["chats"]):
                return None
            data["thread_id"] = thread_id
            self._save()
            return {"thread_id": thread_id}
    
    def rename_chat(self, sid: str, thread_id: str, title: str) -> list | None:
        """Rename a chat. Returns updated chats list or None if not found."""
        with self._lock:
            data = self._get_user_data(sid)
            if not data:
                return None
            for chat in data["chats"]:
                if chat["id"] == thread_id:
                    chat["title"] = title
                    self._save()
                    return data["chats"]
            return None

    def delete_chat(self, sid: str, thread_id: str) -> dict | None:
        """Delete a chat. Switches to nearest chat or creates new one. Returns {thread_id, chats}."""
        with self._lock:
            data = self._get_user_data(sid)
            if not data:
                return None
            data["chats"] = [c for c in data["chats"] if c["id"] != thread_id]
            # If deleted chat was active, switch to most recent or create new
            if data["thread_id"] == thread_id:
                if data["chats"]:
                    data["thread_id"] = data["chats"][-1]["id"]
                else:
                    new_thread_id = self._new_thread_id()
                    data["chats"].append({"id": new_thread_id, "title": "New Chat", "timestamp": time.time()})
                    data["thread_id"] = new_thread_id
            self._save()
            return {"thread_id": data["thread_id"], "chats": data["chats"]}

    def get_active_thread(self, sid: str) -> str | None:
        data = self._get_user_data(sid)
        return data["thread_id"] if data else None

    def get_config(self, sid: str) -> dict | None:
        """Return LangGraph config for the active chat of this user."""
        thread_id = self.get_active_thread(sid)
        if not thread_id:
            return None
        return {"configurable": {"thread_id": thread_id}}

    def set_chat_title(self, sid: str, thread_id: str, first_message: str) -> list | None:
        """Set title from first user message. Returns updated chats if title was set, else None."""
        with self._lock:
            data = self._get_user_data(sid)
            if not data:
                return None
            for chat in data["chats"]:
                if chat["id"] == thread_id and chat["title"] == "New Chat":
                    chat["title"] = first_message[:40] + ("..." if len(first_message) > 40 else "")
                    self._save()
                    return data["chats"]
            return None

    def get_chats(self, sid: str) -> list:
        data = self._get_user_data(sid)
        return data["chats"] if data else []

    def get_session_info(self, sid: str) -> dict | None:
        """Return session info for session_init event."""
        data = self._get_user_data(sid)
        if not data:
            return None
        return {"thread_id": data["thread_id"], "chats": data["chats"]}

    # ── Message history ──

    def get_chat_messages(self, thread_id: str, graph) -> list:
        """Replay messages from LangGraph checkpointer for a given thread."""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snap = graph.get_state(config)
            messages = snap.values.get("messages", [])
            result = []
            for m in messages:
                if isinstance(m, HumanMessage):
                    result.append({"role": "user", "content": m.content})
                elif isinstance(m, AIMessage) and m.content:
                    result.append({"role": "agent", "content": m.content})
            return result
        except Exception:
            return []

    # ── Internal ──

    @staticmethod
    def _new_thread_id() -> str:
        return str(uuid.uuid4())


# Module-level singleton — imported by app.py
chat_agent = ChatAgent()