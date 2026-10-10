from flask import Flask, render_template, request
from flask_socketio import SocketIO
import asyncio
import time
import threading
import logging
import os
from collections import deque                                      # ← added
from langchain_core.messages import HumanMessage, AIMessage
from agent.config import RATE_LIMIT_MAX,RATE_LIMIT_WINDOW,CONTEXT_TPM_LIMIT
from agent.stm import get_context_usage_pct



from agent.ontology_graph import build_graph
from chat_agent import chat_agent
from auth import login, signup, logout, verify_token
from socket_registry import register as register_socket, unregister as unregister_socket

app = Flask(__name__, template_folder="UI", static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

graph = None
loop = None



_rate_buckets: dict = {}     # { user_id: deque of timestamps }
_rate_lock = threading.Lock()

# ── Socket registry — avoids storing socketio in graph state ──
_socket_registry: dict = {}  # { sid: socketio }


def _is_rate_limited(user_id: str) -> bool:
    """Return True if user exceeded RATE_LIMIT_MAX requests in the last RATE_LIMIT_WINDOW seconds."""
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets.setdefault(user_id, deque())
        while bucket and now - bucket[0] > RATE_LIMIT_WINDOW:
            bucket.popleft()
        if len(bucket) >= RATE_LIMIT_MAX:
            return True
        bucket.append(now)
        return False

# ── Per-user logging ──
os.makedirs("logs", exist_ok=True)
_user_loggers = {}

def get_user_logger(user_id: str, username: str) -> logging.Logger:
    if user_id not in _user_loggers:
        logger = logging.getLogger(user_id)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        handler = logging.FileHandler(f"logs/{username}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
        _user_loggers[user_id] = logger
    return _user_loggers[user_id]


# ── Routes ──

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/")
def index():
    return render_template("login.html")

@app.route("/app")
def app_page():
    return render_template("index.html")

@app.route("/app/chat/<thread_id>")
def chat_page(thread_id):
    return render_template("index.html")


# ── Helpers ──

def get_kg_data(sid: str) -> dict:
    kg = chat_agent.get_kg(sid)
    if not kg:
        return {"nodes": [], "edges": []}
    nodes = [
        {"id": name, "label": f"{name}\n({e.type})", "properties": e.properties}
        for name, e in kg.entities.items()
    ]
    edges = [
        {
            "from": r.source,
            "to": r.target,
            "label": r.relation,
            "metadata": r.metadata
        }
        for r in kg.relationships
        if r.source in kg.entities and r.target in kg.entities
    ]
    return {"nodes": nodes, "edges": edges}


def require_auth(sid: str) -> bool:
    if chat_agent.get_user_id(sid) is None:
        socketio.emit("auth_required", {}, to=sid)
        return False
    return True


# ── Auth socket events ──

@socketio.on("authenticate")
def handle_authenticate(data):
    sid = request.sid
    token = data.get("token", "")
    user = verify_token(token)
    if not user:
        socketio.emit("auth_required", {}, to=sid)
        return
    chat_agent.bind_user(sid, user["user_id"])
    session_info = chat_agent.get_session_info(sid)
    log = get_user_logger(user["user_id"], user["username"])
    log.info(f"[CONNECT] sid={sid}")
    socketio.emit("session_init", {
        "username": user["username"],
        "thread_id": session_info["thread_id"],
        "chats": session_info["chats"]
    }, to=sid)
    socketio.emit("kg_update", get_kg_data(sid), to=sid)


@socketio.on("login")
def handle_login(data):
    sid = request.sid
    result = login(data.get("username", ""), data.get("password", ""))
    if not result["success"]:
        socketio.emit("auth_error", {"error": result["error"]}, to=sid)
        return
    socketio.emit("auth_success", {
        "token": result["token"],
        "username": result["username"],
        "remember": data.get("remember", False)
    }, to=sid)


@socketio.on("signup")
def handle_signup(data):
    sid = request.sid
    result = signup(data.get("username", ""), data.get("password", ""))
    if not result["success"]:
        socketio.emit("auth_error", {"error": result["error"]}, to=sid)
        return
    log = get_user_logger(result.get("user_id", "unknown"), result["username"])
    log.info(f"[SIGNUP] sid={sid}")
    socketio.emit("auth_success", {
        "token": result["token"],
        "username": result["username"],
        "remember": data.get("remember", False)
    }, to=sid)


@socketio.on("logout")
def handle_logout(data):
    sid = request.sid
    token = data.get("token", "")
    user_id = chat_agent.get_user_id(sid)
    if user_id and user_id in _user_loggers:
        _user_loggers[user_id].info(f"[LOGOUT] sid={sid}")
    logout(token)
    chat_agent.unbind_user(sid)
    socketio.emit("logged_out", {}, to=sid)


# ── App socket events ──

@socketio.on("connect")
def on_connect():
    pass


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    user_id = chat_agent.get_user_id(sid)
    if user_id and user_id in _user_loggers:
        _user_loggers[user_id].info(f"[DISCONNECT] sid={sid}")
    chat_agent.unbind_user(sid)
    unregister_socket(sid)


@socketio.on("new_chat")
def handle_new_chat():
    sid = request.sid
    if not require_auth(sid):
        return
    result = chat_agent.new_chat(sid)
    if not result:
        return
    user_id = chat_agent.get_user_id(sid)
    if user_id in _user_loggers:
        _user_loggers[user_id].info(f"[NEW_CHAT] thread={result.get('thread_id')}")
    socketio.emit("chat_created", result, to=sid)


@socketio.on("switch_chat")
def handle_switch_chat(data):
    sid = request.sid
    if not require_auth(sid):
        return
    thread_id = data.get("thread_id")
    result = chat_agent.switch_chat(sid, thread_id)
    if result is None:
        return
    messages = chat_agent.get_chat_messages(thread_id, graph)
    socketio.emit("chat_switched", {
        "thread_id": thread_id,
        "messages": messages
    }, to=sid)


@socketio.on("rename_chat")
def handle_rename_chat(data):
    sid = request.sid
    if not require_auth(sid):
        return
    thread_id = data.get("thread_id")
    title = data.get("title", "").strip()
    if not thread_id or not title:
        return
    updated_chats = chat_agent.rename_chat(sid, thread_id, title)
    if updated_chats:
        socketio.emit("chat_renamed", {"chats": updated_chats}, to=sid)


@socketio.on("delete_chat")
def handle_delete_chat(data):
    sid = request.sid
    if not require_auth(sid):
        return
    thread_id = data.get("thread_id")
    if not thread_id:
        return
    result = chat_agent.delete_chat(sid, thread_id)
    if result:
        socketio.emit("chat_deleted", result, to=sid)


@socketio.on("user_message")
def handle_message(data):
    sid = request.sid
    if not require_auth(sid):
        return

    question = data.get("message", "").strip()
    if not question:
        return

    user_id = chat_agent.get_user_id(sid)  # ← moved up before rate check

    if _is_rate_limited(user_id):          # ← rate limit guard
        socketio.emit("stream_end", {"prefix": "System", "time": "0.00s","rate_limit_window": RATE_LIMIT_WINDOW}, to=sid)
        return

    thread_id = chat_agent.get_active_thread(sid)
    config = chat_agent.get_config(sid)

    updated_chats = chat_agent.set_chat_title(sid, thread_id, question)
    if updated_chats:
        socketio.emit("chat_updated", {"chats": updated_chats}, to=sid)

    socketio.emit("status", {"text": "Thinking..."}, to=sid)
    start = time.time()

    # user_id already resolved above — removed duplicate here
    log = _user_loggers.get(user_id)
    if log:
        log.info(f"[QUERY] thread={thread_id} | {question}")

    async def run_streaming():
        prefix = "AI"
        head_route = ""
        ack_emitted = False
        current_node = None
        nodes_visited = []

        # ── Aggregator streaming gate ──
        # kg_sufficient=False: aggregator makes 2 LLM calls
        #   1st = entity_extractor_llm (ainvoke) -- skip tokens
        #   2nd = aggregator_llm (astream) -- emit tokens
        # kg_sufficient=True: entity extractor is skipped, only 1 LLM call
        #   1st = aggregator_llm (astream) -- emit tokens immediately
        aggregator_llm_call_count = 0
        kg_sufficient_live = False   # updated live from kg_check_node output
        research_route_from_first_call = ""  # updated live from first_call output
        domain_route_live = []               # updated live from first_call output

        register_socket(sid, socketio)
        try:
            async for event in graph.astream_events(
                {"messages": [HumanMessage(question)], "user_id": user_id,"sid": sid},
                config=config,
                version="v2"
            ):
                kind = event["event"]
                name = event.get("name", "")

                # ── Track node journey ──
                if kind == "on_chain_start" and name in (
                    "head_agent_node", "kg_check_node", "first_call",
                    "research_node", "ontology_node", "aggregator_node","summarizer_node"
                ):
                    current_node = name
                    nodes_visited.append(name)
                    if name == "aggregator_node":
                        aggregator_llm_call_count = 0

                if kind == "on_chain_end":
                    if name == "head_agent_node":
                        output = event.get("data", {}).get("output", {})
                        head_route = output.get("head_route", "")
                        ack = output.get("acknowledgment")
                        if head_route == "pipeline" and ack and not ack_emitted:
                            socketio.emit("agent_ack", {"message": ack}, to=sid)
                            ack_emitted = True
                        current_node = None
                    elif name == "kg_check_node":
                        output = event.get("data", {}).get("output", {})
                        kg_sufficient_live = output.get("kg_sufficient", False)
                        current_node = None
                    elif name == "first_call":
                        output = event.get("data", {}).get("output", {})
                        research_route_from_first_call = output.get("research_route", "")
                        if output.get("domain_route"):
                            domain_route_live = output.get("domain_route", [])
                        current_node = None
                    elif name in ("aggregator_node", "first_call", "research_node", "ontology_node","summarizer_node"):
                        current_node = None

                # ── Count LLM calls inside aggregator_node ──
                if kind == "on_chat_model_start" and current_node == "aggregator_node":
                    aggregator_llm_call_count += 1

                # ── Stream tokens — only from aggregator_node (head_agent_node outputs structured JSON) ──
                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and chunk.content and current_node == "aggregator_node":
                        # sufficient=True: no entity extractor, stream from 1st call
                        # sufficient=False: entity extractor is 1st, stream from 2nd
                        threshold = 1 if kg_sufficient_live else 2
                        if aggregator_llm_call_count >= threshold:
                            socketio.emit("stream_token", {"token": chunk.content}, to=sid)
                            await asyncio.sleep(0.02)

            # ── Final state — only used for direct answer content and domain_route ──
            # head_route, kg_sufficient, research_route are read from live tracking vars
            # (not from checkpointer state) to avoid stale values from previous queries
            state = await graph.aget_state(config)
            values = state.values
            if head_route == "direct":
                messages = values.get("messages", [])
                last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
                if last_ai and last_ai.content:
                    socketio.emit("stream_token", {"token": last_ai.content}, to=sid)

            # Use live-tracked research_route — avoids stale checkpointer value
            research_route_live = research_route_from_first_call if head_route != "direct" else ""

            if head_route == "direct":
                prefix = "Companion"
            elif kg_sufficient_live:
                prefix = "KG Cache"
            elif research_route_live == "partial":
                prefix = "KG Cache + " + " & ".join(d.capitalize() for d in domain_route_live) + " Agent"
            elif research_route_live == "research":
                prefix = " & ".join(d.capitalize() for d in domain_route_live) + " Agent"
            else:
                prefix = "AI"

            elapsed = time.time() - start

            # ── Log full journey ──
            if log:
                journey = " -> ".join(nodes_visited)
                log.info(f"[JOURNEY] {journey}")
                log.info(f"[RESPONSE] prefix={prefix} time={elapsed:.2f}s")

            socketio.emit("stream_end", {"prefix": prefix, "time": f"{elapsed:.2f}s"}, to=sid)
            socketio.emit("kg_update", get_kg_data(sid), to=sid)
            # ── Context bar update ──
            if "summarizer_node" in nodes_visited and state.values.get("context_token_count", 0) > 0:
                context_token_count = state.values.get("context_token_count", 0)
                usage_pct = round((context_token_count / CONTEXT_TPM_LIMIT) * 100, 1)
            else:
                messages = state.values.get("messages", [])
                usage_pct = get_context_usage_pct(messages)
                context_token_count = int(usage_pct * CONTEXT_TPM_LIMIT / 100)
            socketio.emit("context_update", {
                "token_count": context_token_count,
                "usage_pct": usage_pct,
                "tpm_limit": CONTEXT_TPM_LIMIT,
            }, to=sid)

        except Exception as e:
            if log:
                log.error(f"[ERROR] {e}")
            socketio.emit("stream_token", {"token": f"\n\nAn error occurred: {e}"}, to=sid)
            socketio.emit("stream_end", {"prefix": "Error", "time": "0.00s"}, to=sid)

    def _run_worker():
        try:
            future = asyncio.run_coroutine_threadsafe(run_streaming(), loop)
            future.result()
        except Exception as e:
            if log:
                log.error(f"[WORKER ERROR] {e}")

    threading.Thread(target=_run_worker, daemon=True).start()

@socketio.on("get_entity_detail")
def handle_entity_detail(data):
    sid = request.sid
    if not require_auth(sid):
        return
    kg = chat_agent.get_kg(sid)
    if not kg:
        return
    name = data.get("name")
    entity = kg.get_entity(name)
    relationships = kg.get_relationships_for(name)
    if not entity:
        return
    socketio.emit("entity_detail", {
        "name": entity.name,
        "type": entity.type,
        "properties": entity.properties,
        "relationships": [
            {
                "source": r.source,
                "relation": r.relation,
                "target": r.target,
                "metadata": r.metadata
            }
            for r in relationships
        ]
    }, to=sid)


def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_background_loop, args=(loop,), daemon=True)
    t.start()

    future = asyncio.run_coroutine_threadsafe(build_graph(), loop)
    graph = future.result()

    print("Server running at http://localhost:5000")
    socketio.run(app, debug=False, port=5000)