// ── socket_client.js — all socket communication and state ──

const socket = io();

// App state
let currentThreadId = null;
let currentUsername = null;
let currentKgData = { nodes: [], edges: [] };
let _pendingSwitchThreadId = null;  // thread to switch to after session_init
let _pendingQuery = null;           // query to send after new chat is created

// ── Auth ──

function getToken() {
  return localStorage.getItem("goe_token") || sessionStorage.getItem("goe_token");
}

function clearToken() {
  localStorage.removeItem("goe_token");
  sessionStorage.removeItem("goe_token");
}

// ── URL routing helpers ──

function getThreadIdFromUrl() {
  const match = window.location.pathname.match(/\/app\/chat\/(.+)/);
  return match ? match[1] : null;
}

function pushChatUrl(thread_id) {
  const url = `/app/chat/${thread_id}`;
  if (window.location.pathname !== url) {
    history.pushState({ thread_id }, "", url);
  }
}

function pushLandingUrl() {
  if (window.location.pathname !== "/app") {
    history.pushState({ thread_id: null }, "", "/app");
  }
}

// ── Browser back/forward ──

window.addEventListener("popstate", (e) => {
  const thread_id = e.state ? e.state.thread_id : getThreadIdFromUrl();
  if (thread_id && thread_id !== currentThreadId) {
    socket.emit("switch_chat", { thread_id });
  } else if (!thread_id) {
    showLanding(currentUsername, currentKgData);
  }
});

// ── Auth ──

socket.on("connect", () => {
  const token = getToken();
  if (!token) {
    window.location.href = "/";
    return;
  }
  // Remember which chat was in the URL before auth
  _pendingSwitchThreadId = getThreadIdFromUrl();
  socket.emit("authenticate", { token });
});

socket.on("auth_required", () => {
  clearToken();
  window.location.href = "/";
});

socket.on("logged_out", () => {
  clearToken();
  window.location.href = "/";
});

// ── Session ──

socket.on("session_init", (d) => {
  currentThreadId = d.thread_id;
  currentUsername = d.username;
  document.getElementById("header-username").textContent = d.username || "";
  renderChatList(d.chats);

  if (_pendingSwitchThreadId) {
    // URL had a specific chat — switch to it (renderChatList already called above so _allChats is populated)
    socket.emit("switch_chat", { thread_id: _pendingSwitchThreadId });
    _pendingSwitchThreadId = null;
  } else {
    // No chat in URL — show landing
    showLanding(d.username, currentKgData);
    pushLandingUrl();
  }
});

socket.on("chat_created", (d) => {
  currentThreadId = d.thread_id;
  clearMessages();
  hideLanding();
  renderChatList(d.chats);
  pushChatUrl(d.thread_id);
  // Send pending query if one was queued via sendSuggestedQuery
  if (_pendingQuery) {
    const query = _pendingQuery;
    _pendingQuery = null;
    sendMessage(query);
  }
});

socket.on("chat_switched", (d) => {
  // Abort any in-progress streaming
  _streamingDiv = null;
  _streamingBubble = null;
  _streamBuffer = "";
  document.getElementById("send-btn").disabled = false;
  document.getElementById("status").textContent = "";

  currentThreadId = d.thread_id;
  hideLanding();
  replayMessages(d.messages);
  renderChatList_updateActive();
  pushChatUrl(d.thread_id);
});

socket.on("chat_updated", (d) => {
  renderChatList(d.chats);
});

socket.on("chat_renamed", (d) => {
  renderChatList(d.chats);
});

// ── Messages ──

socket.on("status", (d) => {
  const el = document.getElementById("status");
    if (el.style.display !== "none") el.textContent = d.text;
});

socket.on("agent_ack", (d) => {
  appendMessage("agent", d.message, "Companion");
});

// Streaming state
let _streamingDiv = null;
let _streamingBubble = null;
let _streamBuffer = "";

socket.on("stream_token", (d) => {
  document.getElementById("status").textContent = "";
  if (!_streamingDiv) {
    const c = document.getElementById("messages");
    _streamingDiv = document.createElement("div");
    _streamingDiv.className = "message agent";
    _streamingBubble = document.createElement("div");
    _streamingBubble.className = "bubble";
    _streamingDiv.appendChild(_streamingBubble);
    c.appendChild(_streamingDiv);
  }
  _streamBuffer += d.token;
  _streamingBubble.innerHTML = marked.parse(_streamBuffer);
  const c = document.getElementById("messages");
  c.scrollTop = c.scrollHeight;
});

socket.on("stream_end", (d) => {
  if (d.prefix === "System") {
    // Rate limit — start countdown, keep send disabled
    let remaining = d.rate_limit_window || 60;
    document.getElementById("send-btn").disabled = true;
    document.getElementById("status").textContent = `⚠️ Rate limit reached. Try again in ${remaining}s.`;
    const countdown = setInterval(() => {
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(countdown);
        document.getElementById("status").textContent = "";
        document.getElementById("send-btn").disabled = false;
      } else {
        document.getElementById("status").textContent = `⚠️ Rate limit reached. Try again in ${remaining}s.`;
      }
    }, 1000);
  } else {
    // Normal response
    document.getElementById("send-btn").disabled = false;
    document.getElementById("status").textContent = "";
    if (_streamingDiv) {
      const meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = `${d.prefix} · ${d.time}`;
      _streamingDiv.appendChild(meta);
      const c = document.getElementById("messages");
      c.scrollTop = c.scrollHeight;
    }
  }
  _streamingDiv = null;
  _streamingBubble = null;
  _streamBuffer = "";
});

// ── KG ──

socket.on("kg_update", (d) => {
  currentKgData = d;
  updateKGGraph(d);
  updateLandingKGStats(d);
});

socket.on("entity_detail", (d) => {
  renderEntityCard(d);
});


// ── STM Compacting ──

socket.on("compacting_start", () => {
  showCompactingBubble();
});

socket.on("compacting_progress", (d) => {
  updateCompactingProgress(d.pct);
});

socket.on("compacting_done", () => {
  hideCompactingBubble();
});

// ── Context bar ──

socket.on("context_update", (d) => {
  updateContextBar(d.usage_pct);
});


// ── Actions ──

function newChat() {
  clearMessages();
  hideLanding();
  const input = document.getElementById("input");
  if (input) input.focus();
  if (socket && socket.connected) {
    socket.emit("new_chat");
  }
}

function switchChat(thread_id) {
  if (thread_id === currentThreadId) return;
  socket.emit("switch_chat", { thread_id });
}

function renameChat(thread_id, currentTitle) {
  if (!currentTitle) {
    const chat = _allChats.find(c => c.id === thread_id);
    currentTitle = chat ? chat.title : "";
  }
  const newTitle = prompt("Rename chat:", currentTitle);
  if (!newTitle || newTitle.trim() === "" || newTitle.trim() === currentTitle) return;
  socket.emit("rename_chat", { thread_id, title: newTitle.trim() });
}

function deleteChat(thread_id) {
  if (!confirm("Delete this chat?")) return;
  socket.emit("delete_chat", { thread_id });
}

function goHome() {
  showLanding(currentUsername, currentKgData, true);
  pushLandingUrl();
}


socket.on("chat_deleted", (d) => {
  currentThreadId = d.thread_id;
  clearMessages();
  renderChatList(d.chats);
  if (d.chats.length === 0) {
    showLanding(currentUsername, currentKgData, true);
    pushLandingUrl();
  } else {
    pushChatUrl(d.thread_id);
    socket.emit("switch_chat", { thread_id: d.thread_id });
  }
});

function sendMessage(msg) {
  if (document.getElementById("send-btn").disabled) return;
  hideLanding();
  appendMessage("user", msg);
  document.getElementById("input").value = "";
  if (!socket || !socket.connected) {
    document.getElementById("status").textContent = "⚠️ Reconnecting to server...";
    setTimeout(() => {
      document.getElementById("send-btn").disabled = false;
      document.getElementById("status").textContent = "";
    }, 2000);
    return;
  }
  document.getElementById("send-btn").disabled = true;
  document.getElementById("status").textContent = "Thinking...";
  socket.emit("user_message", { message: msg });
}

function sendSuggestedQuery(query) {
  hideLanding();
  _pendingQuery = query;  // stored safely — chat_created handler will pick it up
  newChat();
}

function doLogout() {
  const token = getToken();
  clearToken();
  if (socket && socket.connected) {
    socket.emit("logout", { token });
  }
  window.location.href = "/";
}