// ── ui.js — all rendering and UI logic ──

let network = null;
const nodes = new vis.DataSet([]);
const edges = new vis.DataSet([]);
let _allChats = [];

// ── KG init ──

function initKG() {
  const container = document.getElementById("kg-container");
  const options = {
    nodes: {
      shape: "dot", size: 16,
      color: { background: "#7c83fd", border: "#5a61f0", highlight: { background: "#a0a6ff" } },
      font: { color: "#ffffff", size: 13, strokeWidth: 3, strokeColor: "#0f1117" },
    },
    edges: {
      color: { color: "#555870", highlight: "#7c83fd" },
      font: { color: "#c0c0c0", size: 10, align: "middle", strokeWidth: 2, strokeColor: "#0f1117" },
      arrows: { to: { enabled: true, scaleFactor: 0.5 } },
      smooth: { type: "curvedCW", roundness: 0.2 }
    },
    physics: { stabilization: false, barnesHut: { gravitationalConstant: -3000 } },
    interaction: { hover: true, zoomView: true }
  };
  network = new vis.Network(container, { nodes, edges }, options);
  network.on("click", (params) => {
    if (params.nodes.length > 0 && typeof socket !== "undefined" && socket.emit) {
      socket.emit("get_entity_detail", { name: params.nodes[0] });
    }
  });
}

function updateKGGraph(d) {
  if (!d || !Array.isArray(d.nodes) || !Array.isArray(d.edges)) return;
  const newNodeIds = new Set(d.nodes.map(n => n.id));
  nodes.getIds().forEach(id => { if (!newNodeIds.has(id)) nodes.remove(id); });
  d.nodes.forEach(n => { if (nodes.get(n.id)) nodes.update(n); else nodes.add(n); });
  edges.clear();
  d.edges.forEach((e, i) => edges.add({ ...e, id: i }));
  document.getElementById("kg-stats").textContent =
    `${d.nodes.length} entities · ${d.edges.length} relationships`;
}

// ── Landing ──

function showLanding(username, kgData, resetThread = false) {
  document.getElementById("landing").style.display = "flex";
  document.getElementById("messages").style.display = "none";
  document.getElementById("status").style.display = "none";       // ← added
  const ctxBar = document.getElementById("context-bar-wrap");
  if (ctxBar) ctxBar.style.display = "none";
  document.getElementById("input-area").style.display = "none";  // ← added
  if (resetThread) {
    currentThreadId = null;
    renderChatList_updateActive();
  }


  // Welcome
  document.getElementById("landing-username").textContent = username || "there";

  // KG stats
  renderLandingKGStats(kgData);

  // Suggested queries — fresh login = starter, else based on KG
  renderSuggestedQueries(kgData);
}

function hideLanding() {
  document.getElementById("landing").style.display = "none";
  document.getElementById("messages").style.display = "flex";
  document.getElementById("status").style.display = "block";      // ← added
  const ctxBar = document.getElementById("context-bar-wrap");
  if (ctxBar) ctxBar.style.display = "flex";
  document.getElementById("input-area").style.display = "flex";  // ← added

}

function updateLandingKGStats(kgData) {
  if (document.getElementById("landing").style.display !== "none") {
    renderLandingKGStats(kgData);
    renderSuggestedQueries(kgData);
  }
}

function renderLandingKGStats(kgData) {
  if (!kgData || !Array.isArray(kgData.nodes) || !Array.isArray(kgData.edges)) return;
  const entityCount = kgData.nodes.length;
  const relCount = kgData.edges.length;
  const maxEntities = 200;
  const healthPct = Math.round((entityCount / maxEntities) * 100);

  // Domain breakdown from entity properties
  const domainCounts = {};
  kgData.nodes.forEach(n => {
    const domain = (n.properties && n.properties.domain) ? n.properties.domain : "general";
    domainCounts[domain] = (domainCounts[domain] || 0) + 1;
  });

  // Last updated
  const now = new Date();
  const lastUpdated = entityCount > 0
    ? `${now.toLocaleDateString()} ${now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}`
    : "Never";

  // Domain breakdown HTML
  const domainHTML = Object.entries(domainCounts).length > 0
    ? Object.entries(domainCounts).map(([domain, count]) =>
        `<div class="landing-domain-item">
          <span class="landing-domain-name">${domain}</span>
          <span class="landing-domain-count">${count}</span>
        </div>`
      ).join("")
    : `<div class="landing-domain-empty">No data yet</div>`;

  document.getElementById("landing-stats").innerHTML = `
    <div class="landing-stat-card">
      <div class="landing-stat-value">${entityCount}</div>
      <div class="landing-stat-label">Entities</div>
    </div>
    <div class="landing-stat-card">
      <div class="landing-stat-value">${relCount}</div>
      <div class="landing-stat-label">Relationships</div>
    </div>
    <div class="landing-stat-card">
      <div class="landing-stat-value">${healthPct}%</div>
      <div class="landing-stat-label">KG Capacity</div>
      <div class="landing-health-bar">
        <div class="landing-health-fill" style="width:${healthPct}%; background:${healthPct > 80 ? '#ff6b6b' : healthPct > 50 ? '#ffd700' : '#7c83fd'}"></div>
      </div>
    </div>
    <div class="landing-stat-card landing-domain-card">
      <div class="landing-stat-label" style="margin-bottom:8px">Domain Breakdown</div>
      ${domainHTML}
    </div>
    <div class="landing-stat-card">
      <div class="landing-stat-label">Last Updated</div>
      <div class="landing-stat-value" style="font-size:13px; margin-top:4px">${lastUpdated}</div>
    </div>
  `;
}

function renderSuggestedQueries(kgData) {
  const nodes = (kgData && Array.isArray(kgData.nodes)) ? kgData.nodes : [];
  const isFresh = nodes.length === 0;
  const queries = isFresh
    ? [
        "What can you do?",
        "What is the current weather in Delhi?",
        "What is the latest news on AI?",
        "Tell me about climate change in India",
      ]
    : [
        "What's new in tech today?",
        "Give me a climate summary for this week",
        "What do you know so far?",
        "Search for the latest developments in AI",
      ];

  document.getElementById("landing-queries").innerHTML = queries.map(q =>
    `<button class="landing-query-btn" onclick="sendSuggestedQuery('${q.replace(/'/g, "\\'")}')">${q}</button>`
  ).join("");
}

// ── Chat list ──

function formatTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function renderChatList(chats) {
  _allChats = chats;
  renderChatList_updateActive();
}

function renderChatList_updateActive() {
  const list = document.getElementById("chat-list");
  list.innerHTML = "";
  [..._allChats].reverse().forEach(chat => {
    const item = document.createElement("div");
    item.className = "chat-item" + (chat.id === currentThreadId ? " active" : "");
    item.innerHTML = `
      <div class="chat-item-row">
        <div class="chat-item-title">${chat.title}</div>
        <div class="chat-item-actions">
          <button class="chat-action-btn" title="Rename" onclick="event.stopPropagation(); renameChat('${chat.id}')">✎</button>
          <button class="chat-action-btn delete" title="Delete" onclick="event.stopPropagation(); deleteChat('${chat.id}')">✕</button>
        </div>
      </div>
      <div class="chat-item-time">${formatTime(chat.timestamp)}</div>
    `;
    item.onclick = () => switchChat(chat.id);
    list.appendChild(item);
  });
}

// ── Messages ──

function clearMessages() {
  document.getElementById("messages").innerHTML = "";
  document.getElementById("status").textContent = "";
  updateContextBar(0);
}

function replayMessages(messages) {
  clearMessages();
  messages.forEach(m => appendMessage(m.role, m.content));
}

function appendMessage(role, content, meta = "") {
  const c = document.getElementById("messages");
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if (role === "agent") {
    bubble.innerHTML = marked.parse(content);
  } else {
    bubble.textContent = content;
  }
  div.appendChild(bubble);
  if (meta) {
    const m = document.createElement("div");
    m.className = "meta";
    m.textContent = meta;
    div.appendChild(m);
  }
  c.appendChild(div);
  c.scrollTop = c.scrollHeight;
}


// ── Compacting bubble ──

let _compactingDiv = null;

function showCompactingBubble() {
  const c = document.getElementById("messages");
  _compactingDiv = document.createElement("div");
  _compactingDiv.className = "message agent";
  _compactingDiv.id = "compacting-bubble";
  _compactingDiv.innerHTML = `
    <div class="bubble compacting-bubble">
      <div class="compacting-label">🗜️ Compacting conversation...</div>
      <div class="compacting-bar-track">
        <div class="compacting-bar-fill" id="compacting-fill" style="width:0%"></div>
      </div>
      <div class="compacting-pct" id="compacting-pct">0%</div>
    </div>
  `;
  c.appendChild(_compactingDiv);
  c.scrollTop = c.scrollHeight;
}

function updateCompactingProgress(pct) {
  const fill = document.getElementById("compacting-fill");
  const label = document.getElementById("compacting-pct");
  if (fill) fill.style.width = `${pct}%`;
  if (label) label.textContent = `${pct}%`;
  const c = document.getElementById("messages");
  if (c) c.scrollTop = c.scrollHeight;
}

function hideCompactingBubble() {
  if (_compactingDiv) {
    _compactingDiv.remove();
    _compactingDiv = null;
  }
}

// ── Context bar ──

function updateContextBar(pct) {
  const bar = document.getElementById("context-fill");
  const label = document.getElementById("context-pct");
  if (!bar || !label) return;
  bar.style.width = `${Math.min(pct, 100)}%`;
  bar.style.background = pct >= 80 ? "#ff6b6b" : pct >= 50 ? "#ffd700" : "#4caf50";
  label.textContent = `Context: ${pct}%`;
}



// ── Entity card ──

function renderEntityCard(d) {
  document.getElementById("card-title").textContent = d.name;
  document.getElementById("card-type").textContent = d.type;
  const props = Object.entries(d.properties || {});
  document.getElementById("card-props").innerHTML = props.length
    ? props.map(([k, v]) => `<span style="color:#666">${k}:</span> <span>${v}</span>`).join(" · ")
    : "";
  document.getElementById("card-rels").innerHTML = d.relationships.length
    ? d.relationships.map(r =>
        `<div class="rel-item">
          <span class="rel-source">${r.source}</span>
          <span class="rel-relation"> — ${r.relation} → </span>
          <span class="rel-target">${r.target}</span>
        </div>`).join("")
    : `<div style="color:#555; font-size:12px;">No relationships found</div>`;
  document.getElementById("entity-card").classList.add("visible");
}

function closeCard() {
  document.getElementById("entity-card").classList.remove("visible");
}

// ── KG panel controls ──

function toggleFullscreen() {
  const panel = document.querySelector(".kg-panel");
  const btn = document.getElementById("fs-btn");
  panel.classList.toggle("fullscreen");
  btn.textContent = panel.classList.contains("fullscreen") ? "⛶ Exit" : "⛶ Full";
  if (network) network.fit();
}

function toggleKG() {
  const panel = document.querySelector(".kg-panel");
  const btn = document.getElementById("kg-toggle-btn");
  const isHidden = panel.style.display === "none";
  panel.style.display = isHidden ? "flex" : "none";
  btn.textContent = isHidden ? "🔵 Knowledge Graph" : "⚪ Knowledge Graph";
  if (isHidden && network) network.fit();
}

function closeKG() {
  document.querySelector(".kg-panel").style.display = "none";
  document.getElementById("kg-toggle-btn").textContent = "⚪ Knowledge Graph";
}

// ── Input ──

function handleSendClick() {
  const input = document.getElementById("input");
  const msg = input.value.trim();
  if (!msg) return;
  sendMessage(msg);
}

// ── Entity card drag ──

function initDrag() {
  const card = document.getElementById("entity-card");
  const cardHeader = document.getElementById("entity-card-header");
  let isDragging = false, dragOffsetX = 0, dragOffsetY = 0;
  cardHeader.addEventListener("mousedown", (e) => {
    isDragging = true;
    dragOffsetX = e.clientX - card.getBoundingClientRect().left;
    dragOffsetY = e.clientY - card.getBoundingClientRect().top;
    e.preventDefault();
  });
  document.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    card.style.left = (e.clientX - dragOffsetX) + "px";
    card.style.top = (e.clientY - dragOffsetY) + "px";
    card.style.right = "auto";
  });
  document.addEventListener("mouseup", () => { isDragging = false; });
}

// ── Init ──

document.addEventListener("DOMContentLoaded", () => {
  initKG();
  initDrag();

  // Render landing with empty data immediately so it looks good before session_init
  renderLandingKGStats({ nodes: [], edges: [] });
  renderSuggestedQueries({ nodes: [] });

  document.getElementById("input").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (document.getElementById("send-btn").disabled) return;
      handleSendClick();
    }
  });
});