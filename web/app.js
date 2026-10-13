// toolgate dashboard: vanilla JS, no build step. All data is inserted with textContent (never innerHTML).
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => el.querySelectorAll(s);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style") el.setAttribute("style", v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const clear = (el) => { while (el.firstChild) el.removeChild(el.firstChild); return el; };

async function api(path, body) {
  const res = await fetch(path, body ? { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) } : undefined);
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `HTTP ${res.status}`);
  return data;
}

const fmt = (v, d = 3) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "%");

let CONFIG = null;
let QUERIES = [];
let ALL_TOOLS_MAP = new Map();

// ---------- Toast Notifications ----------
let toastTimeout;
function showToast(text) {
  const toast = $("#toast");
  const toastText = $("#toast-text");
  if (!toast) return;
  toastText.textContent = text;
  toast.classList.add("show");
  clearTimeout(toastTimeout);
  toastTimeout = setTimeout(() => toast.classList.remove("show"), 2500);
}

// ---------- Theme Management ----------
function initTheme() {
  const saved = localStorage.getItem("toolgate_theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const currentTheme = saved || (prefersDark ? "dark" : "light");
  applyTheme(currentTheme);

  const toggleBtn = $("#theme-toggle");
  if (toggleBtn) {
    toggleBtn.addEventListener("click", () => {
      const isDark = document.documentElement.getAttribute("data-theme") === "dark";
      const newTheme = isDark ? "light" : "dark";
      applyTheme(newTheme);
      localStorage.setItem("toolgate_theme", newTheme);
    });
  }
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const label = $("#theme-label");
  const icon = $("#theme-icon");
  if (label) label.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
  if (icon) {
    if (theme === "dark") {
      icon.innerHTML = '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>';
    } else {
      icon.innerHTML = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>';
    }
  }
}

// ---------- Modal Inspector ----------
function openModal(toolId) {
  const modal = $("#schema-modal");
  const title = $("#modal-tool-id");
  const desc = $("#modal-tool-desc");
  const schemaPre = $("#modal-tool-schema");
  if (!modal) return;

  const tool = ALL_TOOLS_MAP.get(toolId);
  title.textContent = toolId;
  desc.textContent = tool ? tool.description : "Tool details from corpus.";
  schemaPre.textContent = tool && tool.input_schema
    ? JSON.stringify(tool.input_schema, null, 2)
    : (tool ? `Parameters: ${(tool.params || []).join(", ")}` : "Schema details not loaded.");

  modal.classList.add("open");
}

function closeModal() {
  const modal = $("#schema-modal");
  if (modal) modal.classList.remove("open");
}

const modalCloseBtn = $("#modal-close-btn");
if (modalCloseBtn) modalCloseBtn.onclick = closeModal;
const modalOverlay = $("#schema-modal");
if (modalOverlay) {
  modalOverlay.onclick = (e) => { if (e.target === modalOverlay) closeModal(); };
}

// ---------- Tabs Navigation ----------
const tabLoaded = {};
$$(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

function showTab(name) {
  $$(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  $$(".tab").forEach((s) => (s.hidden = s.id !== "tab-" + name));
  if (!tabLoaded[name]) {
    tabLoaded[name] = true;
    ({ eval: loadLatestEval, tools: loadTools, gateway: renderGateway }[name] || (() => {}))();
  }
}

// ---------- Boot & Config ----------
async function boot() {
  initTheme();

  try {
    CONFIG = await api("/api/config");
  } catch (e) {
    const b = $("#banner");
    if (b) {
      b.hidden = false;
      b.textContent = "Could not load gateway config: " + e.message;
    }
    return;
  }

  const chips = clear($("#chips"));
  [
    ["tools", CONFIG.tools, "📦"],
    ["servers", CONFIG.servers, "🔌"],
    ["embedder", CONFIG.embedder, "🧠"],
    ["reranker", CONFIG.reranker, "⚡"]
  ].forEach(([k, v, icon]) => {
    chips.append(h("span", { class: "chip" }, 
      h("span", { class: "chip-icon" }, icon),
      `${k}: `, 
      h("b", {}, v)
    ));
  });

  const notes = [];
  if (CONFIG.embedder === "hash" || CONFIG.reranker === "heuristic") {
    notes.push("Offline stand-ins are active (embedder=hash, reranker=heuristic). Set TOOLGATE_EMBEDDER / TOOLGATE_RERANKER in .env for full neural semantic models.");
  }
  if (CONFIG.corpus_source === "starter-approximation") {
    notes.push("Starter corpus active (139 tools). Run `python -m toolgate collect` to discover live tools from configured MCP servers.");
  }
  if (notes.length) {
    const b = $("#banner");
    b.hidden = false;
    b.textContent = notes.join(" ");
  }

  // Populate method checkboxes
  const box = $("#methods");
  const defaults = new Set(["bm25", "hybrid", "hybrid_rerank"]);
  CONFIG.methods.forEach((m) => {
    box.append(h("label", {},
      h("input", { type: "checkbox", value: m, checked: defaults.has(m) }),
      CONFIG.method_labels[m] || m
    ));
  });
  $("#k").value = CONFIG.default_k;

  // Load example queries
  try {
    QUERIES = (await api("/api/queries")).queries;
    const sel = $("#examples");
    QUERIES.forEach((q) => sel.append(h("option", { value: q.id }, `[${q.type}] ${q.query}`)));
  } catch { /* queries optional */ }

  // Quick query buttons
  $$(".quick-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.dataset.query;
      $("#q").value = text;
      $("#search-form").requestSubmit();
    });
  });

  // Preload tools for modal inspector
  api("/api/tools").then((res) => {
    (res.tools || []).forEach((t) => ALL_TOOLS_MAP.set(t.id, t));
  }).catch(() => {});
}

// ---------- Playground Search ----------
$("#examples").addEventListener("change", (e) => {
  const q = QUERIES.find((x) => x.id === e.target.value);
  if (q) {
    $("#q").value = q.query;
    $("#search-form").requestSubmit();
  }
});

$("#search-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const methods = [...$$("#methods input:checked")].map((i) => i.value);
  const err = $("#play-error");
  err.hidden = true;
  if (!methods.length) {
    err.hidden = false;
    err.textContent = "Please select at least one retrieval method to compare.";
    return;
  }
  const btn = $("#search-form button.primary");
  btn.disabled = true;
  try {
    const data = await api("/api/search", { query: $("#q").value, k: Number($("#k").value), methods });
    renderSearch(data);
  } catch (e) {
    err.hidden = false;
    err.textContent = e.message;
  } finally {
    btn.disabled = false;
  }
});

function renderSearch(data) {
  const gold = data.known_query ? new Set(data.known_query.gold) : null;
  const meta = $("#play-meta");
  meta.textContent = gold
    ? `🎯 Known evaluation query (${data.known_query.type}). Target tools: ${data.known_query.gold.join(", ")}${data.known_query.gold_mode === "any" ? " (any one is sufficient)" : ""}`
    : "⚡ Ad-hoc query (unlabeled live search comparison across retrievers).";

  const out = clear($("#play-results"));
  for (const r of data.results) {
    const isAll = r.method === "all_tools";
    const hits = isAll ? r.hits.slice(0, 15) : r.hits;

    const card = h("div", { class: "card" },
      h("div", { class: "card-header" },
        h("h3", {}, r.label)
      ),
      h("div", { class: "stats" },
        h("span", { class: "stat-pill" }, `⚡ ${r.latency_ms.toFixed(1)} ms`),
        h("span", { class: "stat-pill" }, `📄 ${r.tokens_returned.toLocaleString()} tokens`),
        h("span", { class: "stat-pill savings" }, `🔥 ${r.tokens_saved_pct}% saved`)
      )
    );

    hits.forEach((x) => {
      const isGoldHit = gold && gold.has(x.id);
      const serverName = x.server || (x.id.includes("__") ? x.id.split("__")[0] : "tool");

      const hitEl = h("div", { class: "hit" + (isGoldHit ? " gold" : ""), onclick: () => openModal(x.id), style: "cursor: pointer;", title: "Click to inspect schema" },
        h("span", { class: "rank" }, x.rank),
        h("div", {},
          h("div", { class: "id" }, 
            h("span", { class: "server-tag" }, serverName),
            x.name || x.id,
            isGoldHit ? h("span", { class: "gold-badge" }, "✓ Gold Target") : false
          ),
          h("div", { class: "desc" }, x.description.length > 120 ? x.description.slice(0, 120) + "…" : x.description)
        ),
        h("span", { class: "score" }, isAll ? "–" : x.score.toFixed(3))
      );
      card.append(hitEl);
    });

    if (isAll) {
      card.append(h("div", { class: "muted", style: "padding-top: 10px;" }, `…and ${r.hits.length - hits.length} more tool definitions loaded into prompt context.`));
    }

    if (gold && !isAll) {
      const got = new Set(r.hits.map((x) => x.id));
      const miss = [...gold].filter((g) => !got.has(g));
      const need = data.known_query.gold_mode === "any" ? miss.length === gold.size : miss.length > 0;
      if (need) {
        card.append(h("div", { class: "missing" }, "⚠️ Missing target tool(s): " + miss.join(", ")));
      }
    }

    out.append(card);
  }
}

// ---------- Evaluation Tab ----------
$("#eval-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const btn = $("#eval-form button.primary");
  btn.disabled = true;
  $("#eval-status").textContent = "Running benchmark evaluation...";
  $("#eval-error").hidden = true;
  try {
    const res = await api("/api/eval/run", {
      split: $("#split").value,
      k: Number($("#eval-k").value),
      ablations: $("#ablations").checked
    });
    renderEval(res);
    $("#eval-status").textContent = "✓ Completed & persisted to data/results/";
    showToast("Evaluation run finished successfully!");
  } catch (e) {
    $("#eval-error").hidden = false;
    $("#eval-error").textContent = e.message;
    $("#eval-status").textContent = "";
  } finally {
    btn.disabled = false;
  }
});

async function loadLatestEval() {
  try {
    const d = await api("/api/eval/latest");
    if (d.empty) {
      $("#eval-empty").hidden = false;
      return;
    }
    renderEval(d);
  } catch (e) {
    $("#eval-error").hidden = false;
    $("#eval-error").textContent = e.message;
  }
}

function renderEval(res) {
  $("#eval-empty").hidden = true;
  $("#eval-out").hidden = false;
  const m = res.meta, k = m.failure_k;
  $("#eval-meta").textContent = `📊 Split: ${m.split} (${m.n_queries} queries) · Tools: ${m.n_tools} · Embedder: ${m.embedder} · Reranker: ${m.reranker} · Baseline Tokens: ≈ ${m.tokens_all_tools.toLocaleString()} · ${m.timestamp}`;

  const methods = Object.entries(res.methods);
  const retrievers = methods.filter(([n]) => n !== "all_tools");

  // KPI Summary Cards
  const kpiBox = clear($("#eval-kpis"));
  const bestFh = Math.max(...retrievers.map(([, r]) => r.metrics[`full_hit@${k}`] ?? 0));
  const bestRecall = Math.max(...retrievers.map(([, r]) => r.metrics[`recall@${k}`] ?? 0));
  const maxSavings = Math.max(...retrievers.map(([, r]) => r.tokens_saved_pct ?? 0));
  const minLatency = Math.min(...retrievers.map(([, r]) => r.latency_p95_ms ?? 999));

  kpiBox.append(
    h("div", { class: "kpi-card emerald" },
      h("div", { class: "kpi-title" }, `Best Full-Hit@${k}`),
      h("div", { class: "kpi-val" }, pct(bestFh)),
      h("div", { class: "kpi-sub" }, "Queries where all required tools retrieved")
    ),
    h("div", { class: "kpi-card" },
      h("div", { class: "kpi-title" }, `Best Recall@${k}`),
      h("div", { class: "kpi-val" }, pct(bestRecall)),
      h("div", { class: "kpi-sub" }, "Overall share of target tools retrieved")
    ),
    h("div", { class: "kpi-card cyan" },
      h("div", { class: "kpi-title" }, "Context Tokens Saved"),
      h("div", { class: "kpi-val" }, `${maxSavings}%`),
      h("div", { class: "kpi-sub" }, "Prompt tokens eliminated vs all-tools baseline")
    ),
    h("div", { class: "kpi-card amber" },
      h("div", { class: "kpi-title" }, "P95 Retrieval Latency"),
      h("div", { class: "kpi-val" }, `${minLatency.toFixed(1)} ms`),
      h("div", { class: "kpi-sub" }, "Sub-millisecond query latency")
    )
  );

  // Table
  const t = clear($("#eval-table"));
  const cols = [
    ["Method", null],
    [`Recall@${k}`, `recall@${k}`],
    [`Full-hit@${k}`, `full_hit@${k}`],
    ["MRR", "mrr"],
    [`nDCG@${k}`, `ndcg@${k}`],
    ["Avg tokens", "tok"],
    ["Saved", "saved"],
    ["P50 ms", "p50"],
    ["P95 ms", "p95"]
  ];
  t.append(h("thead", {}, h("tr", {}, cols.map(([c], i) => h("th", { class: i ? "num" : "" }, c)))));

  const body = h("tbody");
  methods.forEach(([name, r]) => {
    const mt = r.metrics;
    const isBest = name !== "all_tools" && (mt[`full_hit@${k}`] ?? 0) === bestFh;
    body.append(h("tr", { class: isBest ? "best" : "" },
      h("td", { style: "font-weight: 600;" }, r.label),
      h("td", { class: "num" }, fmt(mt[`recall@${k}`])),
      h("td", { class: "num" }, isBest ? h("b", {}, fmt(mt[`full_hit@${k}`])) : fmt(mt[`full_hit@${k}`])),
      h("td", { class: "num" }, fmt(mt.mrr)),
      h("td", { class: "num" }, fmt(mt[`ndcg@${k}`])),
      h("td", { class: "num" }, Math.round(r.tokens_avg).toLocaleString()),
      h("td", { class: "num", style: "color: var(--accent-emerald);" }, r.tokens_saved_pct + "%"),
      h("td", { class: "num" }, fmt(r.latency_p50_ms, 2)),
      h("td", { class: "num" }, fmt(r.latency_p95_ms, 2))
    ));
  });
  t.append(body);

  // Visual Bars
  $("#chart1-title").textContent = `Full-hit@${k}: share of queries where every needed tool is in top-${k}`;
  bars($("#chart1"), methods.map(([n, r]) => [r.label, r.metrics[`full_hit@${k}`] ?? 0, pct(r.metrics[`full_hit@${k}`] ?? 0)]), 1, "var(--bar1)");

  const maxTok = Math.max(...methods.map(([, r]) => r.tokens_avg));
  bars($("#chart2"), methods.map(([n, r]) => [r.label, r.tokens_avg, Math.round(r.tokens_avg).toLocaleString() + " tok"]), maxTok, "var(--bar2)");

  // Failure Analysis
  const sel = clear($("#fail-method"));
  Object.keys(res.failures).forEach((n) => sel.append(h("option", { value: n }, res.methods[n].label)));

  const drawFail = () => {
    const f = res.failures[sel.value];
    const bk = clear($("#fail-buckets"));
    const tb = clear($("#fail-table"));
    if (!f) return;

    bk.append(
      h("p", { class: "muted" }, `⚠️ ${f.n_failures} of ${m.n_queries} queries missed at k=${k}. Read through error examples below to evaluate retrieval shortcomings:`),
      h("div", { class: "chips" }, Object.entries(f.buckets).map(([b, c]) => h("span", { class: "chip" }, `${b}: `, h("b", {}, c))))
    );

    tb.append(
      h("thead", {}, h("tr", {}, ["Bucket", "Type", "Query", "Gold Target (rank)", "Retrieved Instead"].map((c) => h("th", {}, c)))),
      h("tbody", {}, f.examples.map((e) => h("tr", {},
        h("td", {}, h("span", { class: "pill" }, e.bucket)),
        h("td", { style: "font-weight: 500;" }, e.type),
        h("td", { style: "font-weight: 500;" }, e.query),
        h("td", { class: "mono", style: "color: var(--accent-emerald);" }, Object.entries(e.gold_ranks).map(([g, r]) => `${g} (${r ?? "not in top-20"})`).join("\n")),
        h("td", { class: "mono" }, e.top.join("\n"))
      )))
    );
    tb.querySelectorAll("td.mono").forEach((td) => (td.style.whiteSpace = "pre-line"));
  };

  sel.onchange = drawFail;
  drawFail();

  // Ablations Section
  const box = $("#ablation-box");
  if (res.ablations) {
    box.hidden = false;
    const out = clear($("#ablations-out"));
    const a = res.ablations, fhKey = `full_hit@${a.k}`;
    const block = (title, rows, note) => {
      if (!rows.length) return;
      const ms = Object.keys(rows[0].results);
      out.append(h("h3", { style: "font-size: 14px; margin: 18px 0 8px;" }, title));
      if (note) out.append(h("p", { class: "muted" }, note));
      out.append(h("div", { class: "scroll" }, h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "Configuration"), ms.map((x) => h("th", { class: "num" }, `${x} ${fhKey}`)), ms.map((x) => h("th", { class: "num" }, `${x} MRR`)))),
        h("tbody", {}, rows.map((r) => h("tr", {}, h("td", { style: "font-weight: 600;" }, String(r.setting)),
          ms.map((x) => h("td", { class: "num" }, fmt(r.results[x][fhKey]))),
          ms.map((x) => h("td", { class: "num" }, fmt(r.results[x].mrr)))
        )))
      )));
    };
    block("Document Enrichment (name_desc = name + description; full = + server + parameters)", a.doc_mode);
    block("Reranker Depth (candidates examined by second-stage model)", a.rerank_depth);
    block("Tool Pool Size (scaling distractors)", a.pool_size, "Distractor pool expansion test (seed 7).");
  } else {
    box.hidden = true;
  }
}

function bars(el, rows, max, color) {
  clear(el);
  rows.forEach(([label, v, text]) => {
    const widthPct = Math.max(0, Math.min(100, (v / (max || 1)) * 100));
    el.append(h("div", { class: "bar-row" },
      h("span", { class: "lab", title: label }, label),
      h("div", { class: "track" }, h("div", { class: "fill", style: `width: ${widthPct}%; background: ${color}` })),
      h("span", { class: "val" }, text)
    ));
  });
}

// ---------- Tools Explorer Tab ----------
let activeServer = "";
async function loadTools() {
  const f = $("#tool-filter");
  const draw = async () => {
    const qs = new URLSearchParams();
    if (activeServer) qs.set("server", activeServer);
    if (f.value) qs.set("q", f.value);

    const d = await api("/api/tools?" + qs);
    $("#tools-count").textContent = `Displaying ${d.tools.length} of ${d.total} registered tools in catalog:`;

    // Cache tools for modal inspection
    (d.tools || []).forEach((t) => ALL_TOOLS_MAP.set(t.id, t));

    const chips = clear($("#server-chips"));
    chips.append(h("button", {
      class: "chip",
      type: "button",
      "aria-pressed": String(!activeServer),
      onclick: () => { activeServer = ""; draw(); }
    }, `All (${d.total})`));

    Object.entries(d.servers).sort().forEach(([s, n]) => {
      chips.append(h("button", {
        class: "chip",
        type: "button",
        "aria-pressed": String(activeServer === s),
        onclick: () => { activeServer = s; draw(); }
      }, `${s} (${n})`));
    });

    const t = clear($("#tools-table"));
    t.append(
      h("thead", {}, h("tr", {}, ["Tool Identifier", "Description", "Parameters", "Tokens"].map((c, i) => h("th", { class: i === 3 ? "num" : "" }, c)))),
      h("tbody", {}, d.tools.map((x) => h("tr", { onclick: () => openModal(x.id), style: "cursor: pointer;", title: "Click to view schema" },
        h("td", { class: "mono", style: "font-weight: 600; color: var(--primary);" }, x.id),
        h("td", {}, x.description.length > 150 ? x.description.slice(0, 150) + "…" : x.description),
        h("td", { class: "mono" }, (x.params || []).join(", ") || "—"),
        h("td", { class: "num", style: "font-weight: 600;" }, x.tokens)
      )))
    );
  };

  let timer;
  f.addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(draw, 180);
  });
  await draw();
}

// ---------- Gateway Configuration Tab ----------
function renderGateway() {
  if (!CONFIG) {
    tabLoaded.gateway = false;
    return;
  }
  const cfg = {
    mcpServers: {
      toolgate: {
        command: CONFIG.python,
        args: ["-m", "toolgate.gateway"],
        cwd: CONFIG.root,
        env: { PYTHONPATH: CONFIG.root }
      }
    }
  };
  const text = JSON.stringify(cfg, null, 2);
  $("#mcp-config").textContent = text;
  $("#copy-config").onclick = async (e) => {
    try {
      await navigator.clipboard.writeText(text);
      showToast("Config JSON copied to clipboard!");
    } catch {
      showToast("Please copy manually from text box.");
    }
  };
}

// Start application
boot();
