const messagesEl = document.querySelector("#messages");
const stepsEl = document.querySelector("#steps");
const formEl = document.querySelector("#chat-form");
const inputEl = document.querySelector("#message-input");
const sendButtonEl = document.querySelector("#send-button");
const newSessionEl = document.querySelector("#new-session");
const sessionBadgeEl = document.querySelector("#session-badge");
const coverageToggleEl = document.querySelector("#coverage-toggle");
const retrievalModeSelectEl = document.querySelector("#retrieval-mode-select");
retrievalModeSelectEl.value = "dense";
// Deliberate opt-in per page/session; never restore a paid option from storage.
coverageToggleEl.checked = false;

let sessionId = "";
let isRunning = false;
let streamingAssistant = null;
let lastAnswerType = null;
const statusItems = new Map();
const retrievalMapEl = document.querySelector("#retrieval-map");
let retrievalTrace;
let coverageCard = null;

function updateCoverageCard(payload, label) {
  if (!coverageCard || !coverageCard.isConnected) {
    coverageCard = document.createElement("section");
    coverageCard.className = "step coverage-card";
    coverageCard.setAttribute("aria-label", "Evidence coverage check");
    coverageCard.innerHTML = `
      <h3><span>Evidence coverage</span><span class="coverage-badge" aria-live="polite"></span></h3>
      <p class="coverage-purpose">One diagnosis · at most one supplementary search</p>
      <ol class="coverage-stages">
        <li data-coverage-stage="gate"><span>1 · Intent gate</span><span class="coverage-stage-value"></span></li>
        <li data-coverage-stage="diagnosis"><span>2 · Gap diagnosis</span><span class="coverage-stage-value"></span></li>
        <li data-coverage-stage="search"><span>3 · Limited search</span><span class="coverage-stage-value"></span></li>
        <li data-coverage-stage="rerank"><span>4 · Original-query rerank</span><span class="coverage-stage-value"></span></li>
      </ol>
      <div class="coverage-detail coverage-aspect" hidden><span>Missing aspect</span><p></p></div>
      <div class="coverage-detail coverage-query" hidden><span>Supplementary query</span><p></p></div>
      <div class="coverage-comparison" hidden>
        <span>Same-request scores · original question</span>
        <div class="coverage-score-new"><span>New</span><strong></strong><p></p></div>
        <div class="coverage-score-old"><span>Old</span><strong></strong><p></p></div>
      </div>
      <div class="coverage-outcome"></div>
      <p class="coverage-footnote"></p>`;
    stepsEl.appendChild(coverageCard);
  }
  const card = coverageCard;
  const status = payload.status;
  const failed = ["check_failed", "search_failed", "rerank_failed"].includes(status);
  const searching = ["searching", "reranking", "searched_no_replacement", "searched_replaced", "search_failed", "rerank_failed"].includes(status);
  const stage = (key, text, state) => {
    const row = card.querySelector(`[data-coverage-stage="${key}"]`);
    row.dataset.state = state;
    row.querySelector(".coverage-stage-value").textContent = text;
  };
  card.dataset.status = status;
  const badges = { checking: "Checking", searching: "Searching", reranking: "Comparing", gate_skipped: "Skipped",
    sufficient: "Kept original", check_failed: "Check failed", search_failed: "Search failed", rerank_failed: "Rerank failed",
    gap_found: "No usable query", searched_no_replacement: "Kept original", searched_replaced: "Replaced 1" };
  card.querySelector(".coverage-badge").textContent = badges[status] || status;
  if (status === "gate_skipped") {
    const reason = { disabled: "Switch off", empty_evidence: "No initial evidence", intent: "Not a legal consultation" };
    stage("gate", reason[payload.reason] || "Skipped by gate", "skipped");
    ["diagnosis", "search", "rerank"].forEach((key) => stage(key, "Not run", "skipped"));
  } else {
    stage("gate", "Legal consultation", "done");
    stage("diagnosis", status === "checking" ? "Running…" : status === "check_failed" ? "Failed" : status === "sufficient" ? "Diagnosed sufficient" : "Gap identified",
      status === "checking" ? "running" : status === "check_failed" ? "warning" : "done");
    const source = { statute: "Statutes", case: "Cases", both: "Both libraries" }[payload.target_source] || "";
    stage("search", status === "search_failed" ? "Failed" : searching ? payload.child_count != null ? `${payload.child_count} child hits · ${payload.new_parent_count ?? "—"} new parents` : `${source} · running…` : status === "sufficient" ? "Not needed" : "Not run",
      status === "searching" ? "running" : status === "search_failed" ? "warning" : searching ? "done" : "skipped");
    stage("rerank", status === "reranking" ? `${payload.comparison_count ?? "≤10"} parents · running…` : status === "rerank_failed" ? "Failed" : payload.comparison ? `${payload.comparison_count} parents · one request` : "Not run",
      status === "reranking" ? "running" : status === "rerank_failed" ? "warning" : payload.comparison ? "done" : "skipped");
  }
  const detail = (selector, value) => {
    const node = card.querySelector(selector);
    node.hidden = !value;
    node.querySelector("p").textContent = value || "";
  };
  detail(".coverage-aspect", payload.missing_aspect);
  detail(".coverage-query", payload.supplementary_query);
  const comparison = card.querySelector(".coverage-comparison");
  comparison.hidden = !payload.comparison;
  if (payload.comparison) {
    ["new", "old"].forEach((key) => {
      const row = card.querySelector(`.coverage-score-${key}`);
      const score = payload.comparison[`${key}_score`];
      row.querySelector("strong").textContent = typeof score === "number" ? score.toFixed(4) : "—";
      row.querySelector("p").textContent = payload.comparison[`${key}_title`] || payload.comparison[`${key}_parent_id`] || "";
    });
  }
  const outcome = card.querySelector(".coverage-outcome");
  outcome.textContent = payload.replaced ? "One item replaced · gap not rechecked"
    : status === "sufficient" ? "Initial evidence retained"
    : status === "gate_skipped" ? "0 diagnosis calls · initial evidence retained"
    : failed ? "Operation failed · initial evidence retained"
    : status === "searched_no_replacement" ? "Searched · no evidence item replaced"
    : status === "gap_found" ? "Gap identified · no search performed"
    : "Initial evidence stays in place while checking.";
  card.querySelector(".coverage-footnote").textContent = payload.replaced
    ? "Replacement is not proof that the missing aspect was resolved."
    : status === "sufficient" ? "A model diagnosis, not an independently verified coverage guarantee."
    : searching ? "At most 20 child hits, 5 new parents and 1 replacement."
    : "Checks run only for eligible legal consultations.";
  stepsEl.scrollTop = stepsEl.scrollHeight;
}

function resetRetrievalMap() {
  retrievalTrace = { mode: null, targets: ["statute_db", "case_db"], libraries: {}, stages: {}, fallback: false, contextWarning: false };
  retrievalMapEl.querySelectorAll(".route-node").forEach((node) => {
    node.classList.remove("is-running", "is-done", "is-disabled", "is-warning");
    node.querySelector(".route-value").textContent = node.dataset.routeStage === "fusion" ? "Up to 30 / library" : "Waiting";
  });
  const mode = retrievalModeSelectEl.value;
  const names = { dense: "Dense", dense_bm25: "Dense + BM25", hybrid: "Hybrid: Dense + BM25 + Exact" };
  document.querySelector("#retrieval-mode").textContent = `${names[mode]} · selected for next question`;
  ["bm25", "exact", "fusion"].forEach((key) => {
    const disabled = mode === "dense" || (key === "exact" && mode !== "hybrid");
    const node = retrievalMapEl.querySelector(`[data-route-stage="${key}"]`);
    node.classList.toggle("is-disabled", disabled);
    if (disabled) node.querySelector(".route-value").textContent = "Not enabled";
  });
  document.querySelector("#retrieval-state").textContent = "Not started";
  document.querySelector("#retrieval-state").classList.remove("is-warning");
  document.querySelector("#retrieval-note").textContent = "Runs after the question has enough details. Counts appear as each stage finishes.";
}

function updateRetrievalMap(payload) {
  const { stage, status, target } = payload;
  const badge = document.querySelector("#retrieval-state");
  const note = document.querySelector("#retrieval-note");
  if (stage === "plan") {
    resetRetrievalMap();
    retrievalTrace.mode = payload.mode;
    retrievalTrace.targets = payload.targets?.length ? payload.targets : ["statute_db"];
    const names = { dense: "Dense", dense_bm25: "Dense + BM25", hybrid: "Hybrid: Dense + BM25 + Exact" };
    document.querySelector("#retrieval-mode").textContent = `${names[payload.mode] || payload.mode} · ${retrievalTrace.targets.length} ${retrievalTrace.targets.length === 1 ? "library" : "libraries"}`;
    ["bm25", "exact", "fusion"].forEach((key) => {
      const disabled = payload.mode === "dense" || (key === "exact" && payload.mode !== "hybrid");
      const node = retrievalMapEl.querySelector(`[data-route-stage="${key}"]`);
      node.classList.toggle("is-disabled", disabled);
      if (disabled) node.querySelector(".route-value").textContent = "Not enabled";
    });
    retrievalMapEl.querySelector('[data-route-stage="rerank"] .route-value').textContent = `Target ${payload.rerank_k}`;
    retrievalMapEl.querySelector('[data-route-stage="evidence"] .route-value').textContent = `Target ${payload.final_k}`;
    badge.textContent = "Retrieving";
    note.textContent = "Statutes and cases are searched separately, then scored together.";
    return;
  }
  if (target) {
    const library = retrievalTrace.libraries[target] ||= {};
    if (stage === "library" && status === "skipped") {
      ["dense", "bm25", "exact", "fusion"].forEach((key) => { library[key] = { status: "skipped" }; });
    } else {
      library[stage] = { ...payload };
    }
    ["dense", "bm25", "exact", "fusion"].forEach((key) => {
      const node = retrievalMapEl.querySelector(`[data-route-stage="${key}"]`);
      if (node.classList.contains("is-disabled")) return;
      const rows = retrievalTrace.targets.map((name) => ({ name, value: retrievalTrace.libraries[name]?.[key] }));
      const running = rows.some(({ value }) => value?.status === "started");
      const done = rows.every(({ value }) => ["completed", "skipped"].includes(value?.status));
      node.classList.toggle("is-running", running);
      node.classList.toggle("is-done", done);
      node.querySelector(".route-value").textContent = rows.map(({ name, value }) => {
        const label = name === "case_db" ? "Cases" : "Law";
        if (value?.status === "skipped") return `${label} skipped`;
        if (value?.status !== "completed") return `${label} ${value?.status === "started" ? "…" : "—"}`;
        if (key === "exact") return `${label} ${value.high_count} high / ${value.weak_count} weak`;
        return `${label} ${value.count}`;
      }).join(" · ");
    });
    if (stage === "fusion" && status === "completed" && retrievalTrace.mode === "hybrid") {
      const reserved = Object.values(retrievalTrace.libraries).reduce((sum, value) => sum + (value.fusion?.reserved_count || 0), 0);
      note.textContent = `${reserved} exact seats used so far (at most 5 per library). Other seats use RRF.`;
    }
    return;
  }
  const node = retrievalMapEl.querySelector(`[data-route-stage="${stage}"]`);
  if (!node) return;
  retrievalTrace.stages[stage] = { ...payload };
  node.classList.toggle("is-running", status === "started");
  node.classList.toggle("is-done", status === "completed");
  node.classList.toggle("is-warning", status === "fallback");
  if (status === "started") node.querySelector(".route-value").textContent = "Running…";
  else if (stage === "parents") node.querySelector(".route-value").textContent = `${payload.count}/${payload.expected_count} recovered`;
  else node.querySelector(".route-value").textContent = `${payload.count} · ${payload.statute_count} law + ${payload.case_count} cases`;
  if (status === "fallback") {
    retrievalTrace.fallback = true;
    badge.classList.add("is-warning");
    note.textContent = "Rerank unavailable: using the retrieval fallback order.";
  }
  if (stage === "parents" && status === "completed" && payload.count < payload.expected_count) {
    retrievalTrace.contextWarning = true;
    node.classList.remove("is-done");
    node.classList.add("is-warning");
  }
  if (stage === "evidence" && status === "completed") {
    badge.textContent = !payload.count ? "No evidence" : retrievalTrace.fallback ? "Used fallback" : retrievalTrace.contextWarning ? "Partial context" : "Complete";
    badge.classList.toggle("is-warning", !payload.count || retrievalTrace.fallback || retrievalTrace.contextWarning);
    const exactRows = Object.values(retrievalTrace.libraries).map((value) => value.exact).filter(Boolean);
    const noExact = retrievalTrace.mode === "hybrid" && exactRows.every((value) => !value.high_count && !value.weak_count);
    if (!payload.count) note.textContent = "No usable evidence was found.";
    else if (retrievalTrace.contextWarning) note.textContent = "Some parent contexts are missing; child text may be used.";
    else if (!retrievalTrace.fallback) note.textContent = retrievalTrace.mode === "dense"
      ? "Dense candidates were reranked and selected by source quotas."
      : noExact ? "No exact matches for this question. Dense and BM25 supplied the candidates."
      : "Candidate counts overlap across channels; RRF removes duplicates within each library.";
  }
}

function interruptRetrievalMap() {
  // A later answer-generation error must not erase a successfully completed retrieval.
  if (retrievalTrace.stages.evidence?.status === "completed") return;
  const badge = document.querySelector("#retrieval-state");
  badge.textContent = retrievalTrace.mode ? "Interrupted" : "Not reached";
  badge.classList.add("is-warning");
  document.querySelector("#retrieval-note").textContent = "Retrieval did not complete. See the error in the conversation.";
  retrievalMapEl.querySelectorAll(".is-running").forEach((node) => {
    node.classList.remove("is-running");
    node.classList.add("is-warning");
  });
}

const emptyText =
  "Ask a question, or keep adding details after a follow-up.\nThe right panel shows the Agent's key decisions and progress.";

const fieldLabels = {
  intents: "Intents",
  entities: "Entities",
  user_goal: "User goal",
  rewritten_query: "Rewritten query",
  query_type: "Query type",
  topic_relation: "Topic relation",
  missing_info: "Missing info",
  need_followup: "Need follow-up",
  next_action: "Next action",
  followup_questions: "Follow-up questions",
  coverage_status: "Coverage check",
  missing_aspect: "Missing aspect",
  supplementary_query: "Supplementary search",
  triggered_search: "Supplementary search attempted",
  replaced: "Replaced one evidence item",
};

function setSession(id) {
  sessionId = id || "";
  sessionBadgeEl.textContent = sessionId || "Not started";
}

function ensureEmptyState() {
  if (messagesEl.children.length > 0) return;
  const div = document.createElement("div");
  div.className = "empty-state";
  div.textContent = emptyText;
  messagesEl.appendChild(div);
}

function removeEmptyState() {
  const empty = messagesEl.querySelector(".empty-state");
  if (empty) empty.remove();
}

function scrollMessages() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendMeta(article, metaItems) {
  if (!metaItems?.length) return;
  let meta = article.querySelector(".message-meta");
  if (!meta) {
    meta = document.createElement("div");
    meta.className = "message-meta";
    article.prepend(meta);
  }
  meta.innerHTML = "";
  metaItems.forEach((item) => {
    const span = document.createElement("span");
    span.className = "pill";
    span.textContent = item;
    meta.appendChild(span);
  });
}

function appendCitations(article, citations) {
  if (!citations?.length) return;
  const existing = article.querySelector(".citations");
  if (existing) existing.remove();

  const refs = document.createElement("div");
  refs.className = "citations";
  refs.textContent = "References";
  const list = document.createElement("ol");
  citations.forEach((citation) => {
    const li = document.createElement("li");
    li.textContent = [citation.source_name, citation.title_path].filter(Boolean).join(" / ");
    list.appendChild(li);
  });
  refs.appendChild(list);
  article.appendChild(refs);
}

function appendMessage(role, content, options = {}) {
  removeEmptyState();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  appendMeta(article, options.meta);

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = content;
  article.appendChild(body);

  appendCitations(article, options.citations);
  messagesEl.appendChild(article);
  scrollMessages();
  return article;
}

function ensureAssistantStream() {
  if (streamingAssistant) return streamingAssistant;
  streamingAssistant = appendMessage("assistant", "");
  streamingAssistant.classList.add("streaming");
  return streamingAssistant;
}

function appendAssistantDelta(delta) {
  const article = ensureAssistantStream();
  const body = article.querySelector(".message-body");
  body.textContent += delta;
  scrollMessages();
}

function finishAssistantStream(result) {
  const article = streamingAssistant || appendMessage("assistant", result.answer || "");
  article.classList.remove("streaming");
  appendMeta(article, [`Type: ${result.answer_type}`]);
  appendCitations(article, result.citations || []);
  streamingAssistant = null;
}

function foldAssistantStream(payload) {
  if (!streamingAssistant) return;

  const article = streamingAssistant;
  const body = article.querySelector(".message-body");
  const content = body?.textContent || "";
  article.classList.remove("streaming");
  article.classList.add("rejected-answer");

  const details = document.createElement("details");
  details.className = "rejected-details";

  const summary = document.createElement("summary");
  summary.textContent =
    payload.stage === "rewrite"
      ? "Revised answer still failed safety review; collapsed"
      : "Previous answer failed safety review; collapsed";
  details.appendChild(summary);

  const note = document.createElement("div");
  note.className = "rejected-note";
  note.textContent = payload.issues?.length
    ? `Reason: ${payload.issues.join("; ")}`
    : "Reason: failed safety review";
  details.appendChild(note);

  const folded = document.createElement("div");
  folded.className = "message-body rejected-body";
  folded.textContent = content;
  details.appendChild(folded);

  article.innerHTML = "";
  article.appendChild(details);
  streamingAssistant = null;
  scrollMessages();
}

function setRunning(value) {
  isRunning = value;
  sendButtonEl.disabled = value;
  inputEl.disabled = value;
  coverageToggleEl.disabled = value;
  retrievalModeSelectEl.disabled = value;
  sendButtonEl.textContent = value ? "Running" : "Send";
}

function clearSteps() {
  stepsEl.innerHTML = "";
  statusItems.clear();
  coverageCard = null;
  resetRetrievalMap();
}

function clearStatusItems() {
  statusItems.forEach((item) => item.remove());
  statusItems.clear();
}

function appendContinuationSeparator() {
  const sep = document.createElement("div");
  sep.className = "continuation-separator";
  sep.textContent = "User follow-up";
  stepsEl.appendChild(sep);
  stepsEl.scrollTop = stepsEl.scrollHeight;
}

function appendStatus(message) {
  const p = document.createElement("p");
  p.className = "status-line";
  p.textContent = message;
  stepsEl.appendChild(p);
  stepsEl.scrollTop = stepsEl.scrollHeight;
}

function isEmptyValue(value) {
  return (
    value === undefined ||
    value === null ||
    value === "" ||
    (Array.isArray(value) && value.length === 0) ||
    (typeof value === "object" && !Array.isArray(value) && Object.keys(value).length === 0)
  );
}

function appendTags(parent, values) {
  const wrap = document.createElement("div");
  wrap.className = "tag-list";
  values.forEach((value) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = String(value);
    wrap.appendChild(tag);
  });
  parent.appendChild(wrap);
}

function appendList(parent, values) {
  const ul = document.createElement("ul");
  ul.className = "step-list";
  values.forEach((value) => {
    const li = document.createElement("li");
    li.textContent = String(value);
    ul.appendChild(li);
  });
  parent.appendChild(ul);
}

function appendEntities(parent, entities) {
  const wrap = document.createElement("div");
  wrap.className = "entity-grid";
  Object.entries(entities).forEach(([key, value]) => {
    if (isEmptyValue(value)) return;
    const group = document.createElement("div");
    group.className = "entity-group";

    const title = document.createElement("span");
    title.className = "entity-key";
    title.textContent = key;
    group.appendChild(title);

    const values = Array.isArray(value) ? value : [value];
    appendTags(group, values);
    wrap.appendChild(group);
  });
  parent.appendChild(wrap);
}

function appendField(parent, key, value) {
  if (isEmptyValue(value)) return;

  const row = document.createElement("div");
  row.className = "step-field";

  const label = document.createElement("div");
  label.className = "step-field-label";
  label.textContent = fieldLabels[key] || key;
  row.appendChild(label);

  const content = document.createElement("div");
  content.className = "step-field-content";
  if (key === "entities") {
    appendEntities(content, value);
  } else if (Array.isArray(value)) {
    if (["intents", "missing_info"].includes(key)) {
      appendTags(content, value);
    } else {
      appendList(content, value);
    }
  } else if (typeof value === "boolean") {
    content.textContent = value ? "Yes" : "No";
  } else {
    content.textContent = String(value);
  }

  row.appendChild(content);
  parent.appendChild(row);
}

function appendSummaryStep(label, payload, fieldOrder) {
  const section = document.createElement("section");
  section.className = "step step-summary";

  const title = document.createElement("h3");
  title.textContent = label;
  section.appendChild(title);

  const body = document.createElement("div");
  body.className = "step-body";
  fieldOrder.forEach((key) => appendField(body, key, payload[key]));
  section.appendChild(body);

  stepsEl.appendChild(section);
  stepsEl.scrollTop = stepsEl.scrollHeight;
}

function updateStatusStep(payload) {
  const key = payload.step || payload.label;
  if (!key) return;

  let item = statusItems.get(key);
  if (!item) {
    item = document.createElement("div");
    item.className = "status-step running";

    const dot = document.createElement("span");
    dot.className = "status-dot";
    item.appendChild(dot);

    const text = document.createElement("span");
    text.className = "status-text";
    item.appendChild(text);

    const state = document.createElement("span");
    state.className = "status-state";
    item.appendChild(state);

    statusItems.set(key, item);
    stepsEl.appendChild(item);
  }

  item.querySelector(".status-text").textContent = payload.label || key;
  const done = payload.status === "completed";
  item.querySelector(".status-state").textContent = done ? "Done" : "Running";
  item.classList.toggle("done", done);
  item.classList.toggle("running", !done);
  stepsEl.scrollTop = stepsEl.scrollHeight;
}

function appendStep(step, label, payload) {
  if (step === "coverage_check") {
    updateCoverageCard(payload, label);
    return;
  }
  if (step === "retrieval_progress") {
    updateRetrievalMap(payload);
    return;
  }
  if (step === "query_understanding") {
    appendSummaryStep(label, payload, [
      "intents",
      "entities",
      "user_goal",
      "rewritten_query",
      "query_type",
      "topic_relation",
      "missing_info",
    ]);
    return;
  }
  if (step === "followup_result") {
    if (payload.need_followup === true) {
      document.querySelector("#retrieval-state").textContent = "Awaiting details";
      document.querySelector("#retrieval-note").textContent = "Retrieval has not run. Answer the follow-up questions first.";
    }
    appendSummaryStep(label, payload, ["need_followup", "next_action", "followup_questions"]);
    return;
  }
  if (step === "followup_analysis") {
    appendSummaryStep(label, payload, [
      "missing_info",
      "need_followup",
    ]);
    return;
  }
  if (step === "topic_continuation") {
    if (payload.action === "restart") {
      clearSteps();
      appendStatus("Question submitted; waiting for the Agent...");
    } else {
      appendContinuationSeparator();
    }
    return;
  }
  if (step === "status_line") {
    appendStatus(payload.text);
    return;
  }
  if (step === "step_status") {
    updateStatusStep(payload);
  }
}

function parseSseChunk(buffer, onEvent) {
  let boundary = buffer.indexOf("\n\n");
  while (boundary !== -1) {
    const raw = buffer.slice(0, boundary);
    buffer = buffer.slice(boundary + 2);
    const event = { event: "message", data: "" };
    raw.split("\n").forEach((line) => {
      if (line.startsWith("event:")) event.event = line.slice(6).trim();
      if (line.startsWith("data:")) event.data += line.slice(5).trim();
    });
    if (event.data) onEvent(event.event, JSON.parse(event.data));
    boundary = buffer.indexOf("\n\n");
  }
  return buffer;
}

async function sendMessage(message) {
  coverageCard = null;
  resetRetrievalMap();
  streamingAssistant = null;
  appendMessage("user", message);
  if (lastAnswerType === "followup") {
    clearStatusItems();
  } else if (lastAnswerType === "answer") {
    clearStatusItems();
  } else {
    clearSteps();
    appendStatus("Question submitted; waiting for the Agent...");
  }
  lastAnswerType = null;
  setRunning(true);

  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId || null, message, enable_coverage_check: coverageToggleEl.checked,
      retrieval_mode: retrievalModeSelectEl.value }),
  });

  if (!response.ok && !response.body) {
    throw new Error(`Request failed: HTTP ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";
  let finalReceived = false;

  const handleEvent = (eventName, data) => {
    if (eventName === "status") {
      appendStatus(data.message);
      return;
    }
    if (eventName === "step") {
      appendStep(data.step, data.label, data.payload);
      return;
    }
    if (eventName === "answer_delta") {
      appendAssistantDelta(data.delta || "");
      return;
    }
    if (eventName === "answer_event") {
      if (data.event === "answer_rejected") {
        foldAssistantStream(data);
      }
      if (data.event === "answer_fallback") {
        appendStatus("Revised result still failed safety review; using safe fallback answer.");
      }
      return;
    }
    if (eventName === "final") {
      finalReceived = true;
      setSession(data.session_id);
      finishAssistantStream(data);
      lastAnswerType = data.answer_type;
      return;
    }
    if (eventName === "error") {
      interruptRetrievalMap();
      appendMessage("error", data.error || "Backend execution failed.");
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    buffer = parseSseChunk(buffer, handleEvent);
  }

  if (!finalReceived && response.ok) {
    interruptRetrievalMap();
    appendMessage("error", "Connection ended without a final answer.");
  }
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (isRunning) return;

  const message = inputEl.value.trim();
  if (!message) return;

  inputEl.value = "";
  try {
    await sendMessage(message);
  } catch (error) {
    interruptRetrievalMap();
    appendMessage("error", error.message || String(error));
  } finally {
    setRunning(false);
    inputEl.focus();
  }
});

inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    formEl.requestSubmit();
  }
});

newSessionEl.addEventListener("click", () => {
  coverageToggleEl.checked = false;
  retrievalModeSelectEl.value = "dense";
  setSession("");
  messagesEl.innerHTML = "";
  clearSteps();
  ensureEmptyState();
  inputEl.focus();
});

retrievalModeSelectEl.addEventListener("change", () => {
  // Completed traces keep the mode actually used; the new selection affects the next request.
  if (!isRunning && document.querySelector("#retrieval-state").textContent === "Not started") resetRetrievalMap();
});

setSession(sessionId);
resetRetrievalMap();
ensureEmptyState();
