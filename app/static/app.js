/* Interview Simulator frontend — plain JS, same-origin API, no build step. */

"use strict";

const state = {
  modes: [],
  selectedMode: null,
  candidateId: null,
  interviewId: null,
  maxQuestions: null,
  answered: 0,
};

const $ = (id) => document.getElementById(id);

/* ---------------- api ---------------- */

async function api(path, options = {}) {
  const spinner = $("spinner");
  spinner.classList.remove("hidden");
  try {
    const response = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (typeof body.detail === "string") detail = body.detail;
        else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
      } catch { /* keep the status text */ }
      throw new Error(detail);
    }
    return await response.json();
  } finally {
    spinner.classList.add("hidden");
  }
}

let toastTimer = null;
function toast(message) {
  const el = $("toast");
  el.textContent = message;
  el.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add("hidden"), 6000);
}

function show(view) {
  for (const section of document.querySelectorAll(".view")) section.classList.add("hidden");
  $(`view-${view}`).classList.remove("hidden");
  window.scrollTo({ top: 0 });
}

/* ---------------- setup ---------------- */

async function loadModes() {
  state.modes = await api("/interview-types");
  const grid = $("mode-grid");
  grid.innerHTML = "";
  for (const mode of state.modes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "mode-card";
    button.dataset.key = mode.key;
    const needsContext = mode.topics.length === 0;
    if (needsContext) button.classList.add("needs-context");
    button.innerHTML =
      `<span class="name"></span>` +
      `<span class="detail">${mode.max_questions} questions</span>`;
    button.querySelector(".name").textContent = mode.display_name;
    button.addEventListener("click", () => selectMode(mode.key));
    grid.appendChild(button);
  }
  selectMode(state.modes[0]?.key);
}

function selectMode(key) {
  state.selectedMode = state.modes.find((mode) => mode.key === key) ?? null;
  for (const card of document.querySelectorAll(".mode-card")) {
    card.classList.toggle("selected", card.dataset.key === key);
  }
  // Context-driven modes cannot run without a resume/JD; open the section.
  if (state.selectedMode && state.selectedMode.topics.length === 0) {
    $("context-details").open = true;
  }
}

async function startInterview() {
  const button = $("start-btn");
  button.disabled = true;
  try {
    if (!state.selectedMode) throw new Error("pick an interview type first");
    state.candidateId = $("candidate-id").value.trim() || null;

    const resume = $("resume-text").value.trim();
    const jd = $("jd-text").value.trim();
    let contextId = null;
    if (resume || jd) {
      const context = await api("/candidate-contexts", {
        method: "POST",
        body: JSON.stringify({
          candidate_id: state.candidateId,
          resume_text: resume || null,
          job_description_text: jd || null,
        }),
      });
      contextId = context.context_id;
    }

    const interview = await api("/interviews", {
      method: "POST",
      body: JSON.stringify({
        interview_type: state.selectedMode.key,
        candidate_id: state.candidateId,
        context_id: contextId,
      }),
    });
    state.interviewId = interview.interview_id;
    state.maxQuestions = state.selectedMode.max_questions;
    state.answered = 0;

    $("iv-mode").textContent = state.selectedMode.display_name;
    $("transcript").innerHTML = "";
    show("interview");

    const question = await withThinking(() =>
      api(`/interviews/${state.interviewId}/start`, { method: "POST" })
    );
    renderQuestion(question);
  } catch (error) {
    toast(error.message);
  } finally {
    button.disabled = false;
  }
}

/* ---------------- interview ---------------- */

function difficultyDots(level) {
  return "●".repeat(level) + "○".repeat(5 - level);
}

function renderQuestion(question) {
  $("iv-topic").textContent = question.topic;
  $("iv-difficulty").textContent = difficultyDots(question.difficulty);
  $("iv-progress").textContent = `question ${question.index}` +
    (state.maxQuestions ? ` of ≤${state.maxQuestions}` : "");

  const message = document.createElement("div");
  message.className = "msg q";
  const meta = document.createElement("div");
  meta.className = "qmeta";
  meta.textContent = `Q${question.index} · ${question.topic}`;
  const text = document.createElement("div");
  text.textContent = question.text;
  message.append(meta, text);
  $("transcript").appendChild(message);
  message.scrollIntoView({ behavior: "smooth", block: "end" });
  $("answer-input").focus();
}

function renderAnswer(answerText) {
  const message = document.createElement("div");
  message.className = "msg a";
  message.textContent = answerText;
  $("transcript").appendChild(message);
}

async function withThinking(call) {
  const thinking = document.createElement("div");
  thinking.className = "msg thinking";
  thinking.textContent = "interviewer is thinking…";
  $("transcript").appendChild(thinking);
  thinking.scrollIntoView({ behavior: "smooth", block: "end" });
  try {
    return await call();
  } finally {
    thinking.remove();
  }
}

async function submitAnswer(event) {
  event.preventDefault();
  const input = $("answer-input");
  const answer = input.value.trim();
  if (!answer) return;

  const submit = $("submit-btn");
  submit.disabled = true;
  input.value = "";
  renderAnswer(answer);

  try {
    const result = await withThinking(() =>
      api(`/interviews/${state.interviewId}/answers`, {
        method: "POST",
        body: JSON.stringify({ answer }),
      })
    );
    state.answered += 1;
    if (result.interview_complete || !result.next_question) {
      await showReport();
    } else {
      renderQuestion(result.next_question);
    }
  } catch (error) {
    toast(error.message);
    input.value = answer; // let the candidate retry rather than lose the text
  } finally {
    submit.disabled = false;
  }
}

async function endInterview() {
  if (state.answered === 0 &&
      !window.confirm("End before answering anything? There will be no report scores.")) {
    return;
  }
  try {
    await api(`/interviews/${state.interviewId}/complete`, { method: "POST" });
    await showReport();
  } catch (error) {
    toast(error.message);
  }
}

/* ---------------- report ---------------- */

function chips(listId, items, emptyText) {
  const list = $(listId);
  list.innerHTML = "";
  if (!items || items.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = emptyText;
    list.appendChild(li);
    return;
  }
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    list.appendChild(li);
  }
}

function bars(containerId, entries) {
  const container = $(containerId);
  container.innerHTML = "";
  for (const [name, score] of entries) {
    const row = document.createElement("div");
    row.className = "dim";
    row.innerHTML =
      `<span class="name"></span>` +
      `<div class="bar"><div style="width:0%"></div></div>` +
      `<span class="value"></span>`;
    row.querySelector(".name").textContent = name.replaceAll("_", " ");
    row.querySelector(".value").textContent = score.toFixed(1);
    container.appendChild(row);
    // let the transition animate from 0
    requestAnimationFrame(() =>
      requestAnimationFrame(() => {
        row.querySelector(".bar > div").style.width = `${Math.min(100, score * 10)}%`;
      })
    );
  }
}

async function showReport() {
  const report = await api(`/interviews/${state.interviewId}/report`);

  $("report-meta").textContent =
    `${state.selectedMode?.display_name ?? report.interview_type} · ` +
    `${report.questions_answered} answered · ` +
    `${Math.round(report.duration_seconds / 60)} min`;
  $("overall-score").textContent = report.overall_score.toFixed(1);
  bars("dimension-bars", Object.entries(report.dimension_scores));

  chips("report-strengths", report.strengths, "none recorded");
  chips("report-weaknesses", report.weaknesses, "none recorded");
  chips("report-missed", report.missed_concepts, "nothing missed");
  chips("report-unaddressed", report.unaddressed_target_skills, "all planned skills reached");

  const evidence = $("report-evidence");
  evidence.innerHTML = "";
  for (const line of report.evidence) {
    const li = document.createElement("li");
    const [head, ...rest] = line.split(" — ");
    const strong = document.createElement("strong");
    strong.textContent = head;
    li.appendChild(strong);
    if (rest.length) li.appendChild(document.createTextNode(" — " + rest.join(" — ")));
    evidence.appendChild(li);
  }

  await renderProfile();
  show("report");
}

async function renderProfile() {
  const card = $("profile-card");
  card.classList.add("hidden");
  if (!state.candidateId) return;
  try {
    const profile = await api(`/candidates/${encodeURIComponent(state.candidateId)}/profile`);
    if (!profile.topics.length) return;
    bars("profile-topics", profile.topics.map((entry) => [entry.concept, entry.score]));
    $("profile-focus").textContent = profile.recommended_focus.length
      ? `Recommended focus next: ${profile.recommended_focus.slice(0, 5).join(", ")}`
      : "";
    card.classList.remove("hidden");
  } catch {
    /* profile is a bonus; never block the report on it */
  }
}

/* ---------------- wiring ---------------- */

$("start-btn").addEventListener("click", startInterview);
$("answer-form").addEventListener("submit", submitAnswer);
$("end-btn").addEventListener("click", endInterview);
$("again-btn").addEventListener("click", () => show("setup"));
$("answer-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    $("answer-form").requestSubmit();
  }
});

loadModes().catch((error) => toast(`could not load interview types: ${error.message}`));
