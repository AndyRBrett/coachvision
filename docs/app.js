"use strict";

// coachvision PWA — a serverless phone interface to the GitHub pipeline.
// Reads reports/ via the GitHub contents API (works for public and private
// repos) and triggers the process-footage workflow via the Actions API.

const API = "https://api.github.com";
const CFG_KEY = "coachvision.cfg";
const DEFAULTS = { owner: "AndyRBrett", repo: "coachvision", branch: "main", token: "" };

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

function cfg() {
  try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem(CFG_KEY) || "{}") }; }
  catch { return { ...DEFAULTS }; }
}
function saveCfg(next) { localStorage.setItem(CFG_KEY, JSON.stringify(next)); }

function headers(raw) {
  const c = cfg();
  const h = { "X-GitHub-Api-Version": "2022-11-28" };
  h["Accept"] = raw ? "application/vnd.github.raw" : "application/vnd.github+json";
  if (c.token) h["Authorization"] = `Bearer ${c.token}`;
  return h;
}

// --- GitHub API ---------------------------------------------------------
async function getContent(path) {
  const c = cfg();
  const url = `${API}/repos/${c.owner}/${c.repo}/contents/${path}?ref=${encodeURIComponent(c.branch)}`;
  const res = await fetch(url, { headers: headers(true) });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${await res.text()}`);
  return res.text();
}

// Fetch a binary file (a rendered clip) via the contents API and hand back an
// object URL so it streams into a <video> tag — works for public and private
// repos alike. Caller revokes the URL when done.
async function getBlobUrl(path) {
  const c = cfg();
  const url = `${API}/repos/${c.owner}/${c.repo}/contents/${path}?ref=${encodeURIComponent(c.branch)}`;
  const res = await fetch(url, { headers: headers(true) });
  if (!res.ok) return null;
  return URL.createObjectURL(await res.blob());
}

async function dispatch(inputs) {
  const c = cfg();
  if (!c.token) throw new Error("A GitHub token is required to start a run (Settings).");
  const url = `${API}/repos/${c.owner}/${c.repo}/actions/workflows/process-footage.yml/dispatches`;
  const res = await fetch(url, {
    method: "POST",
    headers: { ...headers(false), "Content-Type": "application/json" },
    body: JSON.stringify({ ref: c.branch, inputs }),
  });
  if (res.status !== 204) throw new Error(`${res.status} — ${await res.text()}`);
}

async function listRuns() {
  const c = cfg();
  const url = `${API}/repos/${c.owner}/${c.repo}/actions/workflows/process-footage.yml/runs?per_page=6`;
  const res = await fetch(url, { headers: headers(false) });
  if (!res.ok) return [];
  return (await res.json()).workflow_runs || [];
}

// --- Sessions -----------------------------------------------------------
async function loadSessions() {
  const msg = $("#sessionsMsg");
  const gallery = $("#gallery");
  gallery.innerHTML = "";
  msg.className = "msg";
  msg.textContent = "Loading…";
  try {
    const text = await getContent("reports/index.json");
    if (text === null) {
      msg.textContent = "";
      gallery.appendChild(el("p", "empty", "No sessions yet. Use Analyze to process your first clip."));
      return;
    }
    const index = JSON.parse(text);
    const clips = index.clips || [];
    msg.textContent = clips.length ? `${clips.length} session(s)` : "";
    if (!clips.length) gallery.appendChild(el("p", "empty", "No sessions yet."));
    for (const clip of clips) gallery.appendChild(card(clip));
  } catch (e) {
    msg.className = "msg err";
    msg.textContent = `Could not load sessions: ${e.message}`;
  }
}

function card(clip) {
  const b = el("button", "card");
  const h = el("h4");
  h.appendChild(el("span", null, clip.title || clip.id));
  const badge = el("span", `badge ${clip.domain === "volleyball" ? "volleyball" : ""}`,
    clip.domain === "volleyball" ? "volleyball" : "martial arts");
  h.appendChild(badge);
  b.appendChild(h);

  const stats = el("div", "stats");
  const seg = clip.domain === "volleyball" ? "rallies" : "exchanges";
  const playable = clip.rendered_count ?? (clip.clips || []).filter((c) => c.video).length;
  stats.innerHTML =
    `<span><b>${clip.segment_count ?? "–"}</b> ${seg}</span>` +
    (playable ? `<span>▶ <b>${playable}</b> clip${playable === 1 ? "" : "s"}</span>` : "") +
    `<span><b>${clip.detected_frames ?? "–"}</b>/${clip.frames_processed ?? "–"} frames</span>`;
  b.appendChild(stats);
  if (clip.processed_at) b.appendChild(el("div", "when", fmtDate(clip.processed_at)));

  b.addEventListener("click", () => openClip(clip));
  return b;
}

let _videoUrls = [];   // object URLs to revoke when the dialog closes

function clearVideos() {
  _videoUrls.forEach((u) => URL.revokeObjectURL(u));
  _videoUrls = [];
  $("#dlgVideos").innerHTML = "";
}

async function openClip(clip) {
  $("#dlgTitle").textContent = clip.title || clip.id;
  clearVideos();
  const body = $("#dlgBody");
  body.textContent = "Loading…";
  $("#clipDialog").showModal();

  // Coaching summary text.
  try {
    const summary = await getContent(`reports/${clip.id}/coaching/summary.txt`);
    body.textContent = summary ?? "No summary found for this clip.";
  } catch (e) {
    body.textContent = `Could not load summary: ${e.message}`;
  }

  // Playable highlight clips. The catalog lists each segment; render a <video>
  // for the ones that actually got encoded.
  const playable = (clip.clips || []).filter((c) => c.video);
  const wrap = $("#dlgVideos");
  if (!playable.length) {
    wrap.appendChild(el("p", "hint", "No rendered clips for this session yet."));
    return;
  }
  for (const c of playable) {
    const seg = el("div", "clip");
    const label = (c.tags && c.tags.length ? c.tags.join(", ") : c.id) + `  ·  ${c.duration}s`;
    seg.appendChild(el("div", "clip-label", label));
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";
    seg.appendChild(video);
    wrap.appendChild(seg);
    getBlobUrl(c.video).then((u) => {
      if (u) { _videoUrls.push(u); video.src = u; }
      else seg.appendChild(el("div", "hint", "(clip unavailable)"));
    });
  }
}

// --- Analyze ------------------------------------------------------------
async function runAnalysis() {
  const msg = $("#analyzeMsg");
  const btn = $("#analyzeBtn");
  const inputs = {
    domain: $("#domain").value,
    fps: String($("#fps").value || "10"),
  };
  const url = $("#clipUrl").value.trim();
  const path = $("#clipPath").value.trim();
  const name = $("#sessionName").value.trim();
  if (name) inputs.name = name;
  if (url) inputs.clip_url = url;
  else if (path) inputs.clip_path = path;
  else { setMsg(msg, "Provide a video URL or a repo path.", "err"); return; }

  btn.disabled = true;
  setMsg(msg, "Starting run…");
  try {
    await dispatch(inputs);
    setMsg(msg, "Run started. It commits reports back when done — pull to refresh Sessions.", "ok");
    setTimeout(loadRuns, 2500);
  } catch (e) {
    setMsg(msg, `Failed: ${e.message}`, "err");
  } finally {
    btn.disabled = false;
  }
}

async function loadRuns() {
  const wrap = $("#runs");
  wrap.innerHTML = "";
  const runs = await listRuns();
  if (!runs.length) { wrap.appendChild(el("p", "empty", "No runs yet.")); return; }
  for (const r of runs) {
    const state = r.status === "completed" ? (r.conclusion || "completed") : r.status;
    const row = el("div", "run");
    row.appendChild(el("span", `dot ${state}`));
    row.appendChild(el("span", null, `${state} · ${fmtDate(r.created_at)}`));
    const a = el("a", null, "open");
    a.href = r.html_url; a.target = "_blank"; a.rel = "noopener";
    row.appendChild(a);
    wrap.appendChild(row);
  }
}

// --- Settings -----------------------------------------------------------
function loadSettings() {
  const c = cfg();
  $("#owner").value = c.owner;
  $("#repo").value = c.repo;
  $("#branch").value = c.branch;
  $("#token").value = c.token;
  updateRepoTag();
}
function saveSettings() {
  saveCfg({
    owner: $("#owner").value.trim() || DEFAULTS.owner,
    repo: $("#repo").value.trim() || DEFAULTS.repo,
    branch: $("#branch").value.trim() || DEFAULTS.branch,
    token: $("#token").value.trim(),
  });
  updateRepoTag();
  setMsg($("#settingsMsg"), "Saved.", "ok");
  loadSessions();
}
async function testConnection() {
  const c = cfg();
  setMsg($("#settingsMsg"), "Testing…");
  try {
    const res = await fetch(`${API}/repos/${c.owner}/${c.repo}`, { headers: headers(false) });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    const repo = await res.json();
    setMsg($("#settingsMsg"), `Connected: ${repo.full_name} (${repo.private ? "private" : "public"}).`, "ok");
  } catch (e) {
    setMsg($("#settingsMsg"), `Failed: ${e.message}`, "err");
  }
}
function updateRepoTag() {
  const c = cfg();
  $("#repoTag").textContent = `${c.owner}/${c.repo}@${c.branch}`;
}

// --- helpers / wiring ---------------------------------------------------
function setMsg(node, text, kind) { node.className = `msg ${kind || ""}`; node.textContent = text; }
function fmtDate(iso) { try { return new Date(iso).toLocaleString(); } catch { return iso; } }

function switchView(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t.dataset.view === name));
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("is-active", v.id === `view-${name}`));
  if (name === "analyze") loadRuns();
}

function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
  $("#refreshBtn").addEventListener("click", loadSessions);
  $("#runsBtn").addEventListener("click", loadRuns);
  $("#analyzeBtn").addEventListener("click", runAnalysis);
  $("#saveBtn").addEventListener("click", saveSettings);
  $("#testBtn").addEventListener("click", testConnection);
  $("#dlgClose").addEventListener("click", () => $("#clipDialog").close());
  $("#clipDialog").addEventListener("close", clearVideos);

  loadSettings();
  loadSessions();

  if ("serviceWorker" in navigator) {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  }
}

document.addEventListener("DOMContentLoaded", init);
