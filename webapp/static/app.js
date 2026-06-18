// Volleyball Analysis — frontend shell.
// Uploads a clip, polls job status, and renders the annotated video with the
// ball's full trajectory drawn as a path overlay.

const $ = (sel) => document.querySelector(sel);

const pollers = new Set();

async function refreshJobs() {
  let jobs = [];
  try {
    jobs = await (await fetch("/api/jobs")).json();
  } catch (e) {
    return;
  }
  const ul = $("#jobs");
  ul.innerHTML = "";
  if (!jobs.length) {
    ul.innerHTML = '<li class="empty">No jobs yet — upload a clip above.</li>';
    return;
  }
  for (const job of jobs) {
    const li = document.createElement("li");
    li.className = `job ${job.status}`;
    const stats = job.status === "done"
      ? `ball in ${job.ball_pct ?? 0}% of ${job.frames ?? 0} frames`
      : (job.message || job.status);
    li.innerHTML = `
      <span class="job-name">${escapeHtml(job.filename || job.id)}</span>
      <span class="badge ${job.status}">${job.status}</span>
      <span class="job-stats">${escapeHtml(String(stats))}</span>`;
    if (job.status === "done") {
      li.classList.add("clickable");
      li.addEventListener("click", () => openResult(job));
    }
    ul.appendChild(li);
  }
}

function pollJob(jobId) {
  if (pollers.has(jobId)) return;
  pollers.add(jobId);
  const tick = async () => {
    let st;
    try {
      st = await (await fetch(`/api/jobs/${jobId}`)).json();
    } catch (e) {
      st = null;
    }
    await refreshJobs();
    if (st && (st.status === "done" || st.status === "error")) {
      pollers.delete(jobId);
      if (st.status === "done") openResult(st);
      return;
    }
    setTimeout(tick, 2000);
  };
  tick();
}

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = $("#file");
  if (!fileInput.files.length) return;
  const fd = new FormData();
  fd.append("file", fileInput.files[0]);
  fd.append("stride", $("#stride").value);
  fd.append("ball_conf", $("#ball-conf").value);

  const btn = $("#upload-btn");
  btn.disabled = true;
  $("#upload-msg").textContent = "Uploading…";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    const { job_id } = await res.json();
    $("#upload-msg").textContent = `Queued (job ${job_id}). Processing…`;
    fileInput.value = "";
    pollJob(job_id);
  } catch (err) {
    $("#upload-msg").textContent = `Upload failed: ${err.message}`;
  } finally {
    btn.disabled = false;
  }
});

// ---- Result view: video + ball-path overlay ----

let events = null;      // parsed events.json
let metrics = null;     // parsed metrics.json
let frameSize = [0, 0]; // native [w, h] the coords are in
let currentJobId = null;
let calibrating = false;
let cornersImg = [];    // clicked court corners in native image px

async function openResult(job) {
  $("#result-card").classList.remove("hidden");
  $("#result-name").textContent = job.filename || job.id;
  currentJobId = job.id;
  // Reset calibration UI for the newly opened job.
  calibrating = false;
  cornersImg = [];
  canvas.style.pointerEvents = "none";
  $("#calib-panel").classList.add("hidden");
  $("#calibrated").classList.add("hidden");
  const ballNote = job.ball_enabled === false
    ? " (players only — no ball model)"
    : `, ball in ${job.ball_pct ?? 0}% of frames`;
  $("#result-stats").textContent = `${job.frames ?? 0} frames processed${ballNote}.`;

  const video = $("#video");
  // Play the original clip (browser-playable); overlays are drawn on the canvas.
  video.src = `/api/jobs/${job.id}/source`;

  try {
    const data = await (await fetch(`/api/jobs/${job.id}/events`)).json();
    events = data.events || [];
    frameSize = data.frame_size || [0, 0];
  } catch (e) {
    events = [];
  }
  try {
    const res = await fetch(`/api/jobs/${job.id}/metrics`);
    metrics = res.ok ? await res.json() : null;
  } catch (e) {
    metrics = null;
  }
  renderMetrics();
  loadCachedCoaching();
  $("#result-card").scrollIntoView({ behavior: "smooth" });
}

function renderMetrics() {
  const grid = $("#metric-grid");
  const rallyList = $("#rally-list");
  const playerList = $("#player-list");
  if (!metrics) {
    grid.innerHTML = '<div class="metric"><span class="big">—</span>metrics unavailable</div>';
    rallyList.innerHTML = playerList.innerHTML = "";
    return;
  }
  const b = metrics.ball || {};
  const cards = [
    [metrics.rally_count ?? 0, "rallies"],
    [metrics.players?.track_count ?? 0, "players tracked"],
    [Math.round(b.avg_speed_px_s ?? 0), "avg ball speed (px/s)"],
    [Math.round(b.max_speed_px_s ?? 0), "max ball speed (px/s)"],
  ];
  grid.innerHTML = cards.map(([v, l]) =>
    `<div class="metric"><span class="big">${v}</span>${l}</div>`).join("");

  rallyList.innerHTML = (metrics.rallies || []).length
    ? metrics.rallies.map((r) =>
        `<li>Rally ${r.index}: ${r.duration_s}s <span class="muted">(${r.start_s}–${r.end_s}s)</span></li>`).join("")
    : '<li class="muted">No rallies detected.</li>';

  const tracks = (metrics.players?.per_track || []).slice().sort((a, b) => b.distance_px - a.distance_px);
  playerList.innerHTML = tracks.length
    ? tracks.map((p) =>
        `<li>P${p.track_id}: ${p.distance_px}px <span class="muted">(${p.frames_seen} frames)</span></li>`).join("")
    : '<li class="muted">No tracked players.</li>';

  drawHeatmap(metrics.players?.heatmap);
  renderCalibrated();
}

function drawHeatmap(hm) {
  const cv = $("#heatmap");
  const hctx = cv.getContext("2d");
  hctx.clearRect(0, 0, cv.width, cv.height);
  if (!hm || !hm.grid) return;
  const { cols, rows, grid } = hm;
  const cw = cv.width / cols, ch = cv.height / rows;
  let max = 0;
  for (const row of grid) for (const v of row) if (v > max) max = v;
  if (max === 0) return;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = grid[r][c] / max;
      if (v <= 0) continue;
      hctx.fillStyle = `rgba(255, ${Math.round(170 * (1 - v))}, 0, ${0.15 + 0.85 * v})`;
      hctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
    }
  }
}

const video = $("#video");
const canvas = $("#overlay");
const ctx = canvas.getContext("2d");

function sizeCanvas() {
  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;
  drawOverlay();
}

function ballPoints() {
  // Prefer the interpolated metrics path (smoother, gaps filled); fall back to
  // raw per-frame detections from events.
  if (metrics && metrics.ball && Array.isArray(metrics.ball.path) && metrics.ball.path.length) {
    return metrics.ball.path.map((p) => ({ t: p.time_s, x: p.x, y: p.y }));
  }
  if (!events) return [];
  return events
    .filter((ev) => ev.ball && ev.ball.center)
    .map((ev) => ({ t: ev.time_s, x: ev.ball.center[0], y: ev.ball.center[1] }));
}

const BALL_GAP_S = 0.5;  // gap longer than this = ball is absent; don't bridge it

// Ball position interpolated to an exact time `now` (between the two nearest
// samples), so the marker tracks smoothly instead of snapping to the last frame.
// Returns null across a long gap, so the ball goes absent when it's out of frame
// instead of sliding across where it isn't.
function ballAt(now, pts) {
  if (!pts.length) return null;
  let prev = null, next = null;
  for (const p of pts) {
    if (p.t <= now) prev = p;
    else { next = p; break; }
  }
  if (prev && next) {
    if (next.t - prev.t > BALL_GAP_S) return null;  // ball absent in this gap
    const span = next.t - prev.t || 1;
    const a = (now - prev.t) / span;
    return { x: prev.x + (next.x - prev.x) * a, y: prev.y + (next.y - prev.y) * a };
  }
  // Only show the lone endpoint if `now` is right next to it (not far past/before).
  const p = prev || next;
  return p && Math.abs(p.t - now) <= BALL_GAP_S ? p : null;
}

function currentEvent() {
  if (!events || !events.length) return null;
  const now = video.currentTime;
  let cur = events[0];
  for (const e of events) {
    if (e.time_s <= now) cur = e; else break;
  }
  return cur;
}

function drawOverlay() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!frameSize[0]) return;
  const sx = canvas.width / frameSize[0];
  const sy = canvas.height / frameSize[1];

  // Player boxes + IDs for the current frame.
  const ev = currentEvent();
  if (ev && ev.players) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(46, 204, 113, 0.9)";
    ctx.fillStyle = "rgba(46, 204, 113, 0.95)";
    ctx.font = "12px -apple-system, sans-serif";
    for (const p of ev.players) {
      const [x1, y1, x2, y2] = p.bbox;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      if (p.track_id != null) {
        ctx.fillText("P" + p.track_id, x1 * sx, Math.max(11, y1 * sy - 3));
      }
    }
  }

  const pts = ballPoints();
  const now = video.currentTime;

  // Ball path as a trailing "comet" — only the recent stretch up to now, so it
  // follows the ball instead of drawing the whole rally as a static tangle.
  if ($("#show-path").checked && pts.length >= 2) {
    const TRAIL_S = 1.5;
    const seg = pts.filter((p) => p.t >= now - TRAIL_S && p.t <= now);
    for (let i = 1; i < seg.length; i++) {
      if (seg[i].t - seg[i - 1].t > BALL_GAP_S) continue;  // don't bridge a gap
      const a = i / seg.length;  // fade older segments
      ctx.strokeStyle = `rgba(255, 170, 0, ${0.2 + 0.8 * a})`;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(seg[i - 1].x * sx, seg[i - 1].y * sy);
      ctx.lineTo(seg[i].x * sx, seg[i].y * sy);
      ctx.stroke();
    }
  }

  // Ball marker, interpolated to the exact current time (smooth, no lag).
  const ball = ballAt(now, pts);
  if (ball) {
    ctx.beginPath();
    ctx.arc(ball.x * sx, ball.y * sy, 8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255, 60, 60, 0.95)";
    ctx.fill();
  }
}

let rafId = null;
function loop() { drawOverlay(); rafId = requestAnimationFrame(loop); }
function stopLoop() { if (rafId) { cancelAnimationFrame(rafId); rafId = null; } }

video.addEventListener("loadedmetadata", sizeCanvas);
video.addEventListener("play", () => { if (!rafId) loop(); });
video.addEventListener("pause", () => { stopLoop(); drawOverlay(); });
video.addEventListener("ended", stopLoop);
video.addEventListener("timeupdate", drawOverlay);  // covers paused scrubbing
video.addEventListener("seeked", drawOverlay);
window.addEventListener("resize", sizeCanvas);
$("#show-path").addEventListener("change", drawOverlay);

// ---- Court calibration ----

const CORNER_LABELS = [
  "near-left (front-left)", "near-right (front-right)",
  "far-right (back-right)", "far-left (back-left)",
];

function startCalibration() {
  if (!currentJobId || !frameSize[0]) return;
  video.pause();
  stopLoop();
  calibrating = true;
  cornersImg = [];
  canvas.style.pointerEvents = "auto";
  $("#calib-panel").classList.remove("hidden");
  drawCalibration();
  updateCalibInstr();
}

function updateCalibInstr() {
  const n = cornersImg.length;
  if (n < 4) {
    $("#calib-instr").textContent = `Click the ${CORNER_LABELS[n]} corner (${n + 1}/4).`;
    $("#calib-apply").disabled = true;
  } else {
    $("#calib-instr").textContent = "4 corners set — click Apply, or Reset to redo.";
    $("#calib-apply").disabled = false;
  }
}

function drawCalibration() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!frameSize[0]) return;
  const sx = canvas.width / frameSize[0];
  const sy = canvas.height / frameSize[1];
  ctx.fillStyle = "rgba(61, 169, 252, 0.95)";
  ctx.strokeStyle = "rgba(61, 169, 252, 0.95)";
  ctx.lineWidth = 2;
  ctx.font = "13px -apple-system, sans-serif";
  if (cornersImg.length >= 2) {
    ctx.beginPath();
    cornersImg.forEach((c, i) => {
      const x = c[0] * sx, y = c[1] * sy;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    if (cornersImg.length === 4) ctx.closePath();
    ctx.stroke();
  }
  cornersImg.forEach((c, i) => {
    const x = c[0] * sx, y = c[1] * sy;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(String(i + 1), x + 8, y - 8);
  });
}

function onCanvasClick(e) {
  if (!calibrating || cornersImg.length >= 4) return;
  const sx = canvas.width / frameSize[0];
  const sy = canvas.height / frameSize[1];
  cornersImg.push([e.offsetX / sx, e.offsetY / sy]);
  drawCalibration();
  updateCalibInstr();
}

async function applyCalibration() {
  if (cornersImg.length !== 4 || !currentJobId) return;
  $("#calib-apply").disabled = true;
  $("#calib-instr").textContent = "Calibrating…";
  try {
    const body = {
      corners: cornersImg,
      court: $("#court-type").value,
      camera_moves: $("#camera-moves").checked,
    };
    const res = await fetch(`/api/jobs/${currentJobId}/calibrate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const out = await res.json();
    metrics = metrics || {};
    metrics.players_court = out.players_court;
    metrics.calibration = out.calibration;
    calibrating = false;
    canvas.style.pointerEvents = "none";
    $("#calib-panel").classList.add("hidden");
    drawOverlay();
    renderCalibrated();
    $("#calibrated").scrollIntoView({ behavior: "smooth" });
  } catch (err) {
    $("#calib-instr").textContent = "Calibration failed: " + err.message;
    $("#calib-apply").disabled = false;
  }
}

function renderCalibrated() {
  if (!metrics || !metrics.players_court) {
    $("#calibrated").classList.add("hidden");
    return;
  }
  $("#calibrated").classList.remove("hidden");
  const pc = metrics.players_court;
  const cal = metrics.calibration || {};
  $("#calib-approx").classList.toggle("hidden", !cal.approximate);
  const cards = [
    [pc.track_count_in_court ?? 0, "players on court"],
    [(cal.court_m || []).join("×") + " m", "court size"],
  ];
  $("#court-grid").innerHTML = cards.map(([v, l]) =>
    `<div class="metric"><span class="big">${v}</span>${l}</div>`).join("");
  $("#court-players").innerHTML = (pc.per_track || []).length
    ? pc.per_track.map((p) => {
        const spd = p.top_speed_kmh != null
          ? ` · top ${p.top_speed_kmh} km/h (${p.top_speed_mph} mph)` : "";
        return `<li>P${p.track_id}: ${p.distance_m} m${spd} <span class="muted">(${p.frames_in_court} frames)</span></li>`;
      }).join("")
    : '<li class="muted">No players detected inside the court.</li>';
  drawCourtHeatmap(pc.heatmap);
}

function drawCourtHeatmap(hm) {
  const cv = $("#court-heatmap");
  const c = cv.getContext("2d");
  c.clearRect(0, 0, cv.width, cv.height);
  c.fillStyle = "#11141a";
  c.fillRect(0, 0, cv.width, cv.height);
  if (!hm || !hm.grid) return;
  const { cols, rows, grid } = hm;
  const pad = 10, w = cv.width - 2 * pad, h = cv.height - 2 * pad;
  const cw = w / cols, ch = h / rows;
  let max = 0;
  for (const r of grid) for (const v of r) if (v > max) max = v;
  if (max > 0) {
    for (let r = 0; r < rows; r++) {
      for (let cc = 0; cc < cols; cc++) {
        const v = grid[r][cc] / max;
        if (v <= 0) continue;
        c.fillStyle = `rgba(255, ${Math.round(170 * (1 - v))}, 0, ${0.15 + 0.85 * v})`;
        c.fillRect(pad + cc * cw, pad + r * ch, cw + 0.5, ch + 0.5);
      }
    }
  }
  // Court outline + net line (length axis runs along columns; net at mid-length).
  c.strokeStyle = "rgba(255,255,255,0.4)";
  c.lineWidth = 1;
  c.strokeRect(pad, pad, w, h);
  c.beginPath();
  c.moveTo(pad + w / 2, pad);
  c.lineTo(pad + w / 2, pad + h);
  c.stroke();
}

// ---- Coaching analysis (Claude) ----

async function loadCachedCoaching() {
  $("#coach-output").classList.add("hidden");
  $("#coach-msg").textContent = "";
  $("#coach-btn").disabled = false;
  $("#coach-btn").textContent = "Generate coaching analysis";
  if (!currentJobId) return;
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/coaching`);
    if (res.ok) renderCoaching(await res.json());
  } catch (e) { /* none yet — leave the button */ }
}

async function generateCoaching() {
  if (!currentJobId) return;
  $("#coach-btn").disabled = true;
  $("#coach-msg").textContent = "Analyzing with Claude… (this can take ~20s)";
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/coach`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    renderCoaching(await res.json());
    $("#coach-msg").textContent = "";
  } catch (err) {
    $("#coach-msg").textContent = "Coaching failed: " + err.message;
    $("#coach-btn").disabled = false;
  }
}

function renderCoaching(data) {
  $("#coach-output").innerHTML = miniMarkdown(data.analysis || "");
  $("#coach-output").classList.remove("hidden");
  $("#coach-btn").textContent = "Regenerate analysis";
  $("#coach-btn").disabled = false;
}

// Minimal, XSS-safe markdown: escape first, then headings/bold/bullets.
function miniMarkdown(md) {
  const lines = escapeHtml(md).split("\n");
  let html = "", inList = false;
  for (let line of lines) {
    line = line.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    if (/^#{1,6}\s/.test(line)) {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<h4>${line.replace(/^#{1,6}\s/, "")}</h4>`;
    } else if (/^\s*[-*]\s/.test(line)) {
      if (!inList) { html += "<ul>"; inList = true; }
      html += `<li>${line.replace(/^\s*[-*]\s/, "")}</li>`;
    } else if (line.trim() === "") {
      if (inList) { html += "</ul>"; inList = false; }
    } else {
      if (inList) { html += "</ul>"; inList = false; }
      html += `<p>${line}</p>`;
    }
  }
  if (inList) html += "</ul>";
  return html;
}

async function recomputeMetrics() {
  if (!currentJobId) return;
  const btn = $("#recompute-btn");
  btn.disabled = true;
  const orig = btn.textContent;
  btn.textContent = "Recomputing…";
  try {
    const res = await fetch(`/api/jobs/${currentJobId}/recompute`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    metrics = await res.json();
    renderMetrics();
    drawOverlay();
  } catch (err) {
    $("#result-stats").textContent = "Recompute failed: " + err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}

$("#recompute-btn").addEventListener("click", recomputeMetrics);
$("#coach-btn").addEventListener("click", generateCoaching);

$("#calib-btn").addEventListener("click", startCalibration);
$("#calib-reset").addEventListener("click", () => {
  cornersImg = [];
  drawCalibration();
  updateCalibInstr();
});
$("#calib-apply").addEventListener("click", applyCalibration);
canvas.addEventListener("click", onCanvasClick);

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

refreshJobs();
setInterval(refreshJobs, 5000);
