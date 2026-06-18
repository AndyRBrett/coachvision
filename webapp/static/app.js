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

async function openResult(job) {
  $("#result-card").classList.remove("hidden");
  $("#result-name").textContent = job.filename || job.id;
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

  // Full ball trajectory line (toggle).
  const pts = ballPoints();
  if ($("#show-path").checked && pts.length >= 2) {
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "rgba(255, 170, 0, 0.85)";
    ctx.beginPath();
    pts.forEach((p, i) => {
      const x = p.x * sx, y = p.y * sy;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Ball at the current frame (prefer the exact detection, else nearest path point).
  let ball = ev && ev.ball ? { x: ev.ball.center[0], y: ev.ball.center[1] } : null;
  if (!ball) {
    const now = video.currentTime;
    for (const p of pts) { if (p.t <= now) ball = p; else break; }
  }
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

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

refreshJobs();
setInterval(refreshJobs, 5000);
