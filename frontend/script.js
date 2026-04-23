// ═══════════════════════════════════════════════
// FraudShield v3 — Frontend Logic
// Features: particle bg, animated gauge, history,
//           component bars, model weights display
// ═══════════════════════════════════════════════

const API_BASE = "http://localhost:5000";

// ── State ──
const S = {
  txnType:  "pos",
  location: "domestic",
  behavior: "usual",
  timeOverride: false,
  timeSeconds: null,
  history: [],
  txnCount: 0,
  weights: { w_xgb: 0.50, w_rf: 0.25, w_lgbm: 0.25 }, // updated from health check
};

// ═══════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════
window.addEventListener("DOMContentLoaded", () => {
  initParticles();
  tickClock();
  setInterval(tickClock, 30000);
  document.getElementById("amount").addEventListener("input", onAmountInput);
  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && document.activeElement.id === "amount") runAnalysis();
  });
  setStatus("ready", "READY");
  fetchWeights();
});

// ═══════════════════════════════════════════════
// PARTICLE BACKGROUND
// ═══════════════════════════════════════════════
function initParticles() {
  const canvas = document.getElementById("particleCanvas");
  const ctx    = canvas.getContext("2d");
  let W, H, particles;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function Particle() {
    this.x  = Math.random() * W;
    this.y  = Math.random() * H;
    this.vx = (Math.random() - 0.5) * 0.3;
    this.vy = (Math.random() - 0.5) * 0.3;
    this.r  = Math.random() * 1.5 + 0.5;
    this.a  = Math.random() * 0.5 + 0.1;
  }

  function init() {
    resize();
    particles = Array.from({length: 80}, () => new Particle());
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);

    // Draw connections
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < 120) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0,212,255,${0.12 * (1 - d/120)})`;
          ctx.lineWidth = 0.5;
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw nodes
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,212,255,${p.a})`;
      ctx.fill();

      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    });

    requestAnimationFrame(draw);
  }

  init();
  draw();
  window.addEventListener("resize", init);
}

// ═══════════════════════════════════════════════
// CLOCK
// ═══════════════════════════════════════════════
function tickClock() {
  const now  = new Date();
  const h    = now.getHours() % 12 || 12;
  const m    = String(now.getMinutes()).padStart(2, "0");
  const ampm = now.getHours() < 12 ? "AM" : "PM";
  document.getElementById("timeAutoText").textContent =
    `${h}:${m} ${ampm} — auto-detected`;
  if (!S.timeOverride) {
    S.timeSeconds = now.getHours() * 3600 + now.getMinutes() * 60;
  }
}

// ═══════════════════════════════════════════════
// FETCH WEIGHTS FROM BACKEND
// ═══════════════════════════════════════════════
async function fetchWeights() {
  try {
    const res  = await fetch(`${API_BASE}/`);
    const data = await res.json();
    if (data.ensemble) {
      S.weights = {
        w_xgb:  data.ensemble.xgb_weight,
        w_rf:   data.ensemble.rf_weight,
        w_lgbm: data.ensemble.lgbm_weight,
      };
    }
  } catch (e) {
    // backend not running yet — use defaults
  }
}

// ═══════════════════════════════════════════════
// OPTION SELECTION
// ═══════════════════════════════════════════════
function pick(group, val, el) {
  S[group] = val;
  el.closest(".choice-grid").querySelectorAll(".choice-btn")
    .forEach(b => b.classList.remove("active"));
  el.classList.add("active");
  updatePreRisk();
}

// ═══════════════════════════════════════════════
// AMOUNT INPUT
// ═══════════════════════════════════════════════
function setAmount(val) {
  document.getElementById("amount").value = val;
  onAmountInput();
}

function onAmountInput() {
  const v   = parseFloat(document.getElementById("amount").value);
  const err = document.getElementById("amountError");
  if (document.getElementById("amount").value && (v < 0 || v > 50000)) {
    err.textContent = "Amount must be between $0 and $50,000";
  } else {
    err.textContent = "";
  }
  updatePreRisk();
}

// ═══════════════════════════════════════════════
// PRE-ANALYSIS RISK SCORE
// ═══════════════════════════════════════════════
function updatePreRisk() {
  let score = 0;
  if (S.location === "international") score += 0.20;
  if (S.behavior  === "unusual")      score += 0.25;
  if (S.txnType   === "online")       score += 0.12;

  const amt = parseFloat(document.getElementById("amount").value) || 0;
  if (amt > 5000)  score += 0.08;
  if (amt > 15000) score += 0.10;

  const hour = new Date().getHours();
  if (hour < 6) score += 0.10;

  score = Math.min(score, 1);
  const pct = Math.round(score * 100);

  const fill  = document.getElementById("preRiskFill");
  const label = document.getElementById("preRiskVal");

  fill.style.width = `${pct}%`;
  label.textContent = pct > 0 ? `${pct}%` : "—";

  // color shift
  if (score > 0.5)      fill.style.background = "linear-gradient(90deg, #ffb800, #ff3864)";
  else if (score > 0.25) fill.style.background = "linear-gradient(90deg, #00ff9d, #ffb800)";
  else                   fill.style.background = "linear-gradient(90deg, #003333, #00ff9d)";
}

// ═══════════════════════════════════════════════
// TIME OVERRIDE
// ═══════════════════════════════════════════════
function toggleTimeOverride() {
  S.timeOverride = !S.timeOverride;
  const wrap = document.getElementById("timeSliderWrap");
  const btn  = document.getElementById("overrideBtn");
  wrap.classList.toggle("open", S.timeOverride);
  btn.textContent = S.timeOverride ? "USE AUTO" : "OVERRIDE";
  if (!S.timeOverride) tickClock();
}

function updateSlider() {
  const secs = parseInt(document.getElementById("timeSlider").value);
  S.timeSeconds = secs;
  const h    = Math.floor(secs/3600) % 12 || 12;
  const m    = String(Math.floor((secs%3600)/60)).padStart(2, "0");
  const ampm = Math.floor(secs/3600) < 12 ? "AM" : "PM";
  document.getElementById("sliderCurrent").textContent = `${h}:${m} ${ampm}`;
}

// ═══════════════════════════════════════════════
// STATUS
// ═══════════════════════════════════════════════
function setStatus(state, text) {
  const dot  = document.querySelector(".sp-dot");
  const span = document.querySelector(".sp-text");
  dot.className  = `sp-dot ${state}`;
  span.textContent = text;
}

// ═══════════════════════════════════════════════
// MAIN ANALYSIS
// ═══════════════════════════════════════════════
async function runAnalysis() {
  const amtStr = document.getElementById("amount").value.trim();
  if (!amtStr || isNaN(parseFloat(amtStr))) {
    shakeInput();
    document.getElementById("amountError").textContent = "Please enter a transaction amount.";
    return;
  }
  const amount = parseFloat(amtStr);
  if (amount < 0 || amount > 50000) {
    shakeInput();
    document.getElementById("amountError").textContent = "Amount must be 0–$50,000.";
    return;
  }

  const btn = document.getElementById("analyzeBtn");
  const txt = document.getElementById("abText");
  btn.classList.add("loading");
  txt.textContent = "ANALYZING...";
  setStatus("busy", "ANALYZING");

  const payload = {
    amount,
    time: S.timeSeconds,
    transaction_type: S.txnType,
    location: S.location,
    behavior: S.behavior,
  };

  try {
    const res  = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || `Server error (${res.status})`);
      return;
    }

    showResult(data, payload);

  } catch (err) {
    showError(
      "Cannot connect to Flask API.<br>" +
      "<small style='letter-spacing:.04em'>Make sure app.py is running: <code>python app.py</code></small>"
    );
  } finally {
    btn.classList.remove("loading");
    txt.textContent = "ANALYZE TRANSACTION";
  }
}

// ═══════════════════════════════════════════════
// SHOW RESULT
// ═══════════════════════════════════════════════
function showResult(data, payload) {
  const isFraud  = data.prediction === "Fraud";
  const risk     = data.risk_level || "LOW";
  const prob     = data.fraud_probability || 0;
  const probPct  = data.confidence || (prob * 100).toFixed(1);
  const thr      = data.threshold_used || 0;
  const reasons  = data.reasons || [];
  const scores   = data.component_scores || {};
  const base     = data.model_base_score || 0;

  const cls = { HIGH: "fraud", MEDIUM: "medium", LOW: "safe" }[risk];
  const emoji = { HIGH: "🚨", MEDIUM: "⚠️", LOW: "✅" }[risk];

  setStatus(cls, { fraud:"FRAUD DETECTED", medium:"SUSPICIOUS", safe:"SAFE" }[cls]);

  // Show result pane
  document.getElementById("resultIdle").style.display  = "none";
  document.getElementById("resultActive").style.display = "flex";

  // Verdict
  const vb = document.getElementById("verdictBlock");
  vb.className = `verdict-block ${cls}`;
  document.getElementById("verdictIcon").textContent  = emoji;
  document.getElementById("verdictTitle").textContent = isFraud ? "FRAUD DETECTED" : "TRANSACTION SAFE";
  document.getElementById("verdictSub").textContent   = isFraud
    ? `Final probability ${probPct}% exceeds threshold ${(thr*100).toFixed(1)}%`
    : `Final probability ${probPct}% is below threshold ${(thr*100).toFixed(1)}%`;
  const rbl = document.getElementById("riskBadgeLarge");
  rbl.className   = `risk-badge-large ${risk}`;
  rbl.textContent = `${risk} RISK`;

  // Gauge
  animateGauge(parseFloat(probPct), cls);

  // Stats
  document.getElementById("gsBase").textContent    = `${(base*100).toFixed(1)}%`;
  document.getElementById("gsFinal").textContent   = `${probPct}%`;
  document.getElementById("gsThr").textContent     = `${(thr*100).toFixed(1)}%`;
  const gv = document.getElementById("gsVerdict");
  gv.textContent = isFraud ? "FRAUD" : "SAFE";
  gv.style.color = isFraud ? "var(--red)" : "var(--green)";

  // Component bars
  renderComponentBars(scores);

  // Reasons
  renderReasons(reasons, isFraud, risk);

  // Model weights
  renderModelWeights();

  // Add to history
  addHistory(payload, data);

  // Scroll into view
  document.getElementById("resultActive").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// ═══════════════════════════════════════════════
// ANIMATED GAUGE (canvas half-circle)
// ═══════════════════════════════════════════════
let gaugeTarget = 0;
let gaugeCurrent = 0;
let gaugeRAF = null;

function animateGauge(targetPct, riskClass) {
  gaugeTarget = targetPct;
  const color = { fraud:"#ff3864", medium:"#ffb800", safe:"#00ff9d" }[riskClass] || "#00d4ff";
  const el = document.getElementById("gctPct");
  el.style.color = color;

  if (gaugeRAF) cancelAnimationFrame(gaugeRAF);

  function step() {
    gaugeCurrent += (gaugeTarget - gaugeCurrent) * 0.1;
    if (Math.abs(gaugeCurrent - gaugeTarget) < 0.1) gaugeCurrent = gaugeTarget;
    drawGauge(gaugeCurrent, color);
    el.textContent = `${gaugeCurrent.toFixed(1)}%`;
    if (Math.abs(gaugeCurrent - gaugeTarget) > 0.1) {
      gaugeRAF = requestAnimationFrame(step);
    }
  }
  step();
}

function drawGauge(pct, color) {
  const canvas = document.getElementById("gaugeCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H - 16;
  const r = 90;
  const startAngle = Math.PI;
  const endAngle   = Math.PI * 2;
  const fillAngle  = startAngle + (pct / 100) * Math.PI;

  ctx.clearRect(0, 0, W, H);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = "#1a2540";
  ctx.lineWidth = 16;
  ctx.lineCap = "round";
  ctx.stroke();

  // Fill
  if (pct > 0) {
    const grad = ctx.createLinearGradient(cx-r, cy, cx+r, cy);
    grad.addColorStop(0,   "#00ff9d");
    grad.addColorStop(0.5, "#ffb800");
    grad.addColorStop(1,   "#ff3864");

    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, fillAngle);
    ctx.strokeStyle = color;
    ctx.lineWidth = 16;
    ctx.lineCap = "round";
    ctx.stroke();

    // Glow
    ctx.beginPath();
    ctx.arc(cx, cy, r, startAngle, fillAngle);
    ctx.strokeStyle = color.replace(")", ",0.3)").replace("rgb", "rgba");
    ctx.lineWidth = 28;
    ctx.stroke();
  }

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const a = Math.PI + (i / 10) * Math.PI;
    const x1 = cx + (r - 12) * Math.cos(a);
    const y1 = cy + (r - 12) * Math.sin(a);
    const x2 = cx + (r - 20) * Math.cos(a);
    const y2 = cy + (r - 20) * Math.sin(a);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = "rgba(100,130,160,0.4)";
    ctx.lineWidth = i % 5 === 0 ? 2 : 1;
    ctx.stroke();
  }

  // Threshold marker
  const thrPct = gaugeTarget; // use actual value
  const thrAngle = Math.PI + (thrPct / 100) * Math.PI;
  ctx.beginPath();
  ctx.moveTo(cx + (r-30) * Math.cos(thrAngle), cy + (r-30) * Math.sin(thrAngle));
  ctx.lineTo(cx + (r+2)  * Math.cos(thrAngle), cy + (r+2)  * Math.sin(thrAngle));
  ctx.strokeStyle = "rgba(255,255,255,0.3)";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

// ═══════════════════════════════════════════════
// COMPONENT BARS
// ═══════════════════════════════════════════════
function renderComponentBars(scores) {
  const container = document.getElementById("componentBars");
  container.innerHTML = "";

  const labels = {
    ml_model_score:  "ML Model Base",
    location:        "International Location",
    behavior:        "Unusual Behavior",
    transaction_type:"Online Transaction",
    _is_night:       "Night Time (12am-6am)",
    high_amount:     "High Amount",
  };

  const maxVal = Math.max(...Object.values(scores), 0.001);

  Object.entries(scores).forEach(([key, val]) => {
    if (val === 0) return;
    const label = labels[key] || key;
    const pct   = Math.round((val / maxVal) * 100);
    const color = key === "ml_model_score" ? "var(--cyan)" :
                  val > 0.07 ? "var(--red)" : "var(--amber)";

    const row = document.createElement("div");
    row.className = "comp-row";
    row.innerHTML = `
      <div class="comp-row-top">
        <span class="comp-name">${label}</span>
        <span class="comp-val">${(val * 100).toFixed(1)}%</span>
      </div>
      <div class="comp-bar-track">
        <div class="comp-bar-fill" style="width:0%;background:${color}"></div>
      </div>
    `;
    container.appendChild(row);
    setTimeout(() => {
      row.querySelector(".comp-bar-fill").style.width = `${pct}%`;
    }, 50);
  });
}

// ═══════════════════════════════════════════════
// REASONS LIST
// ═══════════════════════════════════════════════
function renderReasons(reasons, isFraud, risk) {
  const list = document.getElementById("reasonsList");
  list.innerHTML = "";

  reasons.forEach((text, i) => {
    const isML   = text.includes("ML ensemble") || text.includes("model");
    const isHigh = text.includes("International") || text.includes("unusual") || text.includes("midnight");
    const type   = isML ? "ri-ml" : isHigh ? (isFraud ? "ri-risk" : "ri-warn") : "ri-safe";

    const item = document.createElement("div");
    item.className = `reason-item ${type}`;
    item.style.animationDelay = `${i * 0.05}s`;
    item.innerHTML = `<div class="ri-dot"></div><div class="ri-text">${text}</div>`;
    list.appendChild(item);
  });
}

// ═══════════════════════════════════════════════
// MODEL WEIGHTS
// ═══════════════════════════════════════════════
function renderModelWeights() {
  const row = document.getElementById("modelWeightsRow");
  const w   = S.weights;

  const models = [
    { name: "XGBOOST",  key: "w_xgb"  },
    { name: "RANDOM FOREST", key: "w_rf"   },
    { name: "LIGHTGBM", key: "w_lgbm" },
  ];

  row.innerHTML = models.map(m => `
    <div class="mw-item">
      <div class="mw-name">${m.name}</div>
      <div class="mw-val">${(w[m.key] * 100).toFixed(0)}%</div>
      <div class="mw-bar"><div class="mw-fill" style="width:${w[m.key]*100}%"></div></div>
    </div>
  `).join("");
}

// ═══════════════════════════════════════════════
// HISTORY
// ═══════════════════════════════════════════════
function addHistory(payload, data) {
  S.txnCount++;
  const isFraud = data.prediction === "Fraud";
  const risk    = data.risk_level;

  const entry = {
    n:        S.txnCount,
    amount:   `$${parseFloat(payload.amount).toLocaleString("en-US", {minimumFractionDigits:2})}`,
    type:     payload.transaction_type?.toUpperCase() || "POS",
    location: payload.location?.toUpperCase() || "DOMESTIC",
    behavior: payload.behavior?.toUpperCase() || "USUAL",
    base:     `${(data.model_base_score * 100).toFixed(1)}%`,
    final:    `${data.confidence}%`,
    verdict:  data.prediction,
    risk,
    isFraud,
  };

  S.history.unshift(entry);
  if (S.history.length > 20) S.history.pop();
  renderHistory();
}

function renderHistory() {
  const tbody = document.getElementById("historyBody");
  const count = document.getElementById("histCount");

  count.textContent = `${S.history.length} entr${S.history.length === 1 ? "y" : "ies"}`;

  if (S.history.length === 0) {
    tbody.innerHTML = `<tr class="history-empty"><td colspan="9">No transactions analyzed yet</td></tr>`;
    return;
  }

  tbody.innerHTML = S.history.map(e => `
    <tr>
      <td style="color:var(--text-3)">${e.n}</td>
      <td style="color:var(--cyan);font-weight:600">${e.amount}</td>
      <td>${e.type}</td>
      <td>${e.location}</td>
      <td>${e.behavior}</td>
      <td>${e.base}</td>
      <td class="${e.isFraud ? "ht-fraud" : "ht-safe"}">${e.final}</td>
      <td class="${e.isFraud ? "ht-fraud" : "ht-safe"}">${e.verdict}</td>
      <td><span class="ht-badge ${e.risk}">${e.risk}</span></td>
    </tr>
  `).join("");
}

function clearHistory() {
  S.history = [];
  S.txnCount = 0;
  renderHistory();
  document.getElementById("histCount").textContent = "0 entries";
}

// ═══════════════════════════════════════════════
// ERROR STATE
// ═══════════════════════════════════════════════
function showError(message) {
  document.getElementById("resultIdle").style.display  = "none";
  document.getElementById("resultActive").style.display = "flex";

  const vb = document.getElementById("verdictBlock");
  vb.className = "verdict-block medium";
  document.getElementById("verdictIcon").textContent  = "⚠️";
  document.getElementById("verdictTitle").textContent = "CONNECTION ERROR";
  document.getElementById("verdictTitle").style.color = "var(--amber)";
  document.getElementById("verdictSub").innerHTML     = message;
  document.getElementById("riskBadgeLarge").textContent = "ERROR";
  document.getElementById("riskBadgeLarge").className    = "risk-badge-large MEDIUM";

  document.getElementById("componentBars").innerHTML = "";
  document.getElementById("reasonsList").innerHTML   = "";
  document.getElementById("modelWeightsRow").innerHTML = "";

  setStatus("medium", "ERROR");
}

// ═══════════════════════════════════════════════
// INPUT SHAKE ANIMATION
// ═══════════════════════════════════════════════
function shakeInput() {
  const wrap = document.querySelector(".amount-wrap");
  wrap.style.animation = "none";
  wrap.style.borderColor = "var(--red)";
  wrap.style.boxShadow   = "0 0 0 3px rgba(255,56,100,0.2)";
  setTimeout(() => {
    wrap.style.borderColor = "";
    wrap.style.boxShadow   = "";
  }, 1200);
}