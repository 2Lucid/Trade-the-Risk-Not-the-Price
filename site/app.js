/* Interactive Lab — recomputes the volatility-managed strategy in the browser.
   The math mirrors src/strategy.py exactly (vol-matching constant c, signed
   leverage cap, turnover cost) so the site agrees with the paper. */
"use strict";
const D = window.VMP_DATA;
const M = (D && D.meta && D.meta.tradingMonths) || 12;

/* ---------- numeric helpers ---------- */
const mean = a => a.reduce((s, x) => s + x, 0) / a.length;
const std = a => { const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };
const clip = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const prod1p = a => a.reduce((p, x) => p * (1 + x), 1);
const cumprod1p = a => { let p = 1; return a.map(x => (p *= (1 + x))); };
const fmtPct = x => (x * 100).toFixed(1) + "%";
const fmt2 = x => x.toFixed(2);

function maxDD(r) {
  let eq = 1, peak = 1, mdd = 0;
  for (const x of r) { eq *= (1 + x); peak = Math.max(peak, eq); mdd = Math.min(mdd, eq / peak - 1); }
  return mdd;
}
function turnover(w) { let s = 0, prev = 1; for (const x of w) { s += Math.abs(x - prev); prev = x; } return s / w.length; }

/* ---------- strategy engine (mirrors strategy.py) ---------- */
function managedVol(c, u, f, cap) {
  let acc = [];
  for (let i = 0; i < f.length; i++) acc.push(clip(c * u[i], -cap, cap) * f[i]);
  return std(acc);
}
function solveC(u, f, target, cap) {
  if (!isFinite(cap)) { const raw = u.map((x, i) => x * f[i]); return target / std(raw); }
  const nz = u.filter(x => x !== 0).map(Math.abs);
  const cSat = cap / Math.min(...nz);
  if (managedVol(cSat, u, f, cap) < target) {
    let best = 0, bd = Infinity;
    for (let k = 0; k <= 512; k++) { const cc = cSat * k / 512, d = Math.abs(managedVol(cc, u, f, cap) - target); if (d < bd) { bd = d; best = cc; } }
    return best;
  }
  let lo = 0, hi = cSat;
  for (let i = 0; i < 100; i++) { const mid = .5 * (lo + hi); managedVol(mid, u, f, cap) < target ? lo = mid : hi = mid; }
  return .5 * (lo + hi);
}
function computeStrategy(asset, opt) {
  const useGarch = opt.forecast === "garch" && asset.varGarch;
  const rawVar = useGarch ? asset.varGarch : asset.varNaive;
  // keep months with a usable forecast
  let dates = [], f = [], v = [];
  for (let i = 0; i < asset.f.length; i++) {
    const vv = rawVar[i];
    if (vv != null && !isNaN(vv) && vv > 0) { dates.push(asset.dates[i]); f.push(asset.f[i]); v.push(vv); }
  }
  let signal = new Array(f.length).fill(1), start = 0;
  if (opt.trend) {
    signal = new Array(f.length).fill(1);
    for (let i = 12; i < f.length; i++) { let p = 1; for (let j = i - 12; j < i; j++) p *= (1 + f[j]); signal[i] = (p - 1) < 0 ? -1 : 1; }
    start = 12;
  }
  dates = dates.slice(start); f = f.slice(start); v = v.slice(start); signal = signal.slice(start);
  const n = f.length;
  const u = f.map((_, i) => signal[i] / v[i]);
  const target = std(f);
  const c = solveC(u, f, target, opt.cap);
  const w = u.map(x => clip(c * x, -opt.cap, opt.cap));
  const tc = opt.costBps / 1e4;
  let net = [], prev = 1;
  for (let i = 0; i < n; i++) { net.push(w[i] * f[i] - tc * Math.abs(w[i] - prev)); prev = w[i]; }
  return {
    dates, eqBH: cumprod1p(f), eqNet: cumprod1p(net),
    shBH: mean(f) / std(f) * Math.sqrt(M), shNet: mean(net) / std(net) * Math.sqrt(M),
    annRet: Math.pow(prod1p(net), M / n) - 1, mdd: maxDD(net),
    avgLev: mean(w.map(Math.abs)), turn: turnover(w), n
  };
}

/* ---------- Plotly theme ---------- */
const C = { blue: "#1f6feb", green: "#1a7f37", grey: "#8a93a6", red: "#cf222e", purple: "#8250df" };
const FONT = { family: "Inter, sans-serif", color: "#2a3350", size: 13 };
function layout(extra) {
  return Object.assign({
    font: FONT, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    margin: { l: 56, r: 20, t: 40, b: 48 }, hovermode: "x unified",
    legend: { orientation: "h", y: 1.12, x: 0 },
    xaxis: { gridcolor: "#eef1f7", zeroline: false },
    yaxis: { gridcolor: "#eef1f7", zeroline: false }
  }, extra || {});
}
const CFG = { responsive: true, displayModeBar: false };

/* ---------- static charts ---------- */
function chartR2() {
  const t = D.predictability;
  const x = t.map(r => r.model + "<br>(" + r.target.split(" ")[0] + ")");
  const y = t.map(r => r.r2 * 100);
  const col = t.map(r => r.target.startsWith("Return") ? C.red : C.green);
  Plotly.newPlot("chartR2", [{
    type: "bar", x, y, marker: { color: col },
    text: y.map(v => v.toFixed(1)), textposition: "outside", hoverinfo: "x+y"
  }], layout({ yaxis: { title: "Out-of-sample R² (%)", gridcolor: "#eef1f7" }, showlegend: false }), CFG);
}
function chartCross() {
  const t = D.crossAsset, x = t.map(r => r.asset);
  const bars = [
    { name: "Buy & hold", y: t.map(r => r.bh), color: C.grey },
    { name: "Vol-managed", y: t.map(r => r.managed), color: C.blue },
    { name: "+ Trend", y: t.map(r => r.trend), color: C.green }
  ].map(b => ({ type: "bar", name: b.name, x, y: b.y, marker: { color: b.color } }));
  Plotly.newPlot("chartCross", bars, layout({ barmode: "group", yaxis: { title: "Sharpe ratio", gridcolor: "#eef1f7" }, xaxis: { tickangle: -35 } }), CFG);
}
function chartDiv() {
  const d = D.diversified, t = [
    { name: "Diversified buy & hold", r: d.bh, color: C.grey, w: 1.6 },
    { name: "Diversified managed", r: d.managed, color: C.blue, w: 1.6 },
    { name: "Managed + trend", r: d.trend, color: C.green, w: 2.4 }
  ].map(s => ({ type: "scatter", mode: "lines", name: s.name, x: d.dates, y: cumprod1p(s.r), line: { color: s.color, width: s.w } }));
  Plotly.newPlot("chartDiv", t, layout({ yaxis: { title: "Growth of $1 (log)", type: "log", gridcolor: "#eef1f7" } }), CFG);
}
function chartRegime() {
  const g = D.regimes, turb = g.labels.length - 1;
  const shapes = []; let runStart = null;
  for (let i = 0; i < g.state.length; i++) {
    const t = g.state[i] === turb;
    if (t && runStart === null) runStart = g.dates[i];
    if ((!t || i === g.state.length - 1) && runStart !== null) {
      shapes.push({ type: "rect", xref: "x", yref: "paper", x0: runStart, x1: g.dates[i], y0: 0, y1: 1, fillcolor: C.red, opacity: .12, line: { width: 0 } });
      runStart = null;
    }
  }
  Plotly.newPlot("chartRegime", [{
    type: "scatter", mode: "lines", x: g.dates, y: g.price, line: { color: "#1a2238", width: 1.7 }, name: "Market"
  }], layout({ shapes, showlegend: false, yaxis: { title: "Market index (log)", type: "log", gridcolor: "#eef1f7" } }), CFG);
}

/* ---------- comparison table ---------- */
function buildTable() {
  const rows = D.comparison.map(r => {
    const a = r.alphaT == null ? "—" : "t=" + r.alphaT.toFixed(2);
    const hl = r.strategy.includes("trend") ? ' class="hl"' : "";
    return `<tr${hl}><td>${r.strategy}</td><td>${r.sharpe.toFixed(2)}</td><td>${fmtPct(r.maxDD)}</td><td>${a}</td></tr>`;
  }).join("");
  document.getElementById("cmpTable").innerHTML =
    `<thead><tr><th>Portfolio</th><th>Sharpe</th><th>Max DD</th><th>Alpha</th></tr></thead><tbody>${rows}</tbody>`;
  const ns = D.netSharpe;
  document.getElementById("netNote").textContent =
    `Net of ${ns.cost_bps} bps costs, the managed + trend Sharpe is ${ns.diversified_trend_net.toFixed(2)}. All portfolios vol-matched to the diversified buy & hold.`;
}

/* ---------- KPI band (hero) ---------- */
function kpiBand() {
  const us = D.assets["US market"];
  const head = computeStrategy(us, { forecast: "naive", trend: false, cap: 2, costBps: 0 });
  const trend = D.comparison.find(r => r.strategy.includes("trend")) || {};
  const cards = [
    { lab: "US Sharpe ratio", val: `${fmt2(head.shBH)} → <span class="up">${fmt2(head.shNet)}</span>`, sub: "buy & hold → vol-managed (2× cap)" },
    { lab: "US max drawdown", val: `<span class="up">${fmtPct(head.mdd)}</span>`, sub: "vs −54% buy & hold" },
    { lab: "Diversified + trend", val: fmt2(trend.sharpe || 0), sub: `Sharpe across ${D.meta.nAssets} assets` },
    { lab: "Diversified alpha", val: "t = " + (trend.alphaT || 0).toFixed(1), sub: "significant, Newey-West" }
  ];
  document.getElementById("kpiBand").innerHTML = cards.map(c =>
    `<div class="kpi"><div class="k-lab">${c.lab}</div><div class="k-val">${c.val}</div><div class="k-sub">${c.sub}</div></div>`).join("");
}

/* ---------- THE LAB ---------- */
const state = { asset: "US market", forecast: "naive", trend: false, cap: 2, costBps: 10 };

function renderLab() {
  const asset = D.assets[state.asset];
  document.getElementById("assetClass").textContent = asset.cls;
  // garch availability
  const hasGarch = !!asset.varGarch;
  const garchBtn = document.querySelector('#fcSeg [data-fc="garch"]');
  garchBtn.disabled = !hasGarch;
  document.getElementById("fcHint").textContent = hasGarch
    ? "GARCH(1,1) refit walk-forward." : "GARCH available for the US market only — using naive here.";
  if (!hasGarch && state.forecast === "garch") setForecast("naive");

  const r = computeStrategy(asset, state);
  const col = state.trend ? C.green : C.blue;
  const name = state.asset + " — buy & hold vs your strategy";
  const traces = [
    { type: "scatter", mode: "lines", name: "Buy & hold", x: r.dates, y: r.eqBH, line: { color: C.grey, width: 1.5 } },
    { type: "scatter", mode: "lines", name: state.trend ? "Managed + trend" : "Vol-managed", x: r.dates, y: r.eqNet, line: { color: col, width: 2.4 } }
  ];
  Plotly.react("chartLab", traces, layout({ title: { text: name, font: { size: 14 } }, yaxis: { title: "Growth of $1 (log)", type: "log", gridcolor: "#eef1f7" } }), CFG);

  const dSh = r.shNet - r.shBH;
  const minis = [
    { lab: "Sharpe (strategy)", val: fmt2(r.shNet), sub: `buy & hold ${fmt2(r.shBH)} · Δ ${dSh >= 0 ? "+" : ""}${fmt2(dSh)}` },
    { lab: "Annual return", val: fmtPct(r.annRet), sub: "net of costs, CAGR" },
    { lab: "Max drawdown", val: fmtPct(r.mdd), sub: "worst peak-to-trough" },
    { lab: "Avg leverage", val: r.avgLev.toFixed(2) + "×", sub: `turnover ${(r.turn * 100).toFixed(0)}%/mo` }
  ];
  document.getElementById("labKpis").innerHTML = minis.map(m =>
    `<div class="mini"><div class="m-lab">${m.lab}</div><div class="m-val">${m.val}</div><div class="m-sub">${m.sub}</div></div>`).join("");
}

function setForecast(fc) {
  state.forecast = fc;
  document.querySelectorAll("#fcSeg button").forEach(b => b.classList.toggle("on", b.dataset.fc === fc));
  renderLab();
}
function setTrend(on) {
  state.trend = on;
  document.querySelectorAll("#trendSeg button").forEach(b => b.classList.toggle("on", (b.dataset.trend === "on") === on));
  renderLab();
}

function wireControls() {
  const sel = document.getElementById("assetSel");
  sel.innerHTML = D.assetOrder.map(a => `<option value="${a}">${a}</option>`).join("");
  sel.value = state.asset;
  sel.addEventListener("change", e => { state.asset = e.target.value; renderLab(); });

  document.querySelectorAll("#fcSeg button").forEach(b => b.addEventListener("click", () => { if (!b.disabled) setForecast(b.dataset.fc); }));
  document.querySelectorAll("#trendSeg button").forEach(b => b.addEventListener("click", () => setTrend(b.dataset.trend === "on")));

  const capS = document.getElementById("capSlider"), capV = document.getElementById("capVal");
  capS.addEventListener("input", () => {
    const v = parseFloat(capS.value);
    state.cap = v >= 5 ? Infinity : v;
    capV.textContent = v >= 5 ? "∞ (uncapped)" : v.toFixed(1) + "×";
    renderLab();
  });
  const costS = document.getElementById("costSlider"), costV = document.getElementById("costVal");
  costS.addEventListener("input", () => { state.costBps = parseFloat(costS.value); costV.textContent = state.costBps + " bps"; renderLab(); });

  document.getElementById("resetBtn").addEventListener("click", () => {
    Object.assign(state, { asset: "US market", forecast: "naive", trend: false, cap: 2, costBps: 10 });
    sel.value = "US market"; capS.value = 2; capV.textContent = "2.0×"; costS.value = 10; costV.textContent = "10 bps";
    setForecast("naive"); setTrend(false);
  });
}

/* ---------- reveal on scroll ---------- */
function reveal() {
  const els = document.querySelectorAll(".section-head, .card, .step, .kpi");
  els.forEach(e => e.classList.add("reveal"));
  const io = new IntersectionObserver(es => es.forEach(en => { if (en.isIntersecting) { en.target.classList.add("in"); io.unobserve(en.target); } }), { threshold: .12 });
  els.forEach(e => io.observe(e));
}

/* ---------- init ---------- */
document.addEventListener("DOMContentLoaded", () => {
  if (!D) { document.body.innerHTML = "<p style='padding:40px'>data.js failed to load.</p>"; return; }
  document.getElementById("year").textContent = "2026";
  document.getElementById("footMeta").textContent = `${D.meta.dataStart} → ${D.meta.dataEnd} · ${D.meta.nMonthly} months`;
  kpiBand(); chartR2(); chartCross(); chartDiv(); chartRegime(); buildTable();
  wireControls(); renderLab(); reveal();
});
