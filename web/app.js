/* MarketTwin AI — front end.
 * All numbers are computed here from the trained model exported by the Python pipeline
 * (data/bundle.js). The language model only explains them; the Veritas check verifies it. */
"use strict";

const DATA = window.MT_DATA;
const MODEL = DATA.model;
const EPISODES = DATA.episodes;
const EVAL = DATA.evaluation;
const FACTORS = MODEL.factors.map(f => f.id);
const ASSETS = MODEL.assets;

const SHORT = {
  SPY: "Market", XLK: "Tech", XLF: "Financials", XLE: "Energy", XLV: "Healthcare",
  XLY: "Consumer disc.", XLP: "Consumer staples", XLI: "Industrials", XLU: "Utilities",
  TLT: "Long bonds", GLD: "Gold",
};

const PORTFOLIOS = [
  { name: "Tech fan", h: { AAPL: 30, NVDA: 25, MSFT: 20, TSLA: 15, META: 10 } },
  { name: "Balanced starter", h: { SPY: 40, TLT: 20, JNJ: 15, KO: 15, GLD: 10 } },
  { name: "Dividends & defensives", h: { KO: 25, PG: 25, JNJ: 20, NEE: 15, XOM: 15 } },
  { name: "Crypto + growth", h: { "BTC-USD": 30, NVDA: 25, TSLA: 25, AMD: 20 } },
];

const SCENARIOS = [
  { id: "tech", name: "Tech sell-off", desc: "Technology stocks fall 20%, like parts of 2022.", shocks: { XLK: -0.20 } },
  { id: "crash", name: "Market crash", desc: "The whole US market drops 30%, close to the 2020 COVID crash.", shocks: { SPY: -0.30 } },
  { id: "rates", name: "Rate shock (2022-style)", desc: "Stocks fall, bonds fall with them, oil stocks rise.", shocks: { SPY: -0.20, TLT: -0.25, XLE: 0.30 } },
  { id: "oil", name: "Oil collapse", desc: "Energy stocks lose a third of their value.", shocks: { XLE: -0.35 } },
  { id: "banks", name: "Banking panic", desc: "Bank stocks drop 25% on fears about deposits.", shocks: { XLF: -0.25 } },
  { id: "safety", name: "Flight to safety", desc: "Stocks slip, investors run to bonds and gold.", shocks: { SPY: -0.15, TLT: 0.10, GLD: 0.10 } },
  { id: "rally", name: "Strong rally", desc: "Markets rise 20%. Stress tests work both ways.", shocks: { SPY: 0.20 } },
];

const state = {
  holdings: {},          // ticker -> raw slider weight
  amount: 1000,
  last: null,            // last simulation result
  evalMode: "full",
};

/* ------------------------------------------------------------------ helpers */
const $ = id => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; };
const r1 = x => Math.round(x * 10) / 10;
const pct = (x, d = 1) => `${x > 0 ? "+" : ""}${(x * 100).toFixed(d)}%`;
const pp = x => (x * 100).toFixed(1);
const usd = x => { const a = Math.abs(x); const s = a >= 100 ? Math.round(a).toLocaleString("en-US") : a.toFixed(2); return `${x < 0 ? "−" : x > 0 ? "+" : ""}$${s}`; };
const usdPlain = x => `$${Math.round(x).toLocaleString("en-US")}`;
const escapeHtml = s => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function assetClass(t) {
  if (t === "BTC-USD") return "Crypto";
  const s = ASSETS[t].sector;
  if (t === "SPY") return "Broad market fund";
  if (s === "TLT") return "Bonds";
  if (s === "GLD") return "Gold";
  return SHORT[s];
}

function weights() {
  const tot = Object.values(state.holdings).reduce((a, b) => a + b, 0) || 1;
  const w = {};
  for (const [t, v] of Object.entries(state.holdings)) w[t] = v / tot;
  return w;
}

/* Solve A x = b (small dense system, partial pivoting). */
function solve(A, b) {
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

/* Conditional expectation of all factors given the shocked ones (log returns). */
function propagate(shocks) {
  const cov = MODEL.stress_cov;
  const ids = Object.keys(shocks).map(k => FACTORS.indexOf(k));
  const s = Object.values(shocks).map(v => Math.log1p(v));
  const Sss = ids.map(i => ids.map(j => cov[i][j]));
  const x = solve(Sss, s);
  const full = FACTORS.map((_, r) => ids.reduce((acc, i, k) => acc + cov[r][i] * x[k], 0));
  ids.forEach((i, k) => { full[i] = s[k]; });
  return full;
}

function assetMove(t, f) {
  const c = ASSETS[t].coefs;
  return Math.expm1(c.reduce((a, b, i) => a + b * f[i], 0));
}

/* ------------------------------------------------------------------ hero */
function renderHero() {
  const o = EVAL.overall_mae.full;
  const rows = [
    ["MarketTwin model", o.ours, true],
    ["Textbook CAPM beta", o.baseline_capm, false],
    ["“Moves like the market”", o.baseline_market, false],
  ];
  const max = Math.max(...rows.map(r => r[1]));
  $("heroStatRows").innerHTML = rows.map(([n, v, ours]) => `
    <div class="stat ${ours ? "ours" : ""}">
      <span class="stat-name">${n}</span><span class="stat-val">${pp(v)} pp</span>
      <div class="stat-track"><div class="stat-fill" style="width:${(v / max) * 100}%"></div></div>
    </div>`).join("");
  $("dataEnd").textContent = `data to ${MODEL.data_end}`;
  $("swVal").textContent = `${1 + MODEL.hyperparams.stress_weight}× more`;
}

/* ------------------------------------------------------------------ builder */
function loadPortfolio(p) {
  state.holdings = { ...p.h };
  document.querySelectorAll("#presets .chip-btn").forEach(b => b.classList.toggle("active", b.dataset.name === p.name));
  refresh();
}

function renderPresets() {
  const box = $("presets");
  PORTFOLIOS.forEach(p => {
    const b = el("button", "chip-btn", p.name);
    b.dataset.name = p.name;
    b.onclick = () => loadPortfolio(p);
    box.appendChild(b);
  });
}

function renderAddSelect() {
  const sel = $("addAsset");
  sel.innerHTML = `<option value="">Choose…</option>` + Object.entries(ASSETS)
    .filter(([t]) => !(t in state.holdings))
    .map(([t, a]) => `<option value="${t}">${t} — ${a.name}</option>`).join("");
  sel.disabled = Object.keys(state.holdings).length >= 6;
}

function renderHoldings() {
  const w = weights();
  const box = $("holdings");
  box.innerHTML = "";
  for (const t of Object.keys(state.holdings)) {
    const a = ASSETS[t];
    const row = el("div", "holding");
    row.innerHTML = `
      <span class="tk">${t}</span>
      <span class="nm">${a.name}<small>${assetClass(t)}</small></span>
      <input type="range" min="1" max="100" value="${state.holdings[t]}" aria-label="Weight of ${t}">
      <span class="pct">${(w[t] * 100).toFixed(0)}%</span>
      <button class="rm" title="Remove ${t}" aria-label="Remove ${t}">×</button>`;
    row.querySelector("input").oninput = e => { state.holdings[t] = +e.target.value; clearPresetHighlight(); refresh(false); };
    row.querySelector(".rm").onclick = () => { delete state.holdings[t]; clearPresetHighlight(); refresh(); };
    box.appendChild(row);
  }
  const n = Object.keys(state.holdings).length;
  $("weightHint").textContent = n < 3 ? "Add at least 3 assets for a meaningful test." : n >= 6 ? "Maximum of 6 assets reached." : "Sliders set each asset's share; shares always add up to 100%.";
}

function clearPresetHighlight() { document.querySelectorAll("#presets .chip-btn").forEach(b => b.classList.remove("active")); }

/* ------------------------------------------------------------------ analyzer */
function analyze() {
  const w = weights();
  const T = Object.keys(w);
  const ci = MODEL.corr.tickers;
  const corr = (a, b) => MODEL.corr.matrix[ci.indexOf(a)][ci.indexOf(b)];
  const hhi = T.reduce((s, t) => s + w[t] ** 2, 0);
  let variance = 0, cw = 0, cs = 0;
  for (const a of T) for (const b of T) {
    const c = corr(a, b);
    variance += w[a] * w[b] * c * ASSETS[a].annual_vol * ASSETS[b].annual_vol;
    if (a !== b) { cw += w[a] * w[b] * c; cs += w[a] * w[b]; }
  }
  const sectors = {};
  T.forEach(t => { const c = assetClass(t); sectors[c] = (sectors[c] || 0) + w[t]; });
  const f10 = propagate({ SPY: -0.10 });
  const mkt10 = T.reduce((s, t) => s + w[t] * assetMove(t, f10), 0);
  const top = Object.entries(sectors).sort((a, b) => b[1] - a[1])[0] || ["—", 0];
  const biggest = T.reduce((m, t) => (w[t] > (w[m] ?? -1) ? t : m), T[0]);
  return {
    w, T, hhi, effN: 1 / hhi, vol: Math.sqrt(Math.max(variance, 0)),
    avgCorr: cs ? cw / cs : 1, sectors, mkt10, topSector: top, biggest,
  };
}

function riskFlags(A) {
  const flags = [];
  const { w, T } = A;
  if (A.biggest && w[A.biggest] > 0.35) flags.push(`<b>${A.biggest}</b> is ${(w[A.biggest] * 100).toFixed(0)}% of your money. One company's bad news would hit the whole portfolio.`);
  if (A.topSector[1] > 0.5) flags.push(`<b>${(A.topSector[1] * 100).toFixed(0)}% in ${A.topSector[0]}.</b> A single sector shock reaches most of your portfolio at once.`);
  if (A.avgCorr > 0.55) flags.push(`<b>Your holdings move together</b> (average correlation ${A.avgCorr.toFixed(2)}). Owning several of them diversifies less than it looks.`);
  if (A.vol > 0.3) flags.push(`<b>High swings:</b> in a typical year this mix moves about ±${(A.vol * 100).toFixed(0)}%, more than the S&P 500's usual ±15–20%.`);
  if (T.includes("BTC-USD")) flags.push(`<b>Bitcoin</b> is poorly explained by stock-market factors, so our scenario numbers for it are the least reliable.`);
  const deep = T.filter(t => ASSETS[t].max_drawdown < -0.6);
  if (deep.length) flags.push(`<b>History check:</b> ${deep.map(t => `${t} once fell ${(ASSETS[t].max_drawdown * -100).toFixed(0)}%`).join(", ")} from a peak since ${deep.map(t => ASSETS[t].history_start.slice(0, 4)).sort()[0]}.`);
  return flags;
}

function renderAnalyzer() {
  const n = Object.keys(state.holdings).length;
  if (!n) { $("metrics").innerHTML = ""; $("sectorBars").innerHTML = ""; $("flags").innerHTML = ""; return; }
  const A = analyze();
  const m = [
    ["Acts like", `${A.effN.toFixed(1)}`, `equal-sized holdings (you own ${n}). Higher = more spread out.`],
    ["Typical yearly swing", `±${(A.vol * 100).toFixed(0)}%`, "From the last 3 years of weekly prices and correlations."],
    ["Moves together", A.avgCorr.toFixed(2), "Average correlation between your holdings (0 = independent, 1 = identical)."],
    ["If the market falls 10%", pct(A.mkt10), "Stress-model estimate for your whole portfolio."],
  ];
  $("metrics").innerHTML = m.map(([a, b, c]) => `<div class="metric"><div class="m-name">${a}</div><div class="m-val">${b}</div><div class="m-note">${c}</div></div>`).join("");
  $("sectorBars").innerHTML = Object.entries(A.sectors).sort((a, b) => b[1] - a[1]).map(([s, v]) =>
    `<div class="bar-row"><span>${s}</span><div class="bar-track"><div class="bar-fill" style="width:${v * 100}%"></div></div><span class="v">${(v * 100).toFixed(0)}%</span></div>`).join("");
  const flags = riskFlags(A);
  $("flags").innerHTML = flags.length
    ? flags.map(f => `<li class="warn">${f}</li>`).join("")
    : `<li class="ok"><b>No major concentration warnings.</b> No single holding or sector dominates. Run a scenario to see how it handles a shock.</li>`;
  state.analysis = { ...A, flags };
}

function refresh(full = true) {
  if (full) { renderHoldings(); renderAddSelect(); }
  else {
    const w = weights();
    document.querySelectorAll(".holding").forEach(row => {
      const t = row.querySelector(".tk").textContent;
      row.querySelector(".pct").textContent = `${(w[t] * 100).toFixed(0)}%`;
    });
  }
  renderAnalyzer();
  if (state.last) rerunLast();
}

/* ------------------------------------------------------------------ simulator */
function shockTags(shocks) {
  return Object.entries(shocks).map(([k, v]) => `<span class="tag ${v < 0 ? "neg" : "pos"}">${SHORT[k]} ${pct(v, 0)}</span>`).join("");
}

function renderScenarios() {
  const g = $("scenarioGrid");
  SCENARIOS.forEach(s => {
    const b = el("button", "scn", `<h4>${s.name}</h4><p>${s.desc}</p><div class="tags">${shockTags(s.shocks)}</div>`);
    b.onclick = () => { markActive(g, b); runScenario({ kind: "scenario", name: s.name, shocks: s.shocks }); };
    g.appendChild(b);
  });
  const rg = $("replayGrid");
  for (const [key, e] of Object.entries(EPISODES)) {
    const ev = EVAL.episodes[key];
    const b = el("button", "scn", `<h4>${e.label}</h4><p>${e.start} → ${e.end}</p><div class="tags"><span class="tag ${ev.spy_move < 0 ? "neg" : "pos"}">S&amp;P 500 ${pct(ev.spy_move, 0)}</span></div>`);
    b.onclick = () => { markActive(rg, b); runReplay(key); };
    rg.appendChild(b);
  }
  const cg = $("customGrid");
  FACTORS.forEach(f => {
    const d = el("div", "cfac");
    d.innerHTML = `<span>${MODEL.factors.find(x => x.id === f).name}</span><span class="cv">0%</span><input type="range" min="-50" max="50" value="0" data-f="${f}">`;
    const inp = d.querySelector("input");
    inp.oninput = () => { d.querySelector(".cv").textContent = `${inp.value > 0 ? "+" : ""}${inp.value}%`; d.classList.toggle("set", +inp.value !== 0); };
    cg.appendChild(d);
  });
  $("runCustom").onclick = () => {
    const shocks = {};
    cg.querySelectorAll("input").forEach(i => { if (+i.value !== 0) shocks[i.dataset.f] = +i.value / 100; });
    if (!Object.keys(shocks).length) { alert("Move at least one slider away from 0%."); return; }
    runScenario({ kind: "custom", name: "Custom scenario", shocks });
  };
}

function markActive(grid, btn) { grid.querySelectorAll(".scn").forEach(x => x.classList.remove("active")); btn.classList.add("active"); }

function runScenario(sc) {
  const w = weights();
  if (Object.keys(w).length === 0) return;
  const f = propagate(sc.shocks);
  const rows = Object.keys(w).map(t => {
    const r = assetMove(t, f);
    return { t, w: w[t], r, usd: state.amount * w[t] * r, err: ASSETS[t].crisis_error };
  });
  const port = rows.reduce((s, x) => s + x.w * x.r, 0);
  const band = rows.reduce((s, x) => s + x.w * x.err, 0);
  const implied = FACTORS.map((id, i) => ({ id, move: Math.expm1(f[i]), user: id in sc.shocks }));
  state.last = { ...sc, rows, port, band, implied };
  renderResult();
}

function runReplay(key) {
  const w = weights();
  const e = EPISODES[key];
  const rows = Object.keys(w).map(t => ({
    t, w: w[t], r: e.actual[t], pred: e.model_pred_market_only[t],
    usd: state.amount * w[t] * e.actual[t], err: ASSETS[t].crisis_error,
  }));
  const port = rows.reduce((s, x) => s + x.w * x.r, 0);
  const portPred = rows.reduce((s, x) => s + x.w * x.pred, 0);
  const implied = FACTORS.map(id => ({ id, move: e.factor_moves[id], user: false }));
  state.last = { kind: "replay", key, name: e.label, period: `${e.start} → ${e.end}`, rows, port, portPred, implied, shocks: {} };
  renderResult();
}

function rerunLast() {
  const L = state.last;
  if (L.kind === "replay") runReplay(L.key); else runScenario(L);
}

function renderResult() {
  const L = state.last;
  $("result").classList.remove("hidden");
  $("resLabel").textContent = L.kind === "replay" ? `Replay · ${L.name} · ${L.period}` : `${L.name} · your ${usdPlain(state.amount)}`;
  const v = $("resValue");
  v.textContent = pct(L.port);
  v.className = `big-number ${L.port < 0 ? "neg" : "pos"}`;
  $("resSub").textContent = `${usd(state.amount * L.port)} → portfolio worth ${usdPlain(state.amount * (1 + L.port))}`;
  $("resSide").innerHTML = L.kind === "replay"
    ? `<span class="pill">Model check</span><br>Using only pre-2020 data and the S&amp;P 500's real move, the model predicted <b>${pct(L.portPred)}</b> for this portfolio. Reality: <b>${pct(L.port)}</b>. White marks show the model's call for each asset.`
    : `<span class="pill">Uncertainty</span><br>For these assets, the model missed by about <b>±${pp(L.band)} pp</b> on average in past crises. Treat the number as a rough guide, not a forecast.`;

  const max = Math.max(...L.rows.map(x => Math.abs(x.r)), ...(L.kind === "replay" ? L.rows.map(x => Math.abs(x.pred)) : []), 0.01);
  const sorted = [...L.rows].sort((a, b) => a.r - b.r);
  const box = $("impactBars");
  box.innerHTML = `<div class="irow legend"><span>Asset</span><span style="text-align:right">loss</span><span>gain</span><span style="text-align:right">change · $</span></div>`;
  sorted.forEach(x => {
    const width = (Math.abs(x.r) / max) * 100;
    const ghost = L.kind === "replay" ? x.pred : null;
    const ghostLeft = ghost != null && ghost < 0 ? `<span class="ghost-mark" style="right:${(Math.abs(ghost) / max) * 100}%"></span>` : "";
    const ghostRight = ghost != null && ghost >= 0 ? `<span class="ghost-mark" style="left:${(ghost / max) * 100}%"></span>` : "";
    box.insertAdjacentHTML("beforeend", `
      <div class="irow">
        <span class="lbl">${x.t}<small>${(x.w * 100).toFixed(0)}% · ${assetClass(x.t)}</small></span>
        <div class="half left">${x.r < 0 ? `<div class="f" style="width:${width}%"></div>` : ""}${ghostLeft}</div>
        <div class="half right">${x.r >= 0 ? `<div class="f" style="width:${width}%"></div>` : ""}${ghostRight}</div>
        <span class="v ${x.r < 0 ? "neg" : "pos"}">${pct(x.r)}<small>${usd(x.usd)}</small></span>
      </div>`);
  });
  $("impliedChips").innerHTML = L.implied.map(x => `<span class="chip ${x.user ? "user" : ""}">${SHORT[x.id]} ${pct(x.move)}${x.user ? " · your input" : ""}</span>`).join("");
  $("impliedBox").querySelector("summary").textContent = L.kind === "replay" ? "What the market actually did in this period" : "How the model filled in the rest of the market";
  $("reportHint").textContent = "Scenario ready. Write the report to get it explained.";
}

/* ------------------------------------------------------------------ AI report */
function buildFacts() {
  const L = state.last, A = state.analysis;
  const holdings = L.rows.map(x => {
    const a = ASSETS[x.t];
    const coefs = Object.fromEntries(FACTORS.map((f, i) => [SHORT[f], a.coefs[i]]).filter(([, c]) => Math.abs(c) >= 0.1).map(([k, c]) => [k, Math.round(c * 100) / 100]));
    const h = {
      ticker: x.t, name: a.name, type: assetClass(x.t), weight_pct: r1(x.w * 100),
      [L.kind === "replay" ? "actual_change_pct" : "predicted_change_pct"]: r1(x.r * 100),
      change_usd: Math.round(x.usd),
      sensitivity_to_factors: coefs,
      worst_historical_drop_pct: r1(a.max_drawdown * 100),
      typical_yearly_swing_pct: r1(a.annual_vol * 100),
    };
    if (L.kind === "replay") h.model_predicted_change_pct = r1(x.pred * 100);
    return h;
  });
  const facts = {
    portfolio_value_usd: Math.round(state.amount),
    scenario: L.kind === "replay"
      ? { type: "historical replay", name: L.name, period: L.period }
      : { type: L.kind, name: L.name,
          user_shocks_pct: Object.fromEntries(Object.entries(L.shocks).map(([k, v]) => [SHORT[k], r1(v * 100)])),
          model_implied_moves_pct: Object.fromEntries(L.implied.filter(x => !x.user).map(x => [SHORT[x.id], r1(x.move * 100)])) },
    result: {
      portfolio_change_pct: r1(L.port * 100),
      portfolio_change_usd: Math.round(state.amount * L.port),
      value_after_usd: Math.round(state.amount * (1 + L.port)),
    },
    holdings,
    portfolio_stats: {
      number_of_holdings: L.rows.length,
      acts_like_n_equal_holdings: r1(A.effN),
      typical_yearly_swing_pct: r1(A.vol * 100),
      average_correlation: Math.round(A.avgCorr * 100) / 100,
      largest_group: A.topSector[0], largest_group_pct: r1(A.topSector[1] * 100),
    },
    risk_flags: A.flags.map(f => f.replace(/<[^>]+>/g, "")),
  };
  if (L.kind === "replay") facts.result.model_predicted_change_pct = r1(L.portPred * 100);
  else facts.result.typical_model_error_pp = r1(L.band * 100);
  return facts;
}

function buildPrompt(facts, lang) {
  return `You are MarketTwin, a calm, friendly risk explainer for beginner investors (often students).
Write the report in ${lang}.

STRICT RULES
1. Use ONLY numbers that appear in FACTS, written exactly as they appear there (you may drop a minus sign and say "fall" or "loss" instead). Never calculate, combine, round differently or invent any number, price, probability or forecast. If a number you want is not in FACTS, describe it in words.
2. Write percentages with one decimal exactly as in FACTS, like 12.3% (never round to 12%), and dollars like $123, with a dot as the decimal separator and no thousands separators.
3. Never tell the user to buy or sell a specific asset. You may explain general ideas such as diversification or concentration.
4. Explain WHY the most exposed holdings move as they do, using their type and sensitivity_to_factors (a sensitivity of 1.2 to Market means the asset tends to move a bit more than the market).
5. For a historical replay, compare the model's prediction with what actually happened and explain the gap in plain words.
6. At most 230 words. Use exactly these markdown headings (### ...), translated into ${lang}: Summary, Most exposed holdings, Why, Hidden risks, What to think about, How sure is this?
7. In the last section mention the model's typical error (or its miss in the replay) and that this is a prototype, not financial advice.

FACTS
${JSON.stringify(facts, null, 1)}`;
}

function offlineReport(facts) {
  const r = facts.result;
  const hs = [...facts.holdings].sort((a, b) => (a.predicted_change_pct ?? a.actual_change_pct) - (b.predicted_change_pct ?? b.actual_change_pct));
  const key = facts.scenario.type === "historical replay" ? "actual_change_pct" : "predicted_change_pct";
  const worst = hs.slice(0, 2);
  const best = hs[hs.length - 1];
  const dir = r.portfolio_change_pct < 0 ? "lose" : "gain";
  const lines = [
    `### Summary`,
    `In the "${facts.scenario.name}" scenario your $${facts.portfolio_value_usd} portfolio would ${dir} about ${Math.abs(r.portfolio_change_pct)}% ($${Math.abs(r.portfolio_change_usd)}), leaving $${r.value_after_usd}.`,
    `### Most exposed holdings`,
    ...worst.map(h => `- **${h.ticker}** (${h.weight_pct}% of the portfolio): ${h[key]}%, or ${h.change_usd < 0 ? "−" : "+"}$${Math.abs(h.change_usd)}.`),
    `- The most resilient holding is **${best.ticker}** at ${best[key]}%.`,
    `### Why`,
    ...worst.map(h => `- ${h.ticker} is a ${h.type} holding. Its sensitivities are ${Object.entries(h.sensitivity_to_factors).map(([k, v]) => `${k} ${v}`).join(", ") || "small"}, so it reacts strongly to this kind of shock.`),
    `### Hidden risks`,
    `- Your ${facts.portfolio_stats.number_of_holdings} holdings act like ${facts.portfolio_stats.acts_like_n_equal_holdings} equal-sized ones, and ${facts.portfolio_stats.largest_group_pct}% sits in ${facts.portfolio_stats.largest_group}.`,
    `- Average correlation between holdings is ${facts.portfolio_stats.average_correlation}.`,
    `### What to think about`,
    `- Would you be comfortable holding on through a loss like this one? If not, spreading money across groups that react differently to shocks reduces how much one event can hurt.`,
    `### How sure is this?`,
    facts.scenario.type === "historical replay"
      ? `The model predicted ${r.model_predicted_change_pct}% for this period and reality was ${r.portfolio_change_pct}%. This is a prototype and is not financial advice.`
      : `In past crises the model missed by about ${r.typical_model_error_pp} percentage points for assets like these. This is a prototype and is not financial advice.`,
  ];
  return lines.join("\n");
}

async function callGemini(prompt, key, model) {
  for (let attempt = 0; attempt < 2; attempt++) {
    const text = await callGeminiOnce(prompt, key, model);
    // The report has 6 "###" sections. Models sometimes leak planning notes first, so keep only
    // the last run of 6 headings; retry once if the structure is missing.
    const heads = [...text.matchAll(/^#{2,4}\s/gm)].map(m => m.index);
    if (heads.length >= 6) return text.slice(heads[heads.length - 6]);
    if (attempt === 1 && heads.length) return text.slice(heads[0]);
  }
  throw new Error("The model did not return a structured report.");
}

async function callGeminiOnce(prompt, key, model) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;
  const [rules, facts] = prompt.split("\nFACTS\n");
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: rules + "\nOutput only the final report, starting with the first ### heading." }] },
      contents: [{ role: "user", parts: [{ text: "FACTS\n" + facts }] }],
      generationConfig: { temperature: 0.2, maxOutputTokens: 8192 },
    }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data?.error?.message || `HTTP ${res.status}`);
  const cand = data.candidates?.[0];
  const text = (cand?.content?.parts || []).filter(p => !p.thought).map(p => p.text || "").join("");
  if (!text) throw new Error("The model returned an empty answer.");
  return cand.finishReason === "MAX_TOKENS" ? text + "\n\n*(answer cut off by the length limit)*" : text;
}

/* ---- Veritas check: every number in the AI text must exist in the facts ---- */
function factNumbers(obj, out = []) {
  if (typeof obj === "number") out.push(obj);
  else if (typeof obj === "string") (obj.match(/-?\d+(?:\.\d+)?/g) || []).forEach(n => out.push(+n));
  else if (obj && typeof obj === "object") Object.entries(obj).forEach(([k, v]) => { factNumbers(k, out); factNumbers(v, out); });
  return out;
}

const NUM_RE = /([−–-]?)(\$?)(\d+(?:,\d{3})*(?:[.,]\d+)?)(\s?%| pp| percentage points)?/g;

function parseToken(raw) {
  // "1,234" -> thousands; "12,3" -> decimal comma
  let s = raw;
  if (/^\d{1,3}(,\d{3})+$/.test(s)) s = s.replace(/,/g, "");
  else s = s.replace(",", ".");
  const dec = (s.split(".")[1] || "").length;
  return { v: parseFloat(s), dec };
}

function verify(value, dec, facts) {
  const v = Math.abs(value);
  return facts.some(f => {
    const a = Math.abs(f);
    return Math.abs(a - v) < 1e-9 || Math.abs(+a.toFixed(dec) - v) < 1e-9 || (dec === 0 && Math.abs(Math.round(a) - v) < 1e-9);
  });
}

function veritasRender(text, facts) {
  const nums = factNumbers(facts);
  const stats = { ok: 0, bad: 0, neutral: 0, badList: [] };
  let html = escapeHtml(text.replace(/`/g, "")).replace(NUM_RE, (m, sign, dollar, digits, unit, offset, whole) => {
    const prev = whole[offset - 1];
    if (prev && /[A-Za-z]/.test(prev)) return m; // part of a word/ticker like "Q4"
    const { v, dec } = parseToken(digits);
    const plain = !dollar && !unit;
    if (plain && dec === 0 && v >= 1900 && v <= 2100) { stats.neutral++; return m; } // years
    if (verify(v, dec, nums)) { stats.ok++; return `<span class="num ok" title="Found in computed facts">${m}</span>`; }
    if (plain && dec === 0 && v <= 10) { stats.neutral++; return m; } // counts like "3 holdings", headings
    stats.bad++; stats.badList.push(m.trim());
    return `<span class="num bad" title="Not found in computed facts">${m}</span>`;
  });
  html = html
    .replace(/^#{1,4}\s*(.+)$/gm, "<h4>$1</h4>")
    .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
    .replace(/^\s*[-*•]\s+(.+)$/gm, "<li>$1</li>")
    .replace(/(<li>[\s\S]*?<\/li>)(?!\s*<li>)/g, "<ul>$1</ul>")
    .replace(/<\/li>\s*<li>/g, "</li><li>")
    .split(/\n{2,}|\n(?=<h4>)|(?<=<\/h4>)\n/).map(b => b.trim()).filter(Boolean)
    .map(b => /^<(h4|ul)/.test(b) ? b : `<p>${b.replace(/\n/g, "<br>")}</p>`).join("");
  return { html, stats };
}

async function generateReport() {
  if (!state.last) { $("reportHint").textContent = "Run a scenario or a replay in step 02 first."; $("simulate").scrollIntoView(); return; }
  const btn = $("genReport");
  const key = $("apiKey").value.trim();
  const model = $("modelName").value.trim() || "gemini-3.5-flash-lite";
  const lang = $("lang").value;
  localStorage.setItem("mt_key", key);
  localStorage.setItem("mt_model", model);
  const facts = buildFacts();
  $("factsDump").textContent = JSON.stringify(facts, null, 2);
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner"></span>Writing…`;
  let text, source;
  try {
    if (key) { text = await callGemini(buildPrompt(facts, lang), key, model); source = `${model} (Gemini API)`; }
    else { text = offlineReport(facts); source = "offline template (no API key)"; }
  } catch (err) {
    text = offlineReport(facts);
    source = `offline template — Gemini error: ${err.message}`;
  }
  btn.disabled = false;
  btn.textContent = "Write my report";
  const { html, stats } = veritasRender(text, facts);
  $("reportText").innerHTML = html;
  $("veritasBar").innerHTML = `
    <b>Veritas check</b>
    <span class="vbadge good">${stats.ok} numbers verified</span>
    <span class="vbadge ${stats.bad ? "bad" : "neutral"}">${stats.bad} not found in facts${stats.bad ? ": " + stats.badList.map(escapeHtml).join(", ") : ""}</span>
    <span class="vbadge neutral">written by ${escapeHtml(source)}</span>`;
  $("reportOut").classList.remove("hidden");
  $("reportHint").textContent = stats.bad
    ? "Red numbers were not produced by our model. Don't trust them."
    : "Every number in this report matches a value our model computed.";
}

/* ------------------------------------------------------------------ evidence */
function renderEval() {
  const mode = state.evalMode;
  const items = [
    ["mae_ours", "MarketTwin model", "ours"],
    ["mae_ours_no_stress", "Same, no stress weighting", ""],
    ["mae_baseline_capm", "CAPM beta baseline", ""],
    ["mae_baseline_market", "“Moves like market” baseline", ""],
  ];
  const ov = EVAL.overall_mae[mode];
  const overall = { mae_ours: ov.ours, mae_ours_no_stress: ov.ours_no_stress, mae_baseline_capm: ov.baseline_capm, mae_baseline_market: ov.baseline_market };
  const eps = Object.values(EVAL.episodes).map(e => ({ name: e.label, sub: `${e.start} → ${e.end} · S&P ${pct(e.spy_move, 0)}`, m: e.modes[mode] }));
  eps.push({ name: "All four crises", sub: "average over every asset and episode", m: overall, overall: true });
  const max = Math.max(...eps.flatMap(e => items.map(([k]) => e.m[k])));
  $("evalTable").innerHTML = eps.map(e => `
    <div class="ep ${e.overall ? "overall" : ""}">
      <div class="ep-name">${e.name}<small>${e.sub}</small></div>
      <div class="ep-bars">${items.map(([k, n, c]) => `
        <div class="eb ${c}"><span>${n}</span><div class="bar-track"><div class="bar-fill" style="width:${(e.m[k] / max) * 100}%"></div></div><span class="v">${pp(e.m[k])}</span></div>`).join("")}
      </div>
    </div>`).join("");
  const best = Math.min(ov.baseline_capm, ov.baseline_market);
  const bestName = ov.baseline_capm <= ov.baseline_market ? "the CAPM beta" : "the “moves like market” rule";
  const rel = (best - ov.ours) / best;
  let verdict;
  if (rel > 0.03) verdict = `<b>Our model wins here:</b> ${pp(ov.ours)} pp average error versus ${pp(best)} pp for ${bestName}, which is ${(rel * 100).toFixed(0)}% lower.`;
  else if (rel > -0.03) verdict = `<b>Roughly a tie:</b> ${pp(ov.ours)} pp for our model versus ${pp(best)} pp for ${bestName}. With this little information, a simple beta does as well.`;
  else verdict = `<b>The baseline wins here:</b> ${bestName} has ${pp(best)} pp average error versus our ${pp(ov.ours)} pp. Knowing only one sector's move isn't enough for our model to add value.`;
  const sw = ov.ours_no_stress - ov.ours;
  verdict += ` Weighting stress weeks ${sw > 0 ? `lowers the error by ${pp(sw)} pp` : "does not help"} in this mode.`;
  if (mode === "market_only") verdict += ` On 2,000 random 4-asset portfolios, portfolio-level error is ${pp(EVAL.portfolio_mae_market_only.ours)} pp for our model versus ${pp(EVAL.portfolio_mae_market_only.baseline_capm)} pp for CAPM.`;
  $("evalVerdict").innerHTML = verdict;
}

function pairBars(entries) {
  const max = Math.max(...entries.flatMap(([, v]) => [v.mae_ours, v.mae_capm]));
  return entries.map(([name, v]) => `
    <div class="pair">
      <div class="pair-name">${name}${v.n_assets ? ` <small>· ${v.n_assets} asset${v.n_assets > 1 ? "s" : ""}</small>` : ""}</div>
      <div class="eb ours"><span>Our model</span><div class="bar-track"><div class="bar-fill" style="width:${(v.mae_ours / max) * 100}%"></div></div><span class="v">${pp(v.mae_ours)}</span></div>
      <div class="eb"><span>CAPM beta</span><div class="bar-track"><div class="bar-fill" style="width:${(v.mae_capm / max) * 100}%"></div></div><span class="v">${pp(v.mae_capm)}</span></div>
    </div>`).join("");
}

function renderGroups() {
  const g = Object.entries(EVAL.group_error.market_only).sort((a, b) => a[1].mae_ours - b[1].mae_ours);
  const p = Object.entries(EVAL.preset_portfolio_error).sort((a, b) => a[1].mae_ours - b[1].mae_ours);
  $("groupBars").innerHTML = pairBars(g);
  $("presetBars").innerHTML = pairBars(p);
  const [bestP, worstP] = [p[0], p[p.length - 1]];
  $("groupVerdict").innerHTML = `<b>The tool is not equally reliable for everyone.</b> It is most accurate for a <b>${bestP[0]}</b> portfolio (${pp(bestP[1].mae_ours)} pp error) and least accurate for <b>${worstP[0]}</b> (${pp(worstP[1].mae_ours)} pp), mostly because of Bitcoin. That is the riskiest kind of beginner, and the one who most needs a good warning. The app says so whenever Bitcoin is in a portfolio.`;
}

function renderAudit() {
  const a = window.MT_AUDIT;
  if (!a) return;
  const f = a.false_accept_pct;
  $("auditMain").innerHTML = `We generated <b>${a.reports} reports</b> with ${a.model}: 4 example portfolios × 3 scenarios × ${a.languages.join(", ")}. They contained <b>${a.numbers_verified + a.numbers_unverified} numbers</b>, and <b>${a.numbers_unverified}</b> of them were not found in the computed facts. All ${a.structured} reports followed the required structure. Earlier, with a larger model, we did see a cut-off and garbled answer. The check flagged it, and that is why the app now uses the smaller, faster model.`;
  $("auditLeak").innerHTML = `The check only matches numbers, so a made-up number can pass by coincidence. We tested this by inserting random fake numbers into reports. A fake dollar amount passes <b>${f.dollars}%</b> of the time, a fake one-decimal percentage <b>${f.one_decimal_percent}%</b>, but a fake <b>whole-number percentage ${f.whole_percent}%</b> of the time, because so many facts round to the same integers. Our prompt therefore asks for one-decimal percentages. Matching each number to the fact it should describe is our next step.`;
}

function renderFails() {
  $("fails").innerHTML = EVAL.failures.slice(0, 6).map(f => `
    <div class="fail"><div class="ft">${f.ticker} <span class="fe">· ${ASSETS[f.ticker]?.name || ""}</span></div>
      <div class="fe">${f.episode}</div>
      <div class="fx"><span>model ${pct(f.predicted)}</span><span>real ${pct(f.actual)}</span></div></div>`).join("");
}

/* ------------------------------------------------------------------ init */
function init() {
  renderHero();
  renderPresets();
  renderScenarios();
  renderEval();
  renderFails();
  $("addAsset").onchange = e => {
    const t = e.target.value;
    if (t && Object.keys(state.holdings).length < 6) { state.holdings[t] = 20; clearPresetHighlight(); refresh(); }
  };
  $("amount").oninput = e => { state.amount = Math.max(+e.target.value || 0, 0); if (state.last) rerunLast(); };
  document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.toggle("active", x === t));
    ["preset", "custom", "replay"].forEach(id => $(`tab-${id}`).classList.toggle("hidden", id !== t.dataset.tab));
  });
  document.querySelectorAll("#modeSwitch .chip-btn").forEach(b => b.onclick = () => {
    state.evalMode = b.dataset.mode;
    document.querySelectorAll("#modeSwitch .chip-btn").forEach(x => x.classList.toggle("active", x === b));
    renderEval();
  });
  $("genReport").onclick = generateReport;
  $("apiKey").value = localStorage.getItem("mt_key") || "";
  $("modelName").value = "gemini-3.5-flash-lite";
  renderGroups();
  renderAudit();
  loadPortfolio(PORTFOLIOS[0]);
}

init();
