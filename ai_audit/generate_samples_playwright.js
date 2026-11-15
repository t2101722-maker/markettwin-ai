async page => page.evaluate(async () => {
  const key = document.getElementById('apiKey').value;
  const jobs = [
    ["Tech fan", "Tech sell-off", "English"],
    ["Tech fan", "Tech sell-off", "Russian"],
    ["Tech fan", "Tech sell-off", "Kazakh"],
    ["Crypto + growth", "Rate shock (2022-style)", "English"],
    ["Balanced starter", "Market crash", "English"],
  ];
  const out = [];
  for (const [pn, sn, lang] of jobs) {
    const p = PORTFOLIOS.find(x => x.name === pn);
    const s = SCENARIOS.find(x => x.name === sn);
    loadPortfolio(p);
    runScenario({ kind: "scenario", name: s.name, shocks: s.shocks });
    const text = await callGemini(buildPrompt(buildFacts(), lang), key, "gemini-3.5-flash-lite");
    out.push({ portfolio: pn, scenario: sn, lang, model: "gemini-3.5-flash-lite", date: "2026-11-15", text });
    await new Promise(r => setTimeout(r, 4500));
  }
  return out;
})
