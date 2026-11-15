async page => page.evaluate(() => {
  let seed = 7; const rand = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const res = { pct1: [0, 0], pct0: [0, 0], usd: [0, 0] };
  for (const p of PORTFOLIOS) for (const s of SCENARIOS) {
    loadPortfolio(p); runScenario({ kind: "scenario", name: s.name, shocks: s.shocks });
    const facts = buildFacts();
    for (let i = 0; i < 200; i++) {
      const cases = { pct1: `${(rand() * 60).toFixed(1)}%`, pct0: `${Math.round(11 + rand() * 50)}%`, usd: `$${Math.round(11 + rand() * 900)}` };
      for (const [k, txt] of Object.entries(cases)) {
        const { stats } = veritasRender(`Value ${txt} here.`, facts);
        res[k][0] += stats.ok; res[k][1] += 1;
      }
    }
  }
  return Object.fromEntries(Object.entries(res).map(([k, [a, n]]) => [k, +(a / n * 100).toFixed(1)]));
})
