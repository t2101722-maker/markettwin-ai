# MarketTwin AI — See the risk before the market moves

**Track:** AI + Finance
**Team:** Temirlan Amanzholov, Camilla Nessipbekova, Aiaru Melikhan
**Live demo:** https://t2101722-maker.github.io/markettwin-ai/

> This project is a prototype and is not financial advice.

## The problem
Beginner investors, such as students who just opened their first brokerage account, often hold 3–6 popular stocks that are mostly tech. They see each stock's price but not how the stocks react *together* when the market turns. They find out how concentrated they are during a crash, which is the worst moment to learn it.

## What we built
A website where you enter your 3–6 holdings and how much money is in each, then:

1. **Portfolio analyzer**: shows how often *all* your holdings fell together in the worst 20% of S&P 500 weeks since 2010, your average move in such a week, your estimated loss if the market falls 10%, your typical yearly swing, your sector exposure, and plain-language risk flags. We started with calm-period correlation, but it badly misleads in a stress test. Over the last 3 years Apple and Microsoft were only 0.18 correlated week to week, versus 0.77 in 2020–2022. Correlation measured only on crash weeks is biased low by the selection. So the headline metric is a simple co-crash rate instead: 47% of bad weeks for the "Tech fan" example versus 12% for a balanced portfolio.

**Example user:** a 17-year-old in Almaty who put $1,000 of summer-job savings into five US tech stocks and asks "what happens to my money if tech drops 20%?" before buying more.
2. **What-if simulator**: pick a scenario ("tech falls 20%", "2022-style rate shock", …), build your own from 11 market factors, or **replay a real crash** (COVID-19, 2022, Aug 2024, Apr 2025) on today's portfolio. Results come asset by asset, in % and in dollars, with the model's typical error.
3. **AI risk report**: Gemini explains the result in English, Russian or Kazakh. The language model gets only a JSON of computed facts. Our **Veritas check** then scans every number in its answer and flags any number our model did not produce. If there is no API key, a deterministic template report is used instead. Judges without a key can also open five real Gemini reports saved during the hackathon (English, Russian, Kazakh). The Veritas check re-runs on them live.

## What we added
Our own idea. On top of the simulator, we added:
- a time-split test of the scenario model on four real crises it never saw;
- a replay mode that shows the model's pre-2020 prediction next to what actually happened;
- the Veritas check against AI-invented numbers;
- reports in Kazakh and Russian for beginner investors in Kazakhstan.

## How it works
**Data:** daily adjusted prices from Yahoo Finance (via `yfinance`), Jan 2010 – Nov 2026, for 31 assets (26 large US stocks, SPY, QQQ, TLT, GLD, Bitcoin) and 11 factor ETFs (S&P 500, nine sector SPDRs, long Treasuries, gold). The prices are converted to weekly log returns.

**Model (`pipeline/model.py`):** for each asset, a **ridge regression** of weekly returns on the market, the asset's sector (as excess over the market), long bonds and gold. **Stress weighting:** the worst 20% of market weeks get extra weight, because assets behave differently in crashes. A scenario usually fixes only one or two factors. The remaining factors are filled in with their **conditional expectation** under the factor covariance measured in past stress weeks: E[F_rest | F_S = s] = Σ_rS Σ_SS⁻¹ s.

**Tuning:** train 2010–2017, choose the ridge strength and stress weight on two 2018 sell-offs. Best: α = 0.01, stress weeks weighted 4×.

**Test:** train 2010–2019, then predict each asset's return in four later crises, given the factor moves. Baselines: (1) "every asset moves like the market"; (2) the textbook CAPM beta (OLS on all weeks).

## Results
Mean absolute error per asset, in percentage points, on 31 assets × 4 unseen crises:

| Scenario information | Our model | No stress weighting | CAPM beta | "Moves like market" |
|---|---|---|---|---|
| All sector moves known* | **8.7** | 8.9 | 10.5 | 11.0 |
| Only the market move known | **10.0** | 10.2 | 10.1 | 11.0 |
| Only the tech-sector move known | 10.5 | 10.7 | **10.4** | 11.7 |

\* SPY, TLT and GLD are left out of this row because they are model inputs in this mode. We caught this ourselves: including them made our model look better (7.9) than it really is.

- When a scenario says how sectors move, the model is **17% more accurate than CAPM** on average. It is not better in every crisis: in the COVID crash, the simple "moves like the market" rule did better (9.7 vs 10.1). When only the market move is known, it **ties** with CAPM. With only the tech sector's move, **CAPM is slightly better**. That is why our preset scenarios describe several factors.
- Stress weighting helps a little but consistently (≈0.2 pp in every mode).
- For random 4-asset portfolios in the market-only mode, our model and CAPM are tied (6.3 pp each).

**Who it works for.** Portfolio-level error per example investor (market-only mode):

| Example investor | Our model | CAPM |
|---|---|---|
| Balanced starter | 4.0 | 3.6 |
| Tech fan | 6.4 | 6.2 |
| Dividends & defensives | 8.5 | 8.5 |
| Crypto + growth | 11.2 | 11.6 |

The tool is least reliable for the riskiest beginner (crypto + growth), mainly because of Bitcoin. For a balanced portfolio, the simple CAPM beta does slightly better than our model. The app shows a Bitcoin warning whenever Bitcoin is in a portfolio.

**Auditing the AI writer** (`ai_audit/`). We generated 36 reports with Gemini 3.5 Flash-Lite: 4 portfolios × 3 scenarios × English, Russian and Kazakh. Out of 478 numbers, **0** were missing from the computed facts, and all 36 reports followed the required structure. The check matches numbers only, so we also measured how often a *random fake* number passes it by coincidence: dollars 2.6%, one-decimal percentages 6.3%, **whole-number percentages 34.1%**. That is why the prompt asks for one-decimal percentages. (We tightened that wording in the prompt after the audit ran.) A larger model (Gemini 3.5 Flash) was slow (20–33 s) and once returned a cut-off, garbled answer. The Veritas check flagged it, and we switched models.

**Where it fails:**
- **2022.** The model learned from 2010–2019 that bonds rise when stocks fall. In 2022 both fell: TLT −29% vs +15% predicted. Oil stocks rose 60% (XOM) while the model predicted −26%.
- **Company-specific news.** Meta −62% in 2022 and Intel −41% in Aug 2024 came from company news that no factor model sees.
- **Bitcoin.** Stock factors barely explain it: −59% actual vs −10% predicted in 2022.

All numbers come from `pipeline/train_and_evaluate.py`. The full log is in `pipeline/data/train_log.txt`.

## What we'd do next
- Learn how factor relationships change between regimes, for example with rolling or regime-switching covariance, so the 2022 bond–stock break is no longer a blind spot.
- Make the Veritas check match each number to the specific fact it describes, not just to any fact.
- Add Kazakhstan-specific assets (e.g. Kaspi.kz, KASE index) and the USD/KZT exchange rate. Then test the tool with real first-time investors.

## Run it
```bash
cd pipeline
pip install -r requirements.txt
python download_data.py          # prices -> pipeline/data/prices.csv
python train_and_evaluate.py     # tune, test, export -> web/data/
# open web/index.html in a browser (works from disk, no server needed)
# for the AI report: paste a free Gemini API key in the page (stored only in your browser)
```

## Sources
- Price data: Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance) (unofficial, for learning)
- Libraries: pandas, NumPy, scikit-learn (Ridge), SciPy
- Language model: Google Gemini API ([Google AI Studio](https://aistudio.google.com/)), free tier
- Fonts: Instrument Serif, Geist (Google Fonts)
- AI tools used while building: Claude Code (coding assistant). Gemini (in-app report generation).
