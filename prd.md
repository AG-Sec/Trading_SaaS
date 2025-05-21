# Product Requirements Document (PRD) v1.1 – Modular Trading Signal Platform

**Project Name:** Modular Trading Signal Platform
**Owner:** Avery M.
**Tech Lead:** ChatGPT (Guided AI Advisor)
**Target Users:** Retail traders (crypto, forex, equities) who want clear, timely buy/sell signals without coding.
**Updated:** 2025‑05‑17

---

## 1 – Objective

Build a modular, AI‑augmented platform that produces high‑quality entry/exit signals and delivers them to users in real time. Architecture follows a multi‑agent pattern (Google A2A‑compatible) with swappable data & notification adapters. Long‑term goal: profitable SaaS with ≥ \$1 k MRR inside 90 days.

---

## 2 – Personas & Jobs‑to‑Be‑Done

| Persona               | JTBD                                                       | Key Needs                                     |
| --------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| **Day‑Trader Dan**    | “Catch intraday breakouts on BTC & ETH.”                   | Mobile push, <1 s latency, paper‑trade toggle |
| **Swing‑Trader Sam**  | “Get daily signals on SPY & EURUSD and weekly P/L recaps.” | E‑mail digest, backtests, risk metrics        |
| **Algorithmic‑Annie** | “Plug raw signals into my own scripts.”                    | REST API, WebSocket feed, JSON schemas        |

---

## 3 – Core Features

### A. Signal Generation (MVP)

* 20‑period high/low breakout strategy
* Assets: BTC‑USD, ETH‑USD, EURUSD=X, SPY
* Timeframes: 15 min, 1 h, 4 h, 1 d (configurable)

### B. Agent Modules

1. **MarketDataAgent** – pulls OHLCV (yfinance or exchange)
2. **SignalScannerAgent** – detect breakout events
3. **RiskManagerAgent** – filter by ATR, volatility, slippage
4. **ExecutionAgent** – paper/live trades via Alpaca, Kraken, Crypto.com
5. **JournalAgent** – write signals/trades to SQLite ➜ PostgreSQL
6. **NotificationService** – calls `notify(channel, payload)` (see § D)
7. **DashboardAgent** – REST/WS API for Next.js UI
8. **Pattern & ML Agents** – **Phase 3** stretch (pattern matching, predictive ML)

### C. Backtesting (Week 2)

* Replay strategy on historical data ➜ hit rate, win %, avg R

### D. Alerts (Week 1)

* **NotificationAdapter Interface** – unified `notify(channel, payload)`
* **MVP Channels**
  • **Telegram bot** – instant push (≈30 msg/s limit)
  • **E‑mail** – daily/weekly digests via SES/Postmark
* Future adapters: Discord, SMS, mobile push, Slack

### E. User Accounts & Subscription (Week 3‑4)

* Firebase/Auth → free vs pro tiers
* Stripe recurring billing; quota tracked per plan

---

## 4 – Non‑Functional & Compliance

* **Latency:** tick → signal < 500 ms (95‑percentile)
* **Uptime SLO:** API 99.5 %, UI 99 %
* **Security:** .env secrets ➜ Hashicorp Vault by GA launch; PAM‑less SSH
* **Regulation:** U.S. CFTC/SEC; no leveraged derivatives for TX retail

---

## 5 – Architecture Overview

```mermaid
sequenceDiagram
  MarketDataAgent->>SignalScanner: OHLCV stream
  SignalScanner->>RiskManager: breakout_event
  RiskManager-->>JournalAgent: accepted_signal
  RiskManager->>NotificationService: notify(['telegram','email'], payload)
  RiskManager->>ExecutionAgent: trade_intent
  ExecutionAgent-->>JournalAgent: trade_record
  DashboardAgent<-.-JournalAgent: WebSocket updates
```

**Data Contract (v0.1):**

```json
{
  "signal_id": "uuid4",
  "asset": "BTC-USD",
  "timeframe": "1h",
  "direction": "long|short",
  "entry": 64650.25,
  "stop": 63200.0,
  "take_profit": 67200.0,
  "timestamp": "2025-05-17T15:44:00Z"
}
```

---

## 6 – Tech Stack

| Layer    | Tooling                                  | Notes                |
| -------- | ---------------------------------------- | -------------------- |
| Backend  | Python 3.12, FastAPI, pydantic‑v2        | Async, typed         |
| Data     | Pandas, yfinance, TA‑Lib                 | TA calculations      |
| Storage  | SQLite ➜ Supabase PG                     | E‑Z deployment       |
| Frontend | Next.js 14 (App Router), Tailwind/shadcn | SSR + API routes     |
| CI/CD    | GitHub Actions                           | lint → test → deploy |
| Hosting  | Render.com MVP ➜ Fly.io                  | Free tier dev        |

---

## 7 – Milestones (Accelerated)

| Week  | Deliverables                                                                                                | Acceptance Criteria                                                        |
| ----- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **1** | Repo scaffold; MarketDataAgent, SignalScanner, RiskManager; NotificationAdapter w/ Telegram+Email; CLI demo | Unit tests ≥ 90 % pass; CLI prints JSON & sends Telegram message under 1 s |
| **2** | Backtesting engine; SQLite journal; Discord adapter stub; CI pipeline                                       | Backtest CLI shows metrics; PR merges must green‑light CI                  |
| **3** | Next.js dashboard (chart, signal feed); Auth; Stripe test mode; ExecutionAgent paper‑trade                  | Login→dashboard in <700 ms; paper order shows ✅ badge                      |
| **4** | Prod deploy; beta user onboarding; mobile‑friendly UI; basic analytics                                      | 10 beta users receive alerts; Mixpanel funnel shows ≥40 % activation       |

---

## 8 – Success Metrics (90‑day target)

* >  70 % 14‑day retention
* 60 % of signals profitable (after fees) over rolling 30 d
* 1 k USD MRR
* Avg tick→alert latency < 500 ms

---

## 9 – Open Questions & Owners

| Question                                      | Decision by        | Owner     |
| --------------------------------------------- | ------------------ | --------- |
| **Primary broker for U.S. equities – Alpaca** | decided 2025‑05‑17 | Avery     |
| Discord vs SMS next?                          | 2025‑05‑26         | ChatGPT   |
| Vault or Doppler for secrets?                 | 2025‑05‑30         | Tech Lead |

---

## 10 – Risk Log

1. **API Rate Limits** – mitigate via rotating keys & caching
2. **Exchange Downtime** – failover data sources (yfinance → Polygon)
3. **Strategy Overfitting** – track live vs backtest delta
4. **Regulatory Shift (TX)** – subscribe to FinCEN feeds

---

## 11 – Developer Onboarding & Windsurf IDE Notes

### A. Repo Skeleton (monorepo)

```
/agents/         # MarketData, SignalScanner, RiskManager, ExecutionAgent
/api/            # FastAPI routers & pydantic schemas
/frontend/       # Next.js 14 app
/tests/          # pytest + httpx async
/scripts/        # one‑off maintenance, DB migrations
infra/
  docker/        # Dockerfiles & docker‑compose
  github/        # GitHub Actions workflows
  terraform/     # optional IaC for Fly.io/Render
requirements/    # pyproject.toml + lock
README.md        # high‑level dev doc
.windsurf.yml    # IDE tasks & context
.env.sample      # required secrets template
```

### B. Local Dev Environment

1. **Python 3.12** (use `pyenv` or asdf).
2. `poetry install` – installs backend deps.
3. `npm i` inside `/frontend` – installs UI deps.
4. Copy `.env.sample` → `.env` and fill:

```
ALPACA_KEY=…
ALPACA_SECRET=…
TELEGRAM_BOT_TOKEN=…
POSTMARK_TOKEN=…
SUPABASE_URL=…
SUPABASE_KEY=…
```

5. `docker compose up` – optional: launches Postgres & local stack.
6. Run dev stack:

```
poe dev      # runs FastAPI with reload, and Celery worker
npm run dev  # runs Next.js 14 with Turbo
```

### C. Windsurf IDE Configuration (`.windsurf.yml`)

```yaml
version: 1
context:
  files:
    - prd.md
    - agents/*.py
    - api/*.py
  model: "gpt-4o"   # default LLM
prompts:
  - name: "GenerateAgent"
    pattern: "agent:([A-Za-z]+)"
    task: |
      Create the {{1}}Agent class in agents/{{1|lower}}.py using pydantic‑v2 models …
  - name: "FixTests"
    pattern: "tests failing"
    task: "Refactor code until pytest passes"
commands:
  dev: "poetry run uvicorn api.main:app --reload"
```

*This lets Windsurf watch the PRD + codebase and auto‑suggest agent stubs or fixes.*

### D. Pre‑commit Hooks

```
pre‑commit install
```

* Runs `ruff`, `mypy`, `black` on every commit. CI will reject non‑formatted PRs.\*

### E. GitHub Actions (`infra/github/ci.yml`)

* Matrix: {os: ubuntu‑latest, python: 3.12}
* Steps: checkout → set‑up‑python → poetry install → `pytest -q` → build Docker image → deploy preview.

### F. CI Secrets (Render → GitHub OIDC)

* `RENDER_DEPLOY_TOKEN` – auto‑deploy `main` to staging.

### G. Suggested Iteration Flow

1. `git checkout -b feature/market-data-agent`
2. Describe task in Windsurf chat → auto‑generate stub
3. Flesh out logic, add tests
4. `pytest` local → commit → PR
5. ChatGPT code‑review → merge when green.

---

### 12 – Definition of Ready (for stories)

* Story is linked to PRD section.
* Acceptance criteria written & testable.
* Env vars & data mocks prepared.
* No external blocker flagged.

> **Next Step for Devs:**
>
> 1. Clone repo skeleton (coming shortly).
> 2. Create `.env` with paper Alpaca creds.
> 3. Run `poe dev`.
> 4. Ask Windsurf: `agent:MarketData` to scaffold the first agent.

Happy coding! 🚀
