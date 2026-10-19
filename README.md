<div align="center">

# PrismAI

### Multi-Agent AI Orchestration Platform

**[Live Demo](https://prism-ai-rust.vercel.app/) · [Repository](https://github.com/karthik-0306/PrismAI)**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-1.89-FF6B6B?style=for-the-badge)](https://litellm.ai)
[![Render](https://img.shields.io/badge/Backend-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com)
[![Vercel](https://img.shields.io/badge/Frontend-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)

*An intelligent, production-grade AI platform that dynamically routes every user query to the most suitable specialized agent, achieving a measured **16.92% average reduction in token overhead** without degrading output quality.*

</div>

---

## System Architecture

Every query flows through a five-stage pipeline before a response is returned to the user:

```mermaid
flowchart TD
    U([User]) -->|Raw Query| RW

    subgraph PREPROCESSING["Preprocessing Layer"]
        RW["Query Rewriter\n(Qwen 3.8 27B, utility chain)\nStrips filler words\nProtects code blocks\nSemantic-similarity guard"]
    end

    RW -->|Compressed Query| SR

    subgraph ROUTING["Smart Router"]
        SR["SmartRouter\n(Qwen 3.8 27B, utility chain)\nClassifies intent with chat context\nResolves follow-ups\nSplits compound queries"]
    end

    SR -->|dsa| DSA
    SR -->|web_search| WS
    SR -->|evaluate| EV
    SR -->|coding / math / reasoning| LLM
    SR -->|summarize / fast / general| LLM

    subgraph AGENTS["Specialized Agents"]
        DSA["DSA Subagent\nAlgorithm Expert\nComplexity Analysis\nStep-by-Step Solutions"]
        WS["WebSearch Subagent\nTavily API — Real-time Data\nAuto Query Broadening\nSource-Cited Synthesis"]
        EV["Evaluator Subagent\nAI Judge Panel\nFactuality · Relevance\nGroundedness Scoring"]
        LLM["Orchestrator LLM\nGPT-OSS 120B / Qwen 3.8 27B\nGPT-OSS 20B / Gemini Flash\n(via LiteLLM, health-aware fallback)"]
    end

    DSA --> JD
    WS --> JD
    EV --> JD
    LLM --> JD

    subgraph QUALITY["Quality Control"]
        JD["Quality Judge\nSeverity Scoring 1-3\nAuto-Retry on Severity 3"]
    end

    JD -->|Compound Query| AGG
    JD -->|Single Query| STREAM

    subgraph FINALIZE["Response"]
        AGG["Aggregator\nMerges parallel sub-responses"]
        STREAM["SSE Streaming\nToken-by-token to client"]
    end

    AGG --> STREAM
    STREAM --> DB
    STREAM --> U

    subgraph PERSIST["Persistence"]
        DB[("SQLite DB\nMessages / Chats\nToken Metrics\nRoute Analytics")]
    end
```

---

## Features

### Intelligent Multi-Agent Routing

PrismAI's `SmartRouter` classifies every query into one of 9 categories using a lightweight model, then dispatches it to the most capable agent. The router sees a snippet of the conversation, so follow-ups like *"what is its time complexity?"* are rewritten into self-contained tasks and keep the right specialist. Compound queries (e.g. *"explain binary search AND what is quantum computing?"*) are split, answered in parallel, and the merged answer is streamed. Every specialist receives the conversation history.

All model ids live in [`backend/config.py`](backend/config.py) and can be overridden with env vars (`PRISMAI_MODEL_SMART`, `PRISMAI_MODEL_CODER`, `PRISMAI_MODEL_FAST`, `PRISMAI_MODEL_GEMINI`) when a provider renames or retires a model.

| Route | Agent / Handler | Best For |
|---|---|---|
| `dsa` | DSA Subagent | Algorithms, LeetCode, complexity analysis |
| `web_search` | WebSearch Subagent | Real-time info, current events, prices |
| `evaluate` | Evaluator Subagent | Scoring AI responses, factuality checks |
| `coding` | Qwen 3.8 27B | Debugging, refactoring, code generation |
| `math` | GPT-OSS 120B | Proofs, calculus, numerical computation |
| `reasoning` | GPT-OSS 120B | Logic puzzles, argument analysis |
| `summarize` | Qwen 3.8 27B | Condensing documents and text |
| `fast` | GPT-OSS 20B | Trivial one-liner answers |
| `general` | Qwen 3.8 27B | Everything else |

---

### Query Rewriter — Prompt Compression

A dedicated preprocessing step compresses verbose user prompts before they reach expensive models, acting as a transparent cost-reduction layer:

- Strips conversational filler (*"Hey, could you please help me with..."*)
- Restructures intent for model clarity
- Strictly preserves code blocks, raw data, and document payloads
- Measures and logs token savings to the database in real time
- **Achieved 16.92% average token reduction** across a 100-query evaluation dataset with the v1 prompt (the prompt has since been shortened — re-run `backend/evaluation/test_rewriter_quality.py` to re-measure)
- Only runs where it can pay for itself (40–600 token queries); rejects rewrites that drop negations, numbers or code, over-compress (<25% kept), or fall below the embedding-similarity threshold
- Displays a per-message savings chip in the UI showing the reduction percentage

---

### Real-Time Web Search Agent

A fully async subagent powered by the **Tavily Search API**:

- Fetches clean, pre-extracted content — no raw HTML scraping
- **Automatic query broadening** — if a hyper-local query returns sparse results, it retries with a broader query automatically
- Follow-ups (*"and what about Google?"*) are searched as complete questions thanks to the context-aware router
- Answers cite numbered sources inline; the UI renders the sources as clickable chips
- If search is unavailable, the model answers from general knowledge and says so

---

### DSA Subagent

A specialized algorithm expert with deep reasoning prompts tuned for:

- Algorithm explanation and correctness proofs
- Time/space complexity analysis
- Structured step-by-step LeetCode-style solutions for full problems, direct answers for conceptual follow-ups
- Conversation-aware: *"now write it in Java"* knows what "it" is

---

### AI Evaluator and Quality Judge

A two-tier quality control system:

1. **Evaluator Subagent** — a panel of three different models scoring a response on seven metrics; *"evaluate your last answer"* scores the previous turn
2. **Auto-Retry** — in compound queries, sub-answers judged Severity 3 (poor quality) are retried once before aggregation

---

### Token-by-Token SSE Streaming

- Real-time streaming via **Server-Sent Events** for every query type, including compound ones
- Live progress events (*"Searching the web…"*, *"Combining the answers…"*)
- **Stop** keeps the partial answer; **Regenerate** / **Retry** replace the last turn
- If a model fails mid-answer, a `reset` event clears the partial text and a fallback model restarts cleanly

### Reliability on Free-Tier APIs

- Per-model health registry: dead keys / retired models are skipped for 10 minutes, rate-limited models until the provider's retry-after
- Groq counts the *requested* `max_tokens` against its per-minute budget, so on a token rate limit the client shrinks `max_tokens` to what still fits instead of failing
- When every model is busy, the request waits briefly for the soonest one rather than erroring
- Utility work (routing, rewriting, titles) runs on a different model than answers to spread load across per-model quotas

---

### Analytics Dashboard

An interactive built-in dashboard (powered by **Recharts**) showing:

- Total token usage and tokens saved by the Query Rewriter over time
- Agent routing distribution across intent categories
- Model usage breakdown
- Chat volume trends

---

### Chat History and Session Management

- `localStorage`-based UUID session management — no login required, every visitor gets a private isolated session
- Every history endpoint is scoped to the caller's session (another session's chat ids return 404)
- Token-budgeted memory with rolling summaries for long conversations
- LLM-generated chat titles, rename, keyword search, one-click deletion

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | React 18 + Vite | UI framework |
| **Styling** | CSS Modules | Scoped, zero-runtime styles |
| **Charts** | Recharts | Analytics dashboard |
| **Backend** | FastAPI (Python 3.11) | API server, SSE streaming |
| **LLM Gateway** | LiteLLM | Universal multi-provider LLM abstraction |
| **LLM Providers** | Groq, Gemini | Model inference |
| **Web Search** | Tavily API | AI-optimised real-time search |
| **Embeddings** | Gemini embedding API (optional local sentence-transformers) | Rewriter meaning-preservation check |
| **Database** | SQLite + aiosqlite | Async persistent storage |
| **Token Counter** | tiktoken | Per-message token accounting |
| **Frontend Host** | Vercel | Static hosting |
| **Backend Host** | Render | Python web service |

---

## Project Structure

```
PrismAI/
│
├── backend/
│   ├── main.py                     # FastAPI app, CORS, lifespan
│   ├── config.py                   # Model ids, chains, limits (env-overridable)
│   ├── tests/                      # Offline pytest suite (fake LLM, temp DB)
│   ├── pipeline/
│   │   ├── orchestrator.py         # Core routing, dispatch, aggregation
│   │   ├── router.py               # SmartRouter (intent classification)
│   │   ├── rewriter.py             # Query compression agent
│   │   └── memory.py               # Conversation memory injection
│   ├── subagents/
│   │   ├── dsa_agent.py            # Algorithm specialist
│   │   ├── web_search_agent.py     # Tavily-powered search agent
│   │   └── evaluator_agent.py      # AI judge panel
│   ├── llm/
│   │   ├── client.py               # LiteLLM abstraction (completions + streaming)
│   │   └── embeddings.py           # Gemini (or local) embeddings for the rewriter
│   ├── database/
│   │   ├── connection.py           # aiosqlite connection + schema init
│   │   ├── models.py               # Typed dataclasses (Chat, Message)
│   │   └── queries.py              # All SQL — parameterised, async
│   ├── routers/
│   │   ├── chat.py                 # POST /api/chat, GET /api/models
│   │   ├── streaming.py            # POST /api/chat/stream (SSE)
│   │   ├── history.py              # GET/PATCH/DELETE /api/chats (session-scoped)
│   │   └── metrics.py              # GET /api/metrics, GET /api/model-status
│   ├── utils/
│   │   ├── session.py              # UUID validation
│   │   └── token_counter.py        # tiktoken wrapper
│   └── evaluation/
│       ├── test_rewriter_quality.py    # Query rewriter benchmark
│       ├── rewriter_eval_results.csv   # Evaluation results (100-query dataset)
│       └── analyze_failures.py         # Failure analysis utilities
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                 # Root component + global state
│   │   ├── index.css               # Global design system / CSS variables
│   │   ├── api/
│   │   │   ├── chat.js             # HTTP/SSE calls to backend
│   │   │   └── metrics.js          # Analytics and model-status calls
│   │   ├── hooks/
│   │   │   ├── useSession.js       # localStorage UUID session hook
│   │   │   └── useBackendStatus.js # Backend health polling hook
│   │   └── components/
│   │       ├── Sidebar.jsx             # Nav, model selector, chat history
│   │       ├── ChatArea.jsx            # Message list + input bar
│   │       ├── MessageBubble.jsx       # Renders messages, badges, savings chip
│   │       ├── AnalyticsDashboard.jsx  # Recharts analytics view
│   │       └── ServerStatusBanner.jsx  # Backend cold-start status banner
│   └── vercel.json                 # SPA routing config
│
├── render.yaml                     # Render deployment config
├── requirements.txt                # Python dependencies
└── .python-version                 # Pins Python 3.11 for Render
```

---

## Running Locally

### Prerequisites

- Python **3.11**
- Node.js **18+**
- API keys for [Groq](https://console.groq.com), [Gemini](https://aistudio.google.com), and [Tavily](https://app.tavily.com)

### 1. Clone the repository

```bash
git clone https://github.com/karthik-0306/PrismAI.git
cd PrismAI
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

### 3. Start the backend

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn backend.main:app --reload --port 8000
```

> The backend will be available at `http://localhost:8000`

> **Note:** The rewriter verifies each compression with Gemini embeddings (`gemini-embedding-001`), plus deterministic checks that negations, numbers and code blocks survive. For fully offline embeddings, `pip install sentence-transformers` and set `PRISMAI_EMBEDDINGS=local` (needs ~300 MB+ RAM — not suitable for Render's free tier).

### Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest backend/tests -q
```

The suite runs fully offline against a fake LLM and a temporary database.

### 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

> The frontend will be available at `http://localhost:5173`. Vite proxies all `/api/*` requests to the FastAPI backend automatically.

---

## Deployment

PrismAI uses a split-stack deployment:

| Service | Platform | URL |
|---|---|---|
| **Frontend** | Vercel | [prism-ai-rust.vercel.app](https://prism-ai-rust.vercel.app) |
| **Backend** | Render | [prismai-backend-mruh.onrender.com](https://prismai-backend-mruh.onrender.com) |

**Environment variables required on Render:**
`GEMINI_API_KEY`, `GROQ_API_KEY`, `TAVILY_API_KEY`, `FRONTEND_URL` (your Vercel URL; comma-separate several to allow CORS). Optional: `PRISMAI_MODEL_*`, `PRISMAI_EMBEDDINGS` (`gemini` default / `local` / `off`), `PRISMAI_DB_PATH`, `PRISMAI_CORS_ORIGIN_REGEX`.

> Render's free tier has no persistent disk, so the SQLite database is wiped on every restart or deploy. The app tolerates this (stale chat ids are re-created), but history won't survive restarts there — attach a disk and point `PRISMAI_DB_PATH` at it to keep history.

> The backend runs on Render's free tier, which cold-starts after inactivity. The first request may take 30–60 seconds; the frontend displays a status banner during this period.

---

## Key Metrics

| Metric | Value |
|---|---|
| Intent categories | 9 |
| LLM providers | 2 (Groq, Gemini) |
| Available models | 4 |
| Avg token reduction (Query Rewriter, v1 prompt) | **16.92%** |
| Evaluation dataset | 100 queries |
| Streaming protocol | Server-Sent Events (SSE) |
| Database | SQLite (async, session-isolated) |

---

## License

MIT License — free to fork, extend, and build upon.
