# MARA — Multi-Agent Research Assistant

Final-year B.Tech CSE project. A user types a research question and receives a
citation-backed literature summary produced by four specialized AI agents:

| Agent | Role | Phase | Status |
|---|---|---|---|
| **Planner** | Breaks the question into 3–5 focused sub-questions | Phase 4 | ✅ Built |
| **Retriever** | Semantic search over indexed papers (RAG) | Phase 2–3 | ✅ Built |
| **Critic** | Fact-checks claims against retrieved evidence | Phase 5 | ✅ Built |
| **Writer** | Composes the final cited summary | Phase 6 | ✅ Built |
| **Orchestrator** | LangGraph pipeline with cross-verification | Phase 7 | ✅ Built |

Orchestrated with LangGraph (Phase 7), served by FastAPI (Phase 8), UI in
Streamlit (Phase 9), evaluated in Phase 11.

Base paper: Flores, Ramirez, Mauricio — *"Expert Multi-Agent Conversational
System Using RAG and Dynamic Text-to-SQL for Government Transparency"*
(IEEE Access, 2025).

> **Roadmap & team guide:** see [`docs/MARA_MASTER_GUIDE.pdf`](docs/MARA_MASTER_GUIDE.pdf)
> (or the [markdown source](docs/MARA_MASTER_GUIDE.md)) — consolidated build
> priorities, feature tiers, team roles, evaluation plan, and viva prep.
> Regenerate the PDF after editing the markdown: `python docs/make_pdf.py`
> (needs `fpdf2`: `pip install fpdf2`).

## Project structure (built so far)

```
mara-project/
├── backend/
│   ├── config.py            # .env loading, model names, Qdrant client factory
│   ├── agents/
│   │   ├── planner.py       # ✅ Phase 4 — Planner agent
│   │   ├── retriever.py     # ✅ Phase 3 — Retriever agent
│   │   └── critic.py        # ✅ Phase 5 — Critic agent (fact-checking gate)
│   ├── graph.py             # ✅ Phase 7 — LangGraph orchestrator (cross-verification)
│   ├── mock_llm.py          # ✅ MOCK_LLM=1 deterministic stubs (no API keys)
│   ├── tracing.py           # ✅ per-run JSON traces -> Phase 11 metrics
│   └── ingestion/
│       ├── fetch_papers.py  # ✅ Phase 2 — arXiv fetching
│       ├── embeddings.py    # ✅ Phase 2 — batched OpenAI embeddings
│       └── index_papers.py  # ✅ Phase 2 — chunk + embed + index into Qdrant
├── tests/
│   ├── test_planner.py              # 12 offline tests
│   ├── test_ingestion_retriever.py  # 16 offline tests (in-memory Qdrant)
│   ├── test_critic.py               # 24 offline tests
│   ├── test_writer_graph.py         # 15 offline tests (Writer + full graph)
│   └── test_mock_tracing.py         # 16 offline tests (mock mode + traces)
├── docs/
│   ├── MARA_MASTER_GUIDE.md     # consolidated roadmap & team guide (source)
│   ├── MARA_MASTER_GUIDE.pdf    # same, as PDF for sharing
│   └── make_pdf.py              # regenerate the PDF from the markdown
├── .env.example             # Copy to .env and fill in keys
├── requirements.txt
└── README.md
```

## Setup

1. **Python 3.10+** — verify with `python --version`.
2. **Virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows (Git Bash: source venv/Scripts/activate)
   pip install -r requirements.txt
   ```
3. **API keys:** copy `.env.example` to `.env` and add your
   [Anthropic key](https://console.anthropic.com). `.env` is git-ignored.
   **No keys yet?** Set `MOCK_LLM=1` (in `.env` or the shell) to run the
   entire pipeline with deterministic stubs — zero keys, zero cost.
   Every run also writes a JSON trace to `traces/` (disable: `TRACE=0`,
   relocate: `TRACE_DIR=<path>`); `python -m backend.tracing` summarizes
   all traces into the evaluation metrics.
4. **Git identity + GitHub (once, per machine):**
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"
   # Create the repo at github.com/new (name: mara), then:
   git remote add origin https://github.com/<your-username>/mara.git
   git push -u origin main
   ```
   Every teammate clones the repo and works in a feature branch; PRs to
   `main` are gated by CI (`.github/workflows/ci.yml` runs the offline
   tests on Python 3.11 & 3.12).

## Test the Planner (Phase 4 checkpoints)

**Offline tests (free, no API key):**
```bash
pytest tests/ -v
```

**Live checkpoint** (uses real API credits) — run from project root:
```bash
python -c "from backend.agents.planner import plan; print(plan('How does federated learning preserve privacy?'))"
```
Expected: 3–5 sensible, non-overlapping sub-questions printed as a list.

### Standalone demo script

```bash
python -m backend.agents.planner
```

## Test the Critic (Phase 5 checkpoint)

**Live checkpoint** (uses real API credits):

```bash
python -m backend.agents.critic
```
Accept the demo excerpt and it should print `Verdict: SUPPORTED`; try a
claim unrelated to the excerpt and it should print `NOT_SUPPORTED`.

## Test the Retriever (Phase 2–3 checkpoints)

Run these in order (all use real APIs: arXiv is free, embeddings need an OpenAI key):

```bash
# 1. arXiv fetching — expect 5 real paper titles
python -m backend.ingestion.fetch_papers

# 2. Index 25 papers into Qdrant (chunk + embed + upsert)
python -m backend.ingestion.index_papers

# 3. Semantic search — expect relevant chunks with scores
python -m backend.agents.retriever
```

**Qdrant backend:** set `QDRANT_PATH=./qdrant_storage` in `.env` for an
embedded on-disk store (no Docker needed), or `QDRANT_URL` for a Docker/
Cloud server. With neither set, an in-memory store is used (lost on restart).

## Design decisions (for your report / viva)

- **Defensive JSON parsing** (`parse_subquestions`): the naive
  `json.loads(response.text)` from the guide crashes when the model wraps JSON
  in code fences or adds prose. The planner handles clean JSON, fenced JSON,
  JSON embedded in prose, and numbered/bulleted lists as a last resort.
- **Graceful degradation:** if parsing fails after a retry, `plan()` falls back
  to the original question instead of raising — the pipeline never dies on a
  malformed planner response.
- **Lazy client creation:** importing `planner.py` works without an API key,
  so unit tests and CI run offline.

### Retriever (Phase 2–3)

- **Idempotent ingestion:** chunks get deterministic UUID5 IDs derived from
  `(paper URL, chunk index)`, so re-running indexing updates points in place
  instead of duplicating them (the guide's auto-increment IDs duplicate).
- **Batched embeddings:** `embed_texts()` batches up to 64 inputs per OpenAI
  call — roughly half the API calls versus one-per-chunk.
- **Three Qdrant backends:** `QDRANT_URL` (server/cloud), `QDRANT_PATH`
  (embedded on-disk — no Docker needed), or in-memory (tests). Chosen via
  `config.get_qdrant_client()`.
- **200-word chunks:** the guide suggested 500, but arXiv abstracts are short;
  smaller chunks stay topical and retrieval precision stays high.
- **Fail-soft retriever:** querying before indexing returns `[]` instead of
  crashing, so the downstream pipeline degrades to "insufficient evidence".
- **16 offline tests** exercise the real chunk → index → retrieve pipeline
  against in-memory Qdrant with deterministic stub embeddings — zero cost.

### Critic (Phase 5)

- **Fail-closed verdicts:** an unrecognized, malformed, or empty response
  parses as `NOT_SUPPORTED` — an unverifiable claim can never reach the
  Writer. (The guide's `text.split()[0]` crashes on Markdown like
  `**SUPPORTED**`; token-scan parsing handles bold/code-fence/space forms.)
- **Bounded prompts:** sources are capped at 8 excerpts × 2000 chars so a
  pathological chunk can't blow the context window or the token budget.
- **Retry, then fail closed:** one retry on malformed output; a claim that
  still can't be verified is dropped by the orchestrator (Phase 7), which is
  exactly the design rule that makes MARA trustworthy.
