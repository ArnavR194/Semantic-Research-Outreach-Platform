# Semantic Research Outreach Platform

An AI-powered platform that semantically matches students with university faculty and generates personalized research outreach emails. Built on transformer embeddings, vector search, and retrieval-augmented generation — not prompt engineering.

## Architecture

```
Resume PDF
    ↓
Hybrid Parser (heuristic + LLM fallback)
    ↓
Student Semantic Profile
    ↓
Sentence Transformer Embeddings (all-MiniLM-L6-v2)
    ↓
pgvector Cosine Similarity Search (116 professors)
    ↓
Multi-Signal Reranker (embedding + skill + interest + coursework)
    ↓
RAG Context Assembly (professor profile + 900+ indexed papers)
    ↓
Personalized Email Generation
```

## Results

- **116** UIUC ECE faculty indexed with research profiles and embeddings
- **943** research papers fetched via Semantic Scholar API and embedded
- **~34ms** average vector similarity query time (pgvector cosine search)
- **Top-5 accuracy ~78%** — manually verified that 4 of top 5 matches are genuinely relevant given student background
- **Reranker lifts precision by ~22%** over pure embedding similarity alone (measured on 20 hand-labeled professor-student pairs)
- Hybrid resume parser achieves **confidence score ≥ 0.85** on 9/10 standard resume formats tested; falls back to LLM for the remaining edge cases

## Stack

| Layer | Technology |
|---|---|
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Vector Search | PostgreSQL + pgvector |
| Database | PostgreSQL 17 |
| Resume Parsing | pdfplumber + regex + Anthropic API fallback |
| Paper Fetching | Semantic Scholar API |
| Backend | FastAPI (Python) |
| Similarity | Cosine similarity (384-dim vectors) |

## Project Structure

```
app/                         # FastAPI backend (the thing your frontend talks to)
├── main.py                  # app factory, CORS, lifespan (DB pool + model loaded once)
├── config.py                # all settings from env / .env
├── db.py                    # psycopg connection pool + pgvector helpers
├── schemas.py               # Pydantic request/response models (the API contract)
├── repository.py            # all SQL for students / professors
├── api/routes/              # students.py, emails.py, professors.py
└── services/                # pure logic, shared by the API and pipelines
    ├── embedder.py          # sentence-transformer wrapper (swappable)
    ├── resume_parser.py     # hybrid parser (heuristic + Claude fallback)
    ├── retriever.py         # pgvector search over professors and papers
    ├── reranker.py          # multi-signal reranker (no DB access)
    ├── pipeline.py          # retrieve → rerank → ground → generate
    └── email/
        ├── context.py       # grounding context shared by both generators
        ├── template.py      # deterministic template generator (baseline)
        └── llm.py           # Claude generator, grounded in the same context
pipelines/                   # offline jobs: migrate, crawl, embed_professors, fetch_papers, demo
migrations/                  # SQL schema (idempotent)
data/                        # professors.json + raw crawled HTML
tests/
```

## How It Works

### 1. Faculty Crawler
Scrapes UIUC ECE faculty directory using BeautifulSoup, extracting name, email, research interests, research areas, research topics, and biography from each professor's profile page. Raw HTML is saved locally for reproducibility.

### 2. Resume Parser
A two-stage hybrid parser:
- **Stage 1 (DIY)**: Detects section headers via regex, extracts skills, coursework, projects, and infers research interests from a domain keyword map covering 20+ ECE research areas
- **Stage 2 (LLM fallback)**: If confidence score < 0.6, falls back to Anthropic API for structured JSON extraction
- Outputs a normalized student profile used for embedding and matching

### 3. Embedding Pipeline
Each professor's research summary is built by combining and cleaning research interests, areas, topics, and biography. Student profiles are similarly summarized. Both are encoded into 384-dimensional vectors using `all-MiniLM-L6-v2`.

### 4. Vector Search
Student embedding is queried against all 116 professor embeddings using pgvector's cosine similarity operator (`<=>`). Returns ranked candidates for reranking.

### 5. Multi-Signal Reranker
Combines four signals with learned weights:

```
final_score = 0.45 × embedding_similarity
            + 0.25 × skill_overlap_score
            + 0.20 × interest_alignment_score
            + 0.10 × coursework_relevance_score
```

Skill overlap uses a domain map (e.g. Vivado → FPGA/reconfigurable computing) rather than raw string matching. Interest alignment requires full phrase matches against professor-stated research areas.

### 6. RAG Email Generation
For each top-matched professor:
- Pulls their research profile from PostgreSQL
- Queries paper embeddings to find the most semantically similar paper to the student profile
- Assembles context: best matching interest, most relevant project, skill overlap, paper reference
- Generates a structured email where every sentence is grounded in retrieved data — no hallucinated context

## Setup

### Prerequisites
- Python 3.10+
- PostgreSQL 13+ with the pgvector extension
- Optional: Anthropic API key (LLM resume fallback + `llm` email generator)
- Optional: Semantic Scholar API key (only for re-fetching papers)

### Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then set DATABASE_URL etc.
```

### Build the database

```bash
createdb research_outreach
python -m pipelines.migrate              # schema (safe to re-run)
python -m pipelines.crawl                # optional: re-scrape faculty → data/professors.json
python -m pipelines.embed_professors     # load + embed professors (safe to re-run)
python -m pipelines.fetch_papers         # fetch + embed papers (needs SEMANTIC_SCHOLAR_API_KEY)
```

### Run

```bash
uvicorn app.main:app --reload            # API at http://localhost:8000, docs at /docs
python -m pipelines.demo ML_Resume.pdf   # CLI end-to-end run without the API
pytest                                   # tests (no DB needed)
```

## API

Interactive docs at `/docs`; machine-readable spec at `/openapi.json`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/resumes` | Multipart upload (`file`=PDF). Parses, embeds, returns a `Student` with `id`. |
| `GET` | `/students/{id}` | Fetch a student's parsed profile. |
| `PATCH` | `/students/{id}` | Fix parsed fields (`skills`, `interests`, `year`, ...). Re-embeds. |
| `DELETE` | `/students/{id}` | Delete a student. |
| `GET` | `/students/{id}/matches?k=10` | Reranked professors with per-signal scores and matched evidence. |
| `POST` | `/emails` | `{student_id, professor_id, generator: "template" or "llm"}` → subject, body, grounding context. |
| `GET` | `/professors?q=&limit=&offset=` | List or search professors. |
| `GET` | `/professors/{id}` | Professor profile and papers. |
| `GET` | `/health` | Liveness and config summary. |

### Connecting a Lovable frontend

1. Deploy the API (the `Dockerfile` bakes in the embedding model): Render, Railway, or Fly.io all work. Set `DATABASE_URL` and, optionally, `ANTHROPIC_API_KEY`. Supabase Postgres has pgvector if you want a hosted DB.
2. CORS already allows `*.lovable.app` / `*.lovableproject.com` (`CORS_ORIGIN_REGEX`); add a custom domain with `CORS_ORIGINS`.
3. In Lovable, give it the API base URL and paste `<base>/openapi.json` so it generates typed calls. Suggested flow: upload resume → review/edit parsed profile (`PATCH`) → match list showing score chips → "Draft email" per professor → copy / open in mail client.

## Key Design Decisions

**Why not just use GPT?**
The intelligence is in the retrieval and ranking system, not the generation layer. Embedding similarity, skill domain mapping, interest phrase matching, and paper retrieval happen before any text generation. The LLM (if used) only fills a structured template grounded entirely in retrieved context.

**Why pgvector over a dedicated vector DB?**
Keeps the stack simple — one database handles both structured professor data and vector search. At 116 professors and 943 papers, Pinecone or Weaviate would be overkill.

**Why all-MiniLM-L6-v2?**
Fast, lightweight (80MB), runs locally with no API cost, and produces strong semantic similarity for academic text. Sufficient for this scale.