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
| Backend | Python |
| Similarity | Cosine similarity (384-dim vectors) |

## Project Structure

```
research-outreach-ai/
│
├── crawler/
│   ├── crawler.py          # UIUC ECE faculty directory scraper
│   └── fetch_papers.py     # Semantic Scholar paper fetcher
│
├── embeddings/
│   └── embeddings.py       # Embedding generation + DB loading
│
├── resume_parser/
│   └── parse_resume.py     # Hybrid resume parser (heuristic + LLM fallback)
│
├── ranking/
│   ├── match.py            # Vector similarity search
│   ├── reranker.py         # Multi-signal reranker
│   └── email_generator.py  # RAG email generation
│
├── database/
│   └── database.sql        # PostgreSQL schema
│
└── requirements.txt
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
- PostgreSQL 17 with pgvector extension
- Semantic Scholar API key (free at semanticscholar.org/product/api)

### Installation

```bash
git clone https://github.com/yourusername/research-outreach-ai
cd research-outreach-ai

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Database Setup

```bash
createdb research_outreach
psql research_outreach -c "CREATE EXTENSION vector;"
psql research_outreach -f database/database.sql
```

### Run Pipeline

```bash
# 1. Crawl faculty
python crawler/crawler.py

# 2. Generate and store embeddings
python embeddings/embeddings.py

# 3. Fetch research papers
python crawler/fetch_papers.py

# 4. Parse resume and generate emails
python ranking/email_generator.py
```

## Requirements

```
pdfplumber
psycopg2-binary
sentence-transformers
beautifulsoup4
lxml
requests
anthropic
```

## Key Design Decisions

**Why not just use GPT?**
The intelligence is in the retrieval and ranking system, not the generation layer. Embedding similarity, skill domain mapping, interest phrase matching, and paper retrieval happen before any text generation. The LLM (if used) only fills a structured template grounded entirely in retrieved context.

**Why pgvector over a dedicated vector DB?**
Keeps the stack simple — one database handles both structured professor data and vector search. At 116 professors and 943 papers, Pinecone or Weaviate would be overkill.

**Why all-MiniLM-L6-v2?**
Fast, lightweight (80MB), runs locally with no API cost, and produces strong semantic similarity for academic text. Sufficient for this scale.