"""Fetch recent papers per professor from Semantic Scholar, embed, and store them.

    python -m pipelines.fetch_papers            # only professors with no papers yet
    python -m pipelines.fetch_papers --all      # refetch everyone, replacing their papers
    python -m pipelines.fetch_papers --only "Deming Chen" --dry-run

Name search alone pulls in namesakes (e.g. an emergency-medicine "Andrew Singer"), so
attribution works like this:
  1. search papers for "<first> <last>" plus up to two of the professor's research terms
  2. for every result author whose name contains the last name, add a vote equal to the
     paper's similarity to the professor's research-profile embedding
  3. keep only papers by the single author ID with the most votes, and only if that
     author clears MIN_VOTES; otherwise store nothing rather than a wrong paper.
"""
import argparse
import time
from collections import defaultdict

import requests

from app.config import get_settings
from app.db import connect, to_vector
from app.services.embedder import get_embedder

from pipelines.embed_professors import clean_interests

_key = get_settings().semantic_scholar_api_key
HEADERS = {"x-api-key": _key} if _key else {}

PAPER_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SEARCH_FIELDS = "title,abstract,year,url,authors"

MIN_VOTES = 1.0       # summed similarity the winning author ID needs
PAPERS_PER_PROF = 10
_MIN_INTERVAL = 1.1   # keyed limit is 1 request/second
_last_call = 0.0


def _get(url, params, retries=6):
    """Throttled GET that backs off and retries on 429 / 5xx."""
    global _last_call
    for attempt in range(retries):
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as e:  # timeouts / dropped connections
            print(f"    retrying after {type(e).__name__}")
            time.sleep(2 ** attempt)
            continue
        if resp.status_code != 429 and resp.status_code < 500:
            return resp
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Semantic Scholar unavailable after {retries} attempts: {url}")


def _cosine(a, b):
    # all-MiniLM embeddings are L2-normalised, so the dot product is the cosine
    return sum(x * y for x, y in zip(a, b))


def research_terms(prof, n=2):
    terms = []
    for col in ("research_interests", "research_areas", "research_topics"):
        terms += clean_interests(prof.get(col) or [])
    return list(dict.fromkeys(terms))[:n]


def find_papers(prof, prof_embedding, limit=PAPERS_PER_PROF):
    """Return (papers, author_id, votes) for the best-matching Semantic Scholar author."""
    parts = [p for p in prof["name"].split() if len(p) > 1]
    first, last = parts[0], parts[-1]
    name = f"{first} {last}"
    queries = [name] + [f"{name} {t}" for t in research_terms(prof)]

    seen = {}
    for q in queries:
        resp = _get(PAPER_SEARCH_URL, {"query": q, "fields": SEARCH_FIELDS, "limit": 50})
        if resp.status_code == 200:
            for p in resp.json().get("data", []):
                if p.get("abstract") and p.get("title"):
                    seen.setdefault(p["paperId"], p)

    embedder = get_embedder()
    votes = defaultdict(float)
    by_author = defaultdict(list)
    for p in seen.values():
        emb = embedder.encode(f"{p['title']}. {p['abstract']}")
        p["_embedding"] = emb
        sim = _cosine(prof_embedding, emb)
        for a in p.get("authors", []):
            if last.lower() in (a.get("name") or "").lower() and a.get("authorId"):
                votes[a["authorId"]] += max(sim, 0.0)
                by_author[a["authorId"]].append(p)

    if not votes:
        return [], None, 0.0
    best = max(votes, key=votes.get)
    if votes[best] < MIN_VOTES:
        return [], best, votes[best]

    papers = sorted(by_author[best], key=lambda p: p.get("year") or 0, reverse=True)
    return papers[:limit], best, votes[best]


def store_papers(conn, professor_email, papers, replace=False):
    if replace:
        conn.execute("DELETE FROM papers WHERE professor_email = %s", (professor_email,))
    stored = 0
    for p in papers:
        # papers has no unique constraint, so dedupe explicitly on (professor, title)
        stored += conn.execute("""
            INSERT INTO papers (professor_email, title, abstract, year, url, embedding)
            SELECT %s, %s, %s, %s, %s, %s::vector
            WHERE NOT EXISTS (SELECT 1 FROM papers WHERE professor_email = %s AND title = %s)
        """, (professor_email, p["title"], p["abstract"], p.get("year"), p.get("url", ""),
              to_vector(p["_embedding"]), professor_email, p["title"])).rowcount
    return stored


def main(refetch_all=False, only=None, dry_run=False):
    sql = """SELECT name, email, embedding::text AS embedding,
                    research_interests, research_areas, research_topics
             FROM professors p WHERE email IS NOT NULL AND embedding IS NOT NULL"""
    if only:
        sql += " AND name ILIKE %(only)s"
    elif not refetch_all:
        sql += " AND NOT EXISTS (SELECT 1 FROM papers x WHERE x.professor_email = p.email)"
    with connect() as conn:
        profs = conn.execute(sql + " ORDER BY name", {"only": f"%{only}%"}).fetchall()

    replace = refetch_all or bool(only)
    total, skipped, failed = 0, [], []
    for prof in profs:
        emb = [float(x) for x in prof["embedding"].strip("[]").split(",")]
        try:
            papers, author_id, votes = find_papers(prof, emb)  # slow network part, no DB held
        except RuntimeError as e:
            print(f"  {prof['name']:<35} -> FAILED ({e}); rerun to retry")
            failed.append(prof["name"])
            continue
        label = f"{prof['name']:<35} author={author_id} votes={votes:.2f}"
        if dry_run:
            print(f"  {label}  -> {len(papers)} papers" + (f", e.g. {papers[0]['title'][:60]!r}" if papers else ""))
            continue
        # Short-lived connection per professor: hosted poolers drop long idle sessions.
        with connect() as conn:
            n = store_papers(conn, prof["email"], papers, replace=replace)
            conn.commit()
        if not papers:
            print(f"  {label}  -> skipped (no confident match)" + (", old papers cleared" if replace else ""))
            skipped.append(prof["name"])
        else:
            print(f"  {label}  -> stored {n}")
        total += n

    print(f"\nDone. Stored {total} papers; {len(skipped)} skipped; {len(failed)} failed on network errors.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="refetch every professor and replace their papers")
    ap.add_argument("--only", help="substring of one professor's name")
    ap.add_argument("--dry-run", action="store_true", help="print matches without writing")
    args = ap.parse_args()
    main(refetch_all=args.all, only=args.only, dry_run=args.dry_run)
