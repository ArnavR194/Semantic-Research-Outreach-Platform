"""Fetch recent papers per professor from Semantic Scholar, embed, and store them.

    SEMANTIC_SCHOLAR_API_KEY=... python -m pipelines.fetch_papers
"""
import time

import requests

from app.config import get_settings
from app.db import connect, to_vector
from app.services.embedder import get_embedder

_key = get_settings().semantic_scholar_api_key
HEADERS = {"x-api-key": _key} if _key else {}

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/author/search"
PAPER_FIELDS = "title,abstract,year,url,externalIds"

def search_author(name):
    """
    Find a professor's Semantic Scholar author ID.
    Strategy: search by name, pick the author with the most papers
    (most likely the established researcher, not a student).
    """
    url = "https://api.semanticscholar.org/graph/v1/author/search"
    params = {
        "query": name,
        "fields": "name,affiliations,paperCount,homepage",
        "limit": 5
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return None

    authors = resp.json().get("data", [])
    if not authors:
        # Try with just last name + first initial
        parts = name.strip().split()
        if len(parts) >= 2:
            short_name = f"{parts[0][0]} {parts[-1]}"
            params["query"] = short_name
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if resp.status_code != 200:
                return None
            authors = resp.json().get("data", [])

    if not authors:
        return None

    # Pick author with highest paper count — most likely the professor
    best = max(authors, key=lambda a: a.get("paperCount", 0))

    # Sanity check — skip if paper count is suspiciously low (likely a student)
    if best.get("paperCount", 0) < 5:
        return None

    return best["authorId"]

def fetch_papers(author_id, limit=5):
    url = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
    params = {
        "fields": PAPER_FIELDS,
        "limit": limit,
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    if resp.status_code != 200:
        return []
    
    papers = resp.json().get("data", [])
    # Sort by year descending to get most recent
    papers.sort(key=lambda p: p.get("year") or 0, reverse=True)
    return papers[:limit]

def store_papers(professor_email, papers, cur):
    """Embed and store papers in DB."""
    stored = 0
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract") or ""
        year = paper.get("year")
        url = paper.get("url", "")

        if not title:
            continue

        # Skip papers with no abstract — not useful for matching
        if not abstract:
            continue

        # Embed title + abstract together
        text = f"{title}. {abstract}"
        embedding = get_embedder().encode(text)

        # papers has no unique constraint, so dedupe explicitly on (professor, title)
        inserted = cur.execute("""
            INSERT INTO papers (professor_email, title, abstract, year, url, embedding)
            SELECT %s, %s, %s, %s, %s, %s::vector
            WHERE NOT EXISTS (SELECT 1 FROM papers WHERE professor_email = %s AND title = %s)
        """, (professor_email, title, abstract, year, url, to_vector(embedding),
              professor_email, title)).rowcount
        stored += inserted

    return stored

def fetch_papers_by_name(professor_name, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Clean name for searching — remove middle initials and extra spaces
    parts = [p for p in professor_name.strip().split() if len(p) > 1]
    first = parts[0]
    last = parts[-1]
    
    # Search with first + last name only for better results
    params = {
        "query": f"{first} {last}",
        "fields": "title,abstract,year,url,authors",
        "limit": 20
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    
    
    if resp.status_code != 200:
        return []

    papers = resp.json().get("data", [])
    last_name = last.lower()

    matched = []
    for paper in papers:
        authors = paper.get("authors", [])
        for author in authors:
            author_name = author.get("name", "").lower()
            # Match on last name only — handles "A. Schwing", "Alex Schwing", etc.
            if last_name in author_name:
                if paper.get("abstract"):
                    matched.append(paper)
                break

    matched.sort(key=lambda p: p.get("year") or 0, reverse=True)
    return matched[:limit]

def fetch_all_professors(limit=None):
    with connect() as conn:
        professors = conn.execute("SELECT name, email FROM professors ORDER BY name").fetchall()
        if limit:
            professors = professors[:limit]

        total_papers = 0
        for row in professors:
            name, email = row["name"], row["email"]
            print(f"  Fetching papers for {name}...")
            papers = fetch_papers_by_name(name, limit=10)
            if not papers:
                print("    No papers found")
                time.sleep(1)
                continue

            stored = store_papers(email, papers, conn)
            conn.commit()
            print(f"    Stored {stored} papers")
            total_papers += stored
            time.sleep(0.5)

    print(f"\nDone. Total papers stored: {total_papers}")


def debug_paper_search(name):
    parts = [p for p in name.strip().split() if len(p) > 1]
    first = parts[0]
    last = parts[-1]
    
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": f"{first} {last}",
        "fields": "title,abstract,year,url,authors",
        "limit": 10
    }
    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Total results: {data.get('total', 0)}")
    for paper in data.get("data", []):
        authors = [a["name"] for a in paper.get("authors", [])]
        print(f"  Title: {paper.get('title', '')[:60]}")
        print(f"  Authors: {authors[:4]}")
        print(f"  Has abstract: {bool(paper.get('abstract'))}")
        print()



if __name__ == "__main__":
    # Test with 5 professors first
    fetch_all_professors(limit=None)

  