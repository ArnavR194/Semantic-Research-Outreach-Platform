"""Load data/professors.json into Postgres (insert missing, update existing) and (re)embed each.

    python -m pipelines.embed_professors
"""
import json

from app.config import DATA_DIR
from app.db import connect, to_vector
from app.services.embedder import get_embedder


def clean_interests(interests):
    """Split long comma-merged interest strings into individual clean terms."""
    cleaned = []
    for item in interests:
        for part in (p.strip() for p in item.split(",")):
            # Remove category prefixes like "Applications:", "Theory and Algorithms:"
            if ":" in part:
                part = part.split(":", 1)[1].strip()
            if part and len(part) > 2:
                cleaned.append(part)
    return cleaned


def build_summary(prof):
    """Normalized summary text for embedding."""
    parts = []
    interests = clean_interests(prof.get("research_interests", []))
    areas = clean_interests(prof.get("research_areas", []))
    topics = clean_interests(prof.get("research_topics", []))

    if interests:
        parts.append("Research interests: " + ", ".join(interests))
    if areas:
        parts.append("Research areas: " + ", ".join(areas))
    if topics:
        parts.append("Research topics: " + ", ".join(topics))
    if prof.get("bio"):
        parts.append("Bio: " + prof["bio"][:300])  # first 300 chars only; avoid noise

    return " | ".join(parts) if parts else prof["name"]


def main(path=DATA_DIR / "professors.json"):
    professors = json.loads(path.read_text())
    embedder = get_embedder()
    summaries = [build_summary(p) for p in professors]
    embeddings = embedder.encode_many(summaries)

    inserted = updated = 0
    with connect() as conn:
        for prof, summary, emb in zip(professors, summaries, embeddings):
            fields = {
                "name": prof["name"], "title": prof.get("title"), "email": prof.get("email"),
                "url": prof.get("url"), "university": prof.get("university"),
                "department": prof.get("department"), "bio": prof.get("bio"),
                "ri": prof.get("research_interests") or [], "ra": prof.get("research_areas") or [],
                "rt": prof.get("research_topics") or [], "summary": summary, "emb": to_vector(emb),
            }
            # Match on profile URL, or on email for rows loaded before URL was stored
            cur = conn.execute(
                """
                UPDATE professors SET name=%(name)s, title=%(title)s, email=%(email)s,
                    university=%(university)s, department=%(department)s, bio=%(bio)s,
                    research_interests=%(ri)s, research_areas=%(ra)s, research_topics=%(rt)s,
                    research_summary=%(summary)s, embedding=%(emb)s::vector
                WHERE url = %(url)s OR (%(email)s::text IS NOT NULL AND email = %(email)s)
                """,
                fields,
            )
            if cur.rowcount:
                updated += 1
            else:
                conn.execute(
                    """
                    INSERT INTO professors (name, title, email, url, university, department, bio,
                        research_interests, research_areas, research_topics, research_summary, embedding)
                    VALUES (%(name)s, %(title)s, %(email)s, %(url)s, %(university)s, %(department)s,
                        %(bio)s, %(ri)s, %(ra)s, %(rt)s, %(summary)s, %(emb)s::vector)
                    """,
                    fields,
                )
                inserted += 1
    print(f"Done. Inserted {inserted}, updated {updated} professors with model {embedder.name}.")


if __name__ == "__main__":
    main()
