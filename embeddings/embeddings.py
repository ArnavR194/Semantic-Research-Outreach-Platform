import json
import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def clean_interests(interests):
    """Split long comma-merged interest strings into individual clean terms."""
    cleaned = []
    for item in interests:
        # Split on commas within each interest string
        parts = [p.strip() for p in item.split(",")]
        for part in parts:
            # Remove category prefixes like "Applications:", "Theory and Algorithms:"
            if ":" in part:
                part = part.split(":", 1)[1].strip()
            if part and len(part) > 2:
                cleaned.append(part)
    return cleaned

def build_summary(prof):
    """Build clean normalized summary text for embedding."""
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
        # Only first 300 chars of bio — avoid noise
        parts.append("Bio: " + prof["bio"][:300])

    return " | ".join(parts) if parts else prof["name"]

def reembed_all():
    conn = psycopg2.connect(dbname="research_outreach")
    cur = conn.cursor()

    with open("professors.json") as f:
        professors = json.load(f)

    updated = 0
    for prof in professors:
        summary = build_summary(prof)
        embedding = model.encode(summary).tolist()

        cur.execute("""
            UPDATE professors
            SET research_summary = %s, embedding = %s
            WHERE email = %s
        """, (summary, embedding, prof["email"]))

        print(f"  Re-embedded: {prof['name']}")
        updated += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"\nDone. Re-embedded {updated} professors.")

if __name__ == "__main__":
    reembed_all()
