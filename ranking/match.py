import psycopg2
from sentence_transformers import SentenceTransformer
import sys
sys.path.append(".")
from resume_parser.parse_resume import parse_resume, build_student_summary

def get_top_matches(pdf_path, top_k=10):
    # Build student embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    profile = parse_resume(pdf_path)
    summary = build_student_summary(profile)
    embedding = model.encode(summary).tolist()

    # Query DB for nearest professor embeddings
    conn = psycopg2.connect(dbname="research_outreach")
    cur = conn.cursor()

    cur.execute("""
        SELECT name, email, research_summary,
               1 - (embedding <=> %s::vector) AS similarity
        FROM professors
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """, (embedding, embedding, top_k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return profile, results

if __name__ == "__main__":
    profile, matches = get_top_matches("ML_Resume.pdf", top_k=10)

    print(f"Top matches for {profile['name']}:\n")
    for i, (name, email, summary, score) in enumerate(matches, 1):
        print(f"{i}. {name} ({email})")
        print(f"   Similarity: {score:.4f}")
        print(f"   Research: {summary[:120]}...")
        print()