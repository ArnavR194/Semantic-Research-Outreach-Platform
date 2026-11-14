"""CLI end-to-end run without the API (replaces the old ranking/email_generator.py __main__).

    python -m pipelines.demo ML_Resume.pdf --top 3 [--generator llm]
"""
import argparse

from app import repository as repo
from app.db import connect
from app.services import pipeline
from app.services.embedder import get_embedder
from app.services.resume_parser import build_student_summary, parse_resume


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--generator", choices=["template", "llm"], default="template")
    args = ap.parse_args()

    result = parse_resume(args.pdf)
    profile = result.profile
    embedding = get_embedder().encode(build_student_summary(profile))
    print(f"Parsed {profile.name} (confidence {result.confidence}, llm={result.used_llm})\n")

    with connect() as conn:
        for i, r in enumerate(pipeline.match(conn, profile, embedding, top_k=args.top), 1):
            s = r.scores
            print("=" * 60)
            print(f"{i}. {r.candidate.name} | final {s['final']} | emb {s['embedding']} | "
                  f"skills {s['skills']} | interests {s['interests']} | course {s['coursework']}")
            prof = repo.get_professor(conn, r.candidate.professor_id, with_papers=False)
            subject, body, ctx = pipeline.draft_email(conn, prof, profile, embedding, args.generator)
            if ctx.paper_title:
                print(f"Paper referenced: {ctx.paper_title}")
            print(f"\nSubject: {subject}\n\n{body}\n")


if __name__ == "__main__":
    main()
