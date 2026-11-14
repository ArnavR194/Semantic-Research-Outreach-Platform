"""Apply migrations/*.sql in order. All migrations are idempotent (IF NOT EXISTS).

    python -m pipelines.migrate
"""
from app.config import ROOT
from app.db import connect


def main():
    files = sorted((ROOT / "migrations").glob("*.sql"))
    with connect() as conn:
        for f in files:
            print(f"Applying {f.name}")
            conn.execute(f.read_text())
    print("Done.")


if __name__ == "__main__":
    main()
