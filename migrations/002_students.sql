-- Students uploaded through the API. gen_random_uuid() is built in on Postgres 13+.
CREATE TABLE IF NOT EXISTS students (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    profile JSONB NOT NULL,
    summary TEXT NOT NULL,
    embedding VECTOR(384),
    embed_model TEXT NOT NULL,
    parser_confidence REAL,
    parser_used_llm BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS papers_professor_email_idx ON papers (professor_email);
CREATE INDEX IF NOT EXISTS professors_email_idx ON professors (email);
