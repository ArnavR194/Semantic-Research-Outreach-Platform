CREATE TABLE IF NOT EXISTS professors (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    title TEXT,
    email TEXT,
    url TEXT,
    university TEXT,
    department TEXT,
    bio TEXT,
    research_interests TEXT[],
    research_areas TEXT[],
    research_topics TEXT[],
    research_summary TEXT,
    embedding VECTOR(384)
);


CREATE TABLE IF NOT EXISTS papers (
    id SERIAL PRIMARY KEY,
    professor_email TEXT,
    title TEXT,
    abstract TEXT,
    year INTEGER,
    url TEXT,
    embedding VECTOR(384)
);
