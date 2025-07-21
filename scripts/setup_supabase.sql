-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create messages table
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    content TEXT NOT NULL,
    image_path VARCHAR,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);