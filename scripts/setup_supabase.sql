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
-- Create user_documents table
CREATE TABLE user_documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    storage_path VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);