-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create messages table
CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR NOT NULL,
    role VARCHAR NOT NULL,
    content TEXT NOT NULL,
    image_url VARCHAR,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Create user_documents table
CREATE TABLE user_documents (
    id SERIAL PRIMARY KEY,
    document_id VARCHAR NOT NULL,
    file_name VARCHAR NOT NULL,
    storage_path VARCHAR NOT NULL,
    session_id VARCHAR NOT NULL,
    public_url VARCHAR NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

-- Drop existing documents table if it exists
DROP TABLE IF EXISTS documents;

-- Create documents table for vector storage
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT,
    metadata JSONB,
    embedding VECTOR(3072)
);

-- Create index for faster vector searches
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);

-- Create match_documents function for similarity search
CREATE OR REPLACE FUNCTION match_documents (
    query_embedding VECTOR(3072),
    match_count INT DEFAULT NULL,
    filter JSONB DEFAULT '{}'
) RETURNS TABLE (
    id UUID,
    content TEXT,
    metadata JSONB,
    similarity FLOAT
) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        id,
        content,
        metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM documents
    WHERE metadata @> filter
    ORDER BY similarity DESC
    LIMIT match_count;
END;
$$;

-- Enable row-level security on documents table
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Set permissions for service role to insert into documents table
CREATE POLICY "Allow service role insert on documents" ON documents
FOR INSERT
WITH CHECK (auth.role() = 'service_role');

-- Set permissions for service role to select from documents table
CREATE POLICY "Allow service role select on documents" ON documents
FOR SELECT
USING (auth.role() = 'service_role');