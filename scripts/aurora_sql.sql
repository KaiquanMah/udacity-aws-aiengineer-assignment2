-- enable 'pgvector' extension in the Postgres DB
-- so DB can store/write n query/read vector embeddings
"CREATE EXTENSION IF NOT EXISTS vector;",


-- create schema/logical namespace 'bedrock_integration'
-- to organise DB objs eg tables, fns
"CREATE SCHEMA IF NOT EXISTS bedrock_integration;",


-- Create DB role bedrock_user
-- if the role already exists, print/raise a notice
-- what does 'LOGIN' do here???? users with the role can login/establish connection w the postgres DB
"DO $$ BEGIN CREATE ROLE bedrock_user LOGIN; EXCEPTION WHEN duplicate_object THEN RAISE NOTICE 'Role already exists'; END $$;",


-- grant ALL permissions on 'bedrock_integration' schema objects to the role
-- read, create, modify/update, delete
"GRANT ALL ON SCHEMA bedrock_integration to bedrock_user;",


-- all subsequent cmds use permissions assigned to 'bedrock_user'
"SET SESSION AUTHORIZATION bedrock_user;",


-- create table for bedrock's RAG knowledge base
"""
CREATE TABLE IF NOT EXISTS bedrock_integration.bedrock_kb (
    id uuid PRIMARY KEY,
    -- Amazon Titan Embeddings - 1536 dims vector
    embedding vector(1536),
    chunks text,
    -- metadata = eg name, author, creation date
    metadata json
);
""",


-- create vector search index
-- Hierarchical Navigable Small World (HNSW) algorithm = approximate nearest neighbor (ANN) search
-- using 'embedding' col
-- measure cosine similarity
"CREATE INDEX IF NOT EXISTS bedrock_kb_embedding_idx ON bedrock_integration.bedrock_kb USING hnsw (embedding vector_cosine_ops);"