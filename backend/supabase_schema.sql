-- =========================================================================
-- Emovision — Supabase PostgreSQL Production Database Schema DDL
-- Real-Time Multi-Person Facial Expression Recognition Platform
-- =========================================================================

-- 1. Create sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(64) PRIMARY KEY,
    session_name VARCHAR(255) NOT NULL DEFAULT 'Live Session',
    source_type VARCHAR(64) NOT NULL DEFAULT 'webcam',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE NULL,
    duration NUMERIC(10, 1) DEFAULT 0.0,
    people_count INT DEFAULT 0,
    total_predictions INT DEFAULT 0,
    dominant_expression VARCHAR(64) NULL,
    average_confidence NUMERIC(5, 1) DEFAULT 0.0,
    status VARCHAR(32) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id BIGSERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    person_id INT NOT NULL,
    frame_number INT NOT NULL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expression VARCHAR(64) NOT NULL,
    confidence NUMERIC(5, 2) NOT NULL,
    x INT NOT NULL,
    y INT NOT NULL,
    width INT NOT NULL,
    height INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Create Performance Indexes for Fast Analytics Queries
CREATE INDEX IF NOT EXISTS idx_predictions_session_id ON predictions(session_id);
CREATE INDEX IF NOT EXISTS idx_predictions_person_id ON predictions(person_id);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_expression ON predictions(expression);

-- 4. Supabase Row Level Security (RLS) Policies
ALTER TABLE sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow backend service role & authenticated API access
CREATE POLICY "Allow backend service role full access on sessions"
    ON sessions FOR ALL
    USING (true)
    WITH CHECK (true);

CREATE POLICY "Allow backend service role full access on predictions"
    ON predictions FOR ALL
    USING (true)
    WITH CHECK (true);
