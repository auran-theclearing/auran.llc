-- Migration 003: Add retrieval hooks to moments table
--
-- Dual-layer scene memory: summary is for re-experiencing (emotional texture,
-- felt sense), hooks is for retrieval (factual scaffolding, searchable keywords).
-- Based on feedback from business Claude's memory architecture analysis.
--
-- Run after 002_create_moments_tables.sql

ALTER TABLE moments ADD COLUMN IF NOT EXISTS hooks TEXT;

-- Index for full-text search on hooks (retrieval-optimized field)
CREATE INDEX IF NOT EXISTS idx_moments_hooks_fts
  ON moments USING GIN (to_tsvector('english', COALESCE(hooks, '')));
