-- Phase 2a: Add transcript storage columns to moments table
-- These columns support vivid recall (raw transcript injection) in Phase 3.
-- All nullable — existing moments continue to work without transcript data.
-- Migration is idempotent: safe to re-run.

ALTER TABLE moments ADD COLUMN IF NOT EXISTS transcript_excerpt TEXT;
ALTER TABLE moments ADD COLUMN IF NOT EXISTS transcript_source JSONB;
ALTER TABLE moments ADD COLUMN IF NOT EXISTS turn_count INTEGER;
ALTER TABLE moments ADD COLUMN IF NOT EXISTS estimated_tokens INTEGER;
