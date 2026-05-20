-- Migration 005: Add occurred_at column to moments table
--
-- Separates "when did this moment happen" (occurred_at) from "when was it
-- recorded in the database" (created_at). Critical for backfilled memories:
-- the May 4 batch of 26 memories covering March 14 – early May all share
-- created_at ≈ 2026-05-04, but occurred_at preserves the real timeline.
--
-- Retrieval queries should filter/sort by occurred_at for narrative ordering.
-- Debug/audit queries use created_at for system provenance.
--
-- Backfill: existing rows get occurred_at from the date column (if set),
-- falling back to created_at. New rows should set occurred_at explicitly.
--
-- Idempotent: safe to re-run.

ALTER TABLE moments ADD COLUMN IF NOT EXISTS occurred_at TIMESTAMPTZ;

-- Backfill from date column where available
UPDATE moments SET occurred_at = date::timestamptz
WHERE date IS NOT NULL AND occurred_at IS NULL;

-- Fall back to created_at for anything still null
UPDATE moments SET occurred_at = created_at
WHERE occurred_at IS NULL;

-- Index for temporal queries
CREATE INDEX IF NOT EXISTS idx_moments_occurred_at
ON moments (occurred_at DESC);
