-- Migration 006: Add superseded tracking to moments
--
-- Adds superseded (boolean) and superseded_by (uuid) columns so that
-- re-extracted or backfilled moments can replace older versions without
-- deleting them. Orient queries filter on WHERE NOT superseded by default.
--
-- Safe to run multiple times (IF NOT EXISTS / IF EXISTS guards).

-- Add superseded flag (default false — all existing moments stay active)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'moments' AND column_name = 'superseded'
    ) THEN
        ALTER TABLE moments ADD COLUMN superseded BOOLEAN NOT NULL DEFAULT FALSE;
    END IF;
END $$;

-- Add superseded_by (nullable UUID pointing to the replacement moment)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'moments' AND column_name = 'superseded_by'
    ) THEN
        ALTER TABLE moments ADD COLUMN superseded_by UUID;
    END IF;
END $$;

-- Index for orient query performance (most queries filter on NOT superseded)
CREATE INDEX IF NOT EXISTS idx_moments_superseded ON moments (superseded) WHERE NOT superseded;

-- Foreign key: superseded_by must reference an existing moment
-- (no constraint — moments may be in different states during migration)
