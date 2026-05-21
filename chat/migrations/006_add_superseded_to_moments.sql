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

-- Partial index for orient query: covers WHERE NOT superseded ORDER BY occurred_at DESC
DROP INDEX IF EXISTS idx_moments_superseded;
CREATE INDEX IF NOT EXISTS idx_moments_active_occurred
    ON moments (occurred_at DESC) WHERE NOT superseded;

-- No FK constraint on superseded_by — keeps backfill/migration flexible
-- (caller is responsible for valid references)
