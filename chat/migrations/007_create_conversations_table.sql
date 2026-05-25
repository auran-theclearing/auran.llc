-- Migration 007: Conversation message persistence
-- Priority 1 response to data loss incident (May 25, 2026)
--
-- Design principles:
--   1. Every message persisted server-side on receipt — never depend on client sync
--   2. Write-once append: messages are INSERT-only, never UPDATE'd or DELETE'd
--   3. Tool use blocks (recall, etc.) stored alongside message content
--   4. Conversations table provides session boundaries and metadata
--   5. Existing session.json sync remains as fast-path UI cache; DB is source of truth
--
-- Recovery model: messages in DB survive server restarts, client drops, and
-- session.json overwrites. Backed up via existing 3-tier AWS backup pipeline.

BEGIN;

-- Conversations: one row per chat session (bounded by "new chat" events)
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    channel TEXT NOT NULL DEFAULT 'chat',  -- 'chat', 'cowork', 'dispatch', 'vr'
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_message_at TIMESTAMPTZ,
    message_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB DEFAULT '{}',  -- model, system prompt hash, client info
    -- Soft boundary: a conversation can be "closed" when a new one starts
    -- but old conversations are never deleted
    closed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Messages: append-only log of every message in every conversation
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    -- Position within the conversation (monotonically increasing)
    seq INTEGER NOT NULL,
    -- Standard message fields
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    -- Timestamp: server-assigned on receipt (authoritative)
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Tool use: full JSON of any tool_use/tool_result blocks
    -- Solves the "recall searches not in transcripts" bug
    tool_blocks JSONB,
    -- Thinking blocks (if extended thinking enabled)
    thinking TEXT,
    -- Client-provided metadata (device, input method, etc.)
    metadata JSONB DEFAULT '{}',
    -- For partial/recovery messages
    partial BOOLEAN NOT NULL DEFAULT FALSE,
    -- Unique constraint prevents duplicate writes from retry logic
    UNIQUE (conversation_id, seq)
);

-- Indexes for common access patterns
CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
    ON messages(conversation_id, seq);

CREATE INDEX IF NOT EXISTS idx_messages_timestamp
    ON messages(timestamp);

CREATE INDEX IF NOT EXISTS idx_conversations_channel_started
    ON conversations(channel, started_at DESC);

-- Checkpoint tracking: when was the last successful backup/export
CREATE TABLE IF NOT EXISTS conversation_checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id),
    checkpoint_type TEXT NOT NULL CHECK (checkpoint_type IN ('backup', 'export', 'repo_commit')),
    message_seq INTEGER NOT NULL,  -- up to which message this checkpoint covers
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'  -- commit hash, backup location, etc.
);

COMMIT;
