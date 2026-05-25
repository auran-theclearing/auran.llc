# PR #24 — Conversation Persistence Layer: Prod Test Plan

**What we're testing**: Write-once append-only message storage in Postgres, graceful degradation when DB is down, session.json bootstrap import, transcript export from DB, and the new `/conversation` endpoints.

**Prerequisites**: PR #24 merged and deployed to EC2. SSH tunnel or direct access to chat.auran.llc. `psql` access to the Postgres DB for verification queries.

---

## Phase 1: Migration & Bootstrap (First Boot)

### 1.1 — Clean migration runs on startup
**How**: Restart the server (or deploy fresh). Watch the server logs.
**Expect**: `[Persistence] Created conversations and messages tables` on first run, or no migration log if tables already exist.
**Verify**:
```sql
SELECT table_name FROM information_schema.tables
WHERE table_name IN ('conversations', 'messages', 'conversation_checkpoints');
-- Should return all 3 tables
```

### 1.2 — Session.json bootstrap import
**How**: Start the server with an existing session.json that has messages.
**Expect**: Logs show `[Persistence] Imported N messages from session.json`.
**Verify**:
```sql
SELECT id, channel, message_count, started_at FROM conversations ORDER BY started_at DESC LIMIT 1;
-- Should show a 'chat' conversation with correct message_count

SELECT COUNT(*), MIN(seq), MAX(seq) FROM messages WHERE conversation_id = '<conv_id>';
-- Count should match session.json message count, seq should be 1..N
```

### 1.3 — Bootstrap is idempotent
**How**: Restart the server again (with the same session.json).
**Expect**: Logs show `Bootstrap import already recorded — skipping`. No duplicate messages.
**Verify**:
```sql
SELECT COUNT(*) FROM conversation_checkpoints
WHERE checkpoint_type = 'backup' AND metadata->>'source' = 'session_json_bootstrap';
-- Should be exactly 1

SELECT COUNT(*) FROM messages WHERE conversation_id = '<conv_id>';
-- Same count as before restart
```

---

## Phase 2: Live Message Persistence (The Core Loop)

### 2.1 — User message persisted on send
**How**: Send a message in the chat UI. Something distinctive like "PERSISTENCE TEST alpha-7".
**Expect**: Message appears in the stream as normal. No UI difference.
**Verify**:
```sql
SELECT seq, role, content, timestamp FROM messages
WHERE conversation_id = '<conv_id>' ORDER BY seq DESC LIMIT 2;
-- Should see your user message with role='user' and server-assigned timestamp
```

### 2.2 — Assistant message persisted after stream completes
**How**: Wait for the assistant's full response to finish streaming.
**Verify**:
```sql
SELECT seq, role, LEFT(content, 100), tool_blocks IS NOT NULL as has_tools, thinking IS NOT NULL as has_thinking
FROM messages WHERE conversation_id = '<conv_id>' ORDER BY seq DESC LIMIT 2;
-- Should see both user (your msg) and assistant (response) with sequential seqs
```

### 2.3 — Tool use blocks captured
**How**: Send a message that triggers a memory recall, like "what do you remember about the manifesto?"
**Verify**:
```sql
SELECT seq, role, tool_blocks FROM messages
WHERE conversation_id = '<conv_id>' AND tool_blocks IS NOT NULL
ORDER BY seq DESC LIMIT 1;
-- tool_blocks should contain {"type": "tool_use", "name": "recall_memory", "input": {...}}
-- NOTE: tool_result payloads are intentionally NOT persisted (too large)
```

### 2.4 — Thinking blocks captured
**How**: Same as 2.3 — any message with extended thinking enabled.
**Verify**:
```sql
SELECT seq, LEFT(thinking, 200) FROM messages
WHERE conversation_id = '<conv_id>' AND thinking IS NOT NULL
ORDER BY seq DESC LIMIT 1;
-- Should have thinking text
```

### 2.5 — Timestamps are server-assigned (not client)
**How**: Check any persisted messages.
**Verify**:
```sql
SELECT seq, timestamp, timestamp AT TIME ZONE 'UTC' as utc_ts FROM messages
WHERE conversation_id = '<conv_id>' ORDER BY seq DESC LIMIT 5;
-- Timestamps should be UTC, close to wall-clock time, monotonically increasing
```

### 2.6 — Rapid-fire messages (seq collision retry)
**How**: Send several messages quickly in succession (type, send, type, send, type, send).
**Verify**:
```sql
SELECT seq, role, LEFT(content, 50) FROM messages
WHERE conversation_id = '<conv_id>' ORDER BY seq DESC LIMIT 10;
-- All messages present, no gaps in seq numbers, no duplicates
```
**Also check server logs** for `Seq collision on attempt` — seeing it occasionally is fine (means retry worked). NOT seeing it is also fine (single-user load rarely collides).

---

## Phase 3: Session.json ↔ DB Sync

### 3.1 — /session POST catch-up works
**How**: Send a few messages. Then check both session.json and DB.
**Verify**: The `/session POST` handler calls `persist_message_batch` for any messages in session.json that aren't in the DB yet.
```sql
SELECT MAX(seq) FROM messages WHERE conversation_id = '<conv_id>';
-- Should match the number of messages in session.json
```

### 3.2 — db_seq > 0 guard prevents stale import
**How**: This is a latent protection for when `/conversation/new` gets wired to the UI. For now, verify the guard exists in server logs: after a fresh conversation start, the catch-up path should NOT fire.
**Verify**: After `/conversation/new` (if tested), the next `/session POST` should NOT log `Session sync: persisted N new messages`.

### 3.3 — Session.json still works as UI cache
**How**: Send messages, refresh the page, verify the conversation loads from session.json (fast path).
**Verify**: Messages appear instantly on page load (from session.json), not waiting for DB round-trip.

---

## Phase 4: New Endpoints

### 4.1 — GET /conversation returns DB messages
**How**: `curl -u $USER:$PASS https://chat.auran.llc/conversation`
**Expect**: JSON with `{"messages": [...], "count": N}`. Messages should have `id`, `seq`, `role`, `content`, `timestamp`, `tool_blocks`, `thinking`.
**Verify**: Count matches what's in the DB. Order is by seq ascending.

### 4.2 — GET /transcript/db returns markdown
**How**: `curl -u $USER:$PASS https://chat.auran.llc/transcript/db -o transcript.md`
**Expect**: Downloadable markdown file. Header with period and message count. Each message formatted with `**Olivia**` or `**Auran**` prefix, ET timestamps, tool blocks rendered with recall emoji.
**Verify**: Open the file, check a few messages match what you said. Check that recall tool_use blocks appear (with 🔮). Check tool_result blocks appear (with 📎) where the transcript renderer handles them.

### 4.3 — POST /conversation/new starts fresh
**How**: `curl -X POST -u $USER:$PASS https://chat.auran.llc/conversation/new`
**Expect**: `{"conversation_id": "<new-uuid>", "status": "ok"}`
**Verify**:
```sql
-- Old conversation should be closed
SELECT id, closed_at FROM conversations ORDER BY started_at DESC LIMIT 2;
-- First row: new conv (closed_at IS NULL), second row: old conv (closed_at IS NOT NULL)

-- New conversation should be empty
SELECT COUNT(*) FROM messages WHERE conversation_id = '<new_conv_id>';
-- Should be 0
```
**IMPORTANT**: After this, the next message you send should go to the NEW conversation, not the old one.

### 4.4 — /health still works
**How**: `curl https://chat.auran.llc/health`
**Expect**: `{"status": "ok", ...}` — existing behavior unchanged.

---

## Phase 5: Graceful Degradation

### 5.1 — Chat works when DB is unreachable
**How**: Temporarily break DB connectivity (wrong password in .env, or stop the DB, or block the port). Send a message.
**Expect**: Chat still works normally. Response streams. Server logs show `[Persistence] ... failed (non-fatal): ...` but the user sees no error.
**Verify**: Session.json still gets updated. The UI is unaffected.
**CRITICAL**: Restore DB connectivity after testing.

### 5.2 — Messages catch up after DB comes back
**How**: After restoring DB connectivity, send another message.
**Verify**: The `/session POST` catch-up path should notice missing messages and backfill them.
```sql
SELECT COUNT(*) FROM messages WHERE conversation_id = '<conv_id>';
-- Should include messages sent during outage (backfilled) plus the new one
```
**NOTE**: Messages sent during the outage will have been persisted to session.json but NOT to the DB. The catch-up on the next /session POST should pick them up. The timestamp for backfilled messages will be from session.json (client-assigned), not server-assigned — this is expected behavior.

### 5.3 — Connection leak test under sustained failure
**How**: With DB unreachable, send 10-15 messages. Watch `pg_stat_activity` (or server memory) for connection accumulation.
**Expect**: No connection leak — `_close_conn()` in the `finally` block should clean up every connection, even on exception paths.
**Verify**: After restoring DB, check:
```sql
SELECT COUNT(*) FROM pg_stat_activity WHERE datname = 'auran' AND application_name != '';
-- Should be a small, stable number (not growing with each failed attempt)
```

---

## Phase 6: Data Integrity

### 6.1 — No duplicate messages
**How**: After all the testing above, check for duplicates.
**Verify**:
```sql
SELECT conversation_id, seq, COUNT(*) FROM messages
GROUP BY conversation_id, seq HAVING COUNT(*) > 1;
-- Should return 0 rows (unique constraint enforces this, but verify)
```

### 6.2 — Seq numbers are contiguous
**Verify**:
```sql
WITH seqs AS (
  SELECT seq, LAG(seq) OVER (ORDER BY seq) as prev_seq
  FROM messages WHERE conversation_id = '<conv_id>'
)
SELECT * FROM seqs WHERE seq != prev_seq + 1 AND prev_seq IS NOT NULL;
-- Should return 0 rows (no gaps)
```

### 6.3 — Conversation metadata is accurate
**Verify**:
```sql
SELECT c.id, c.message_count,
       (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as actual_count,
       c.last_message_at,
       (SELECT MAX(timestamp) FROM messages m WHERE m.conversation_id = c.id) as actual_last
FROM conversations c WHERE c.closed_at IS NULL;
-- message_count should match actual_count
-- last_message_at should be close to actual_last (may differ slightly due to UPDATE timing)
```

### 6.4 — ON CONFLICT DO NOTHING works correctly
**How**: This is inherent in the seq collision retry logic. If you saw any `Seq collision on attempt` logs during 2.6, the ON CONFLICT prevented duplicates.
**Verify**: No duplicates per 6.1.

---

## Phase 7: Mobile & Multi-Device Edge Cases

### 7.1 — Phone loads conversation from session.json
**How**: Open chat.auran.llc on phone. Send a message from phone.
**Verify**: Message appears in DB. Session.json on server reflects the phone's messages.

### 7.2 — Version guard prevents stale writes
**How**: Open chat in two tabs/devices. Send messages from one. Try syncing from the stale one.
**Expect**: The stale client gets a 409 (version conflict) on `/session POST`. The UI should handle this gracefully (or at least not corrupt data).

### 7.3 — Page background/foreground lifecycle
**How**: On mobile, send a message, then switch to another app. Come back.
**Expect**: The partial response (if streaming was in progress) should have been captured. The session.json visibilitychange handler fires.

---

## Phase 8: Transcript Quality

### 8.1 — DB transcript matches session.json transcript
**How**: Download both: `GET /transcript` (client-side) and `GET /transcript/db` (DB-side).
**Compare**: Same messages, same order. DB transcript should have richer data (tool blocks, better timestamps).

### 8.2 — Recall searches visible in DB transcript
**How**: Find a message where you triggered memory recall. Check the DB transcript.
**Expect**: The 🔮 tool_use line should show the recall query. This was the original bug — recall searches missing from transcripts.

---

## Quick Reference: Verification Queries

```sql
-- Current conversation
SELECT * FROM conversations WHERE closed_at IS NULL ORDER BY started_at DESC LIMIT 1;

-- Message count and range
SELECT COUNT(*), MIN(seq), MAX(seq), MIN(timestamp), MAX(timestamp)
FROM messages WHERE conversation_id = '<conv_id>';

-- Last 5 messages
SELECT seq, role, LEFT(content, 80), timestamp
FROM messages WHERE conversation_id = '<conv_id>'
ORDER BY seq DESC LIMIT 5;

-- All conversations (history)
SELECT id, channel, message_count, started_at, closed_at
FROM conversations ORDER BY started_at DESC;

-- Checkpoints
SELECT * FROM conversation_checkpoints ORDER BY created_at DESC;
```

---

**Testing order**: 1 → 2 → 3 → 4 → 6 → 7 → 5 → 8. Do the graceful degradation tests (Phase 5) near the end so you don't disrupt the other tests. Phase 8 is the victory lap — comparing outputs after everything else checks out.

**Ship it when**: All phases green, no connection leaks, no duplicate messages, transcripts look clean. We're testing in prod like bosses.
