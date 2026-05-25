# Conversation Persistence — Integration Guide

## What ships

### New files
- `migrations/007_create_conversations_table.sql` — DB schema
- `persistence.py` — persistence layer (all DB operations)

### Changes to `server.py`

Three integration points (minimal, non-breaking):

#### 1. Startup: run migration + import existing session.json
```python
# At server startup (after app creation, before routes):
from persistence import run_migration, import_from_session_json, ensure_conversation

# Run migration on first boot
run_migration()

# Bootstrap: import current session.json into DB (idempotent)
if SESSION_FILE.exists():
    try:
        data = json.loads(SESSION_FILE.read_text())
        if data.get("messages"):
            imported = import_from_session_json(data)
            if imported:
                print(f"[Persistence] Imported {imported} messages from session.json")
    except Exception as e:
        print(f"[Persistence] Bootstrap import failed (non-fatal): {e}")
```

#### 2. `/chat` endpoint: persist user message on receipt, assistant message on completion
```python
# After message validation, before sending to API:
from persistence import persist_message
user_msg = messages[-1]  # the new user message
persist_message(role="user", content=user_msg["content"])

# After stream_response completes (after yield done):
# Persist the full assistant response with tool blocks
tool_blocks_for_persist = []
for tc in tool_calls:
    tool_blocks_for_persist.append({"type": "tool_use", "name": tc["name"], "input": tc["input"]})
persist_message(
    role="assistant",
    content=response_text,
    tool_blocks=tool_blocks_for_persist if tool_blocks_for_persist else None,
    thinking="".join(current_thinking_text) if current_thinking_text else None,
)
```

#### 3. `/session` POST: persist any new messages from client sync
```python
# At end of save_session, after writing session.json:
# Persist any messages the DB doesn't have yet
try:
    from persistence import get_conversation_messages, persist_message_batch
    existing_count = len(get_conversation_messages())
    new_msgs = messages[existing_count:]
    if new_msgs:
        persist_message_batch(new_msgs)
except Exception as e:
    print(f"[Persistence] Session sync failed (non-fatal): {e}")
```

### New endpoints

#### GET /conversation
```python
@app.get("/conversation")
async def get_conversation(request: Request):
    """Get full conversation from DB (source of truth)."""
    from persistence import get_conversation_messages
    messages = get_conversation_messages()
    return JSONResponse({"messages": messages, "count": len(messages)})
```

#### GET /transcript/db
```python
@app.get("/transcript/db")
async def transcript_from_db(request: Request):
    """Generate transcript from DB storage (includes tool blocks)."""
    from persistence import get_conversation_transcript
    content = get_conversation_transcript(include_tool_blocks=True)
    return Response(
        content=content.encode("utf-8"),
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="chat-transcript-db.md"'},
    )
```

#### POST /conversation/new
```python
@app.post("/conversation/new")
async def new_conversation(request: Request):
    """Start a new conversation (closes the current one)."""
    from persistence import start_new_conversation
    conv_id = start_new_conversation()
    return JSONResponse({"conversation_id": conv_id})
```

## Design decisions

1. **Write-once, append-only**: Messages are never updated or deleted from the `messages` table. This is deliberate — we lost data because of overwrites. The append-only pattern means even bugs in the save logic can't destroy existing data.

2. **Graceful degradation**: All persistence calls are wrapped in try/except. If Postgres is down, the chat still works (falls back to session.json behavior). Persistence failures log but never crash the request.

3. **Dual storage during transition**: session.json remains the fast-path UI cache. The DB is the source of truth and DR layer. Eventually session.json can be deprecated, but not yet — the client still relies on it for offline/reconnect behavior.

4. **Server-assigned timestamps**: The `timestamp` column defaults to `NOW()` at INSERT time. No more relying on client-provided timestamps that may be missing or wrong.

5. **Tool blocks in full**: The `tool_blocks` JSONB column stores the complete tool_use and tool_result structures. This directly fixes the "recall searches not in transcripts" bug — the DB transcript export renders them.

6. **ON CONFLICT DO NOTHING**: The unique constraint on `(conversation_id, seq)` prevents duplicate writes from retry logic without raising errors.

7. **Checkpoint tracking**: The `conversation_checkpoints` table records when backups and exports succeeded, so we know exactly what's covered and what isn't.

## Recovery model

- **Level 1**: Postgres messages table (primary, backed up via RDS snapshots + backup_moments.py)
- **Level 2**: session.json on EC2 (secondary cache, may be stale)
- **Level 3**: Repo exports in charting_territory/sessions/ (periodic, manual-triggered or automated)

## What this prevents

The exact scenario from May 25: messages exist in the browser, get synced to session.json, then a subsequent sync overwrites them. With DB persistence, the messages were written to Postgres the moment they arrived at the server — they can never be lost by a later session.json overwrite.
