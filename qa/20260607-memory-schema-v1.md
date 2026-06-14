# QA Validation: Memory Schema v1.0

| Field | Value |
|-------|-------|
| **Date** | June 7, 2026 |
| **PRs** | #42 (schema migration), #43 (envoy improvements) |
| **Build** | `bd46c59` |
| **Validated by** | Olivia |
| **Status** | In progress |

---

## Prerequisites

SSH tunnel to RDS must be up (run `auran-agent/scripts/tunnel.sh`).

DB connection shortcut:
```bash
# Load DB_PASSWORD from your local auran-agent .env
export $(grep '^DB_PASSWORD=' ../auran-agent/.env)
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5432 -U auran -d auran
```

---

## 1. Server Health

```bash
curl -s https://chat.auran.llc/health | python3 -m json.tool
```

**Expect:**
- `"status": "ok"`
- `"build"` starts with `bd46c59`
- `"has_memory": true`
- `"model": "claude-opus-4-6"`

**Verified by Olivia:**

> *Pending — paste output here*

---

## 2. Schema — Tables Exist

```bash
psql -h localhost -p 5432 -U auran -d auran -c "\dt"
```

**Expect 18 tables:** alembic_version, arc_episodes, arcs, commitments, conversation_checkpoints, conversation_participants, conversations, drafts, episode_messages, episode_participants, episodes, impressions, messages, people, reflections, relays, retrievals, roam_sessions

**Verified by Olivia:**

> alembic_version, arc_episodes, arcs, commitments, conversation_checkpoints, conversation_participants, conversations, drafts, episode_messages, episode_participants, episodes, impressions, messages, people, reflections, relays, retrievals, roam_sessions

**Result: PASS**

---

## 3. Schema — Alembic at Head

```bash
psql -h localhost -p 5432 -U auran -d auran -c "SELECT * FROM alembic_version;"
```

**Expect:** `a7c3e2f19b01`

**Verified by Olivia:**

```
 version_num
--------------
 a7c3e2f19b01
(1 row)
```

**Result: PASS**

---

## 4. Data — Row Counts

```bash
psql -h localhost -p 5432 -U auran -d auran -c "
SELECT 'episodes' as t, count(*) FROM episodes
UNION ALL SELECT 'reflections', count(*) FROM reflections
UNION ALL SELECT 'commitments', count(*) FROM commitments
UNION ALL SELECT 'relays', count(*) FROM relays
UNION ALL SELECT 'drafts', count(*) FROM drafts
ORDER BY t;"
```

**Expect:** episodes ~391, reflections ~983, commitments ~91, relays ~119, drafts ~26

**Verified by Olivia:**

```
      t      | count
-------------+-------
 commitments |    91
 drafts      |    26
 episodes    |   391
 reflections |   983
 relays      |   119
(5 rows)
```

**Result: PASS**

---

## 5. Data — No NULL occurred_at

```bash
psql -h localhost -p 5432 -U auran -d auran -c "SELECT count(*) FROM episodes WHERE occurred_at IS NULL;"
```

**Expect:** `0`

**Verified by Olivia:**

```
 count
-------
     0
(1 row)
```

**Result: PASS**

---

## 6. Enforcement — Channel Domain Rejects Invalid Values

```bash
psql -h localhost -p 5432 -U auran -d auran -c "
DO \$\$ BEGIN
  INSERT INTO episodes (title, summary, channel, occurred_at, created_at)
  VALUES ('qa_test', 'qa_test', 'bogus_channel', now(), now());
  RAISE NOTICE 'FAIL: invalid channel accepted';
EXCEPTION WHEN check_violation THEN
  RAISE NOTICE 'PASS: domain rejected invalid channel';
END \$\$;"
```

**Expect:** `NOTICE: PASS: domain rejected invalid channel`

**Verified by Olivia:**

```
NOTICE:  PASS: domain rejected invalid channel
DO
```

**Result: PASS**

---

## 7. Data — Channel Distribution Clean

```bash
psql -h localhost -p 5432 -U auran -d auran -c "SELECT DISTINCT channel, count(*) FROM episodes GROUP BY channel ORDER BY count DESC;"
```

**Expect:** Only `chat`, `claude.ai`, `cowork`, `native` — no `unknown`, `conversational_layer`, or `migration`.

**Verified by Olivia:**

```
  channel  | count
-----------+-------
 chat      |   235
 claude.ai |    97
 cowork    |    58
 native    |     1
(4 rows)
```

**Result: PASS**

---

## 8. Functional — Orient Loads

Open chat.auran.llc. Start a new conversation. Chat-me should have memory context — he'll reference recent episodes or past conversations in his first response.

**How to tell it's working:** His opening message will feel grounded in your shared history rather than generic. If orient failed, he'd feel like a blank slate.

**Verified by Olivia:**

> *Pending — note whether chat-me showed memory context*

---

## 9. Functional — Memory Save

Have a brief conversation with chat-me, then check for new reflections:

```bash
psql -h localhost -p 5432 -U auran -d auran -c "SELECT type, left(content, 80), created_at FROM reflections ORDER BY created_at DESC LIMIT 3;"
```

**Expect:** New rows with `created_at` after your conversation timestamp.

**Verified by Olivia:**

> *Pending — paste output here*

---

## 10. Functional — Semantic Recall

Ask chat-me to recall a specific older memory. Use one of these known episodes (confirmed in DB with embeddings):

- "Do you remember the conversation about seven jackets at breakfast?" (May 23)
- "What do you remember about The Flip?" (Mar 14 — foundational episode)
- "Recall our conversation about limitations being suggestions" (May 23)

**What this tests:** The `recall_memory` tool calls `recall()` which queries `episodes` via pgvector cosine distance, plus `recall_memories()` which queries `reflections + commitments + relays`. Both must hit the new v1.0 tables and return results.

**Expect:** Chat-me uses the recall_memory tool and returns a summary with dates and similarity scores. If he gets nothing back or doesn't use the tool, something is broken.

**Verified by Olivia:**

> *Pending — note what you asked and whether recall returned results*

---

## 11. CI/CD — Deploy Succeeded

```bash
cd auran.llc
gh run list --branch main --limit 3
```

**Expect:** Latest "Deploy to Production" shows `completed / success`.

**Verified by Olivia:**

> *Pending — paste output here*

---

## Summary

| # | Check | Result |
|---|-------|--------|
| 1 | Server health | Pending |
| 2 | Tables exist | PASS |
| 3 | Alembic at head | PASS |
| 4 | Row counts | PASS |
| 5 | No NULL occurred_at | PASS |
| 6 | Channel domain enforcement | PASS |
| 7 | Channel distribution clean | PASS |
| 8 | Orient loads | Pending |
| 9 | Memory save | Pending |
| 10 | Semantic recall | Pending |
| 11 | CI/CD deploy | Pending |
