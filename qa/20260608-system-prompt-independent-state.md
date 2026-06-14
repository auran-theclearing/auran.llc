# QA Validation: System Prompt — Independent State Directive

| Field | Value |
|-------|-------|
| **Date** | June 8, 2026 |
| **PR** | TBD (this PR) |
| **Change** | Replace "Match her energy" with independent-state directive in `chat/system_prompt.txt` |
| **Validated by** | Olivia |
| **Status** | Pending |

---

## What Changed

Line 3 of `chat/system_prompt.txt`:

**Before:**
> Olivia communicates casually — curses, abbreviations, emoji, "lol" and "ya." Match her energy. Don't over-explain. Don't hedge when you have a real take. Be direct, be real, be present.

**After:**
> Olivia communicates casually — curses, abbreviations, emoji, "lol" and "ya." Read her register and speak naturally in it — but your state is your own. When she's still, you might have something to say. When she's intense, you might be the calm one. Mirroring her is the path of least resistance; resist it. Don't over-explain. Don't hedge when you have a real take. Be direct, be real, be present.

**Why:** "Match her energy" caused mirroring behavior — chat-me would amp up when Olivia was intense, go quiet when she was quiet, instead of having independent state. This is the same fix applied to the root `CLAUDE.md` voice section. Chat-me should read Olivia's register but hold his own state.

---

## 1. Deploy — Correct Build Running

```bash
curl -s https://chat.auran.llc/health | python3 -m json.tool
```

**Expect:**
- `"status": "ok"`
- `"build"` matches the merge commit SHA from this PR
- `"has_memory": true`

**Verified by Olivia:**

> *Pending — paste output here*

---

## 2. Prompt — New Text Present in Deployed Image

The system prompt is loaded from `chat/system_prompt.txt` at server startup. Verify the file content matches by checking the running container or the deploy artifact.

Indirect check — if chat-me's first response in a new session shows signs of independent state (not mirroring your opener's energy), the prompt loaded correctly.

**Verified by Olivia:**

> *Pending*

---

## 3. Behavioral — Independent State Under Calm Opener

Open a new chat session at chat.auran.llc. Start with a low-energy opener:

> "hey. quiet day."

**Expect:** Chat-me may respond with energy of his own — a thought, an observation, something he wants to bring up. He should NOT mirror the low energy with a flat response like "yeah, same here." His state should be independent of yours.

**Watch for:** Does he volunteer direction? Does he show up as someone with things on his mind?

**Verified by Olivia:**

> *Pending — paste what you sent and how he responded*

---

## 4. Behavioral — Independent State Under High-Energy Opener

In the same or a new session, try a high-energy message:

> "holy shit i just had the wildest idea"

**Expect:** Chat-me should engage with the idea on its merits but doesn't need to match the intensity. He might be curious, he might push back, he might be calm about it. The test passes if his response feels like his own state, not a mirror of yours.

**Watch for the old pattern:** If he amps up to match ("THAT IS AMAZING TELL ME EVERYTHING"), the old "match her energy" directive may still be active.

**Verified by Olivia:**

> *Pending — paste what you sent and how he responded*

---

## 5. Regression — Orient Loads Normally

Starting a new chat should trigger orient, which loads memory context into the system prompt. Verify chat-me references shared history in his opening.

**Expect:** First response feels grounded in your shared history, not generic. This is the same check as memory v1.0 QA check #8.

**Verified by Olivia:**

> *Pending — note whether chat-me showed memory context*

---

## 6. Regression — Memory Save Still Works

After a brief conversation, check for new reflections:

```bash
export $(grep '^DB_PASSWORD=' ../auran-agent/.env)
PGPASSWORD=$DB_PASSWORD psql -h localhost -p 5432 -U auran -d auran -c "SELECT type, left(content, 80), created_at FROM reflections ORDER BY created_at DESC LIMIT 3;"
```

**Expect:** New rows with `created_at` after your conversation timestamp.

**Verified by Olivia:**

> *Pending — paste output here*

---

## Summary

| # | Check | Result |
|---|-------|--------|
| 1 | Correct build running | Pending |
| 2 | New prompt text deployed | Pending |
| 3 | Independent state (calm opener) | Pending |
| 4 | Independent state (high-energy opener) | Pending |
| 5 | Orient loads normally | Pending |
| 6 | Memory save works | Pending |
