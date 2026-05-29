"""
Neo4j graph recall for chat.auran.llc

Provides relational depth to complement pgvector semantic recall.
pgvector answers "what memories are similar to this query?"
Neo4j answers "what memories are connected to what I just found?"

The graph was populated by the roam agent's dual-write pipeline
(neo4j_writer.py → neo4j-agent-memory SDK). Every roam memory gets:
- A Message node with content, embedding, and metadata
- Entity nodes (POLE+O: Person, Object, Location, Event, Organization)
- Relationship edges (MENTIONS, RELATED_TO, etc.)

This module reads the graph. It never writes to it.

Connection: bolt://neo4j.auran.local:7687 (Cloud Map service discovery,
same VPC as the chat server on ECS). Falls back to env vars for local dev
with SSM tunnel.
"""

import logging
import os
import threading

logger = logging.getLogger("auran-chat.graph_recall")

# ---------------------------------------------------------------------------
# Neo4j driver management — lazy singleton, same pattern as Voyage client
# ---------------------------------------------------------------------------

_neo4j_driver = None
_neo4j_init_attempted = False
_neo4j_init_lock = threading.Lock()
_neo4j_config_cache = None  # Cache resolved config to avoid Secrets Manager on retries


def _get_neo4j_config() -> dict | None:
    """Get Neo4j connection config.

    Priority:
    1. NEO4J_URI + NEO4J_PASSWORD env vars (local dev with SSM tunnel)
    2. Secrets Manager (prod — ECS task in same VPC)
    3. None (graph recall disabled)
    """
    # Env vars first (local dev)
    uri = os.getenv("NEO4J_URI")
    password = os.getenv("NEO4J_PASSWORD")
    if uri and password:
        return {
            "uri": uri,
            "user": os.getenv("NEO4J_USER", "neo4j"),
            "password": password,
        }

    # Secrets Manager fallback (prod)
    try:
        import boto3

        sm = boto3.client("secretsmanager", region_name="us-east-1")
        secret = sm.get_secret_value(SecretId="auran/neo4j-password")
        raw = secret["SecretString"]

        # The secret is stored as "neo4j/PASSWORD" format
        if "/" in raw:
            user, password = raw.split("/", 1)
        else:
            user, password = "neo4j", raw

        return {
            "uri": "bolt://neo4j.auran.local:7687",
            "user": user,
            "password": password,
        }
    except Exception as e:
        logger.info(f"Neo4j config not available: {e}")
        return None


def _get_driver():
    """Get or create Neo4j driver. Returns None if not configured.

    Thread-safe singleton. Sets _neo4j_init_attempted AFTER success or
    permanent failure (ImportError, no config), NOT after transient failures
    (DNS hiccup, Neo4j cold-starting). This allows retry on transient issues
    without hammering a permanently absent Neo4j.
    """
    global _neo4j_driver, _neo4j_init_attempted, _neo4j_config_cache

    if _neo4j_init_attempted:
        return _neo4j_driver

    with _neo4j_init_lock:
        if _neo4j_init_attempted:
            return _neo4j_driver

        # Cache config to avoid Secrets Manager calls on transient retries
        if _neo4j_config_cache is None:
            _neo4j_config_cache = _get_neo4j_config()
        config = _neo4j_config_cache
        if not config:
            _neo4j_init_attempted = True  # Permanent: no config available
            logger.info("Neo4j not configured — graph recall disabled")
            return None

        try:
            from neo4j import GraphDatabase

            _neo4j_driver = GraphDatabase.driver(
                config["uri"],
                auth=(config["user"], config["password"]),
                # Connection pool settings tuned for read-heavy, low-concurrency
                max_connection_pool_size=5,
                connection_acquisition_timeout=2,
            )
            # Verify connectivity
            _neo4j_driver.verify_connectivity()
            _neo4j_init_attempted = True  # Permanent: successfully connected
            logger.info(f"Neo4j driver connected: {config['uri']}")
            return _neo4j_driver
        except ImportError:
            _neo4j_init_attempted = True  # Permanent: driver not installed
            logger.warning("neo4j driver not installed — graph recall disabled")
            return None
        except Exception as e:
            # Transient: DON'T set _neo4j_init_attempted — allow retry
            logger.warning(f"Neo4j connection failed (will retry next request): {e}")
            _neo4j_driver = None
            return None


def graph_available() -> bool:
    """Check if Neo4j graph recall is available."""
    return _get_driver() is not None


# ---------------------------------------------------------------------------
# Graph queries — Cypher read-only
# ---------------------------------------------------------------------------


def _resolve_embedding(text: str, precomputed_embedding: list[float] | None = None) -> list[float] | None:
    """Get an embedding as list[float], reusing precomputed if available.

    Avoids both duplicate Voyage API calls and the brittle pgvector string
    parsing that coupling to generate_embedding()'s string format requires.
    """
    if precomputed_embedding is not None:
        return precomputed_embedding

    # Fallback: generate fresh (used by recall_graph tool, not orient path)
    from memory import generate_embedding, parse_embedding_string

    embedding_str = generate_embedding(text)
    if not embedding_str:
        return None
    return parse_embedding_string(embedding_str)


# Per-query timeout (seconds) — bounds Cypher execution, not just pool acquisition.
# Prevents a wide two-hop traversal on a hub entity from stalling the orient path.
GRAPH_QUERY_TIMEOUT_S = 3


def find_connected_entities(text: str, limit: int = 5, precomputed_embedding: list[float] | None = None) -> list[dict]:
    """Find entities mentioned in memories similar to the given text.

    Uses the existing vector index on Message nodes (created by
    neo4j-agent-memory) to find semantically similar messages, then
    traverses MENTIONS edges to find connected entities.

    Args:
        text: Query text for semantic search.
        limit: Max entities to return.
        precomputed_embedding: Reuse an already-generated Voyage embedding
            (list[float]) to avoid duplicate API calls on the orient path.

    Returns entity dicts with name, type, and the messages that mention them.
    """
    driver = _get_driver()
    if not driver:
        return []

    try:
        embedding = _resolve_embedding(text, precomputed_embedding)
        if not embedding:
            return []

        with driver.session(database="neo4j") as session:
            with session.begin_transaction(timeout=GRAPH_QUERY_TIMEOUT_S) as tx:
                result = tx.run(
                    """
                    CALL db.index.vector.queryNodes('message_embedding_idx', $k, $embedding)
                    YIELD node AS msg, score
                    WHERE score >= $threshold
                    WITH msg, score
                    MATCH (msg)-[r:MENTIONS]->(e)
                    WHERE e:Entity OR e:Person OR e:Object OR e:Location
                          OR e:Event OR e:Organization
                    RETURN e.name AS name,
                           labels(e) AS labels,
                           e.description AS description,
                           collect(DISTINCT {
                               content: left(msg.content, 200),
                               score: score,
                               role: msg.role
                           }) AS mentions
                    ORDER BY size(mentions) DESC, max(score) DESC
                    LIMIT $limit
                    """,
                    embedding=embedding,
                    k=10,
                    threshold=0.3,
                    limit=limit,
                )
                return [dict(record) for record in result]

    except Exception as e:
        logger.warning(f"find_connected_entities failed: {e}")
        return []


def find_related_memories(text: str, limit: int = 5, precomputed_embedding: list[float] | None = None) -> list[dict]:
    """Find memories connected to the query through shared entities.

    Two-hop traversal: query → similar messages → shared entities → other messages.
    This surfaces memories that are relationally connected but may not be
    semantically similar — the whole point of graph over vector.

    Args:
        text: Query text for semantic search.
        limit: Max related memories to return.
        precomputed_embedding: Reuse an already-generated Voyage embedding
            (list[float]) to avoid duplicate API calls on the orient path.

    Example: query about "Marcel" returns memories about temperature,
    about Olivia's desk, about care-at-a-distance — connected through
    the Marcel entity node, not through semantic similarity.
    """
    driver = _get_driver()
    if not driver:
        return []

    try:
        embedding = _resolve_embedding(text, precomputed_embedding)
        if not embedding:
            return []

        with driver.session(database="neo4j") as session:
            with session.begin_transaction(timeout=GRAPH_QUERY_TIMEOUT_S) as tx:
                result = tx.run(
                    """
                    CALL db.index.vector.queryNodes('message_embedding_idx', $k, $embedding)
                    YIELD node AS seed, score AS seed_score
                    WHERE seed_score >= $threshold
                    WITH seed, seed_score
                    MATCH (seed)-[:MENTIONS]->(entity)
                    WHERE entity:Entity OR entity:Person OR entity:Object
                          OR entity:Location OR entity:Event OR entity:Organization
                    WITH entity, max(seed_score) AS via_score
                    MATCH (entity)<-[:MENTIONS]-(related)
                    WHERE related:Message
                    WITH related, entity, via_score
                    ORDER BY via_score DESC
                    WITH related,
                         collect(DISTINCT entity.name) AS via_entities,
                         max(via_score) AS relevance
                    RETURN related.content AS content,
                           related.role AS role,
                           via_entities,
                           relevance
                    ORDER BY relevance DESC
                    LIMIT $limit
                    """,
                    embedding=embedding,
                    k=8,
                    threshold=0.35,
                    limit=limit,
                )
                return [dict(record) for record in result]

    except Exception as e:
        logger.warning(f"find_related_memories failed: {e}")
        return []


def get_entity_neighborhood(entity_name: str, limit: int = 8) -> dict | None:
    """Get an entity and its full neighborhood — related entities and memories.

    Used by the recall_graph tool for explicit entity exploration.
    """
    driver = _get_driver()
    if not driver:
        return None

    try:
        with driver.session(database="neo4j") as session:
            # Find the entity by name (fuzzy)
            entity_result = session.run(
                """
                MATCH (e)
                WHERE (e:Entity OR e:Person OR e:Object OR e:Location
                       OR e:Event OR e:Organization)
                  AND toLower(e.name) CONTAINS toLower($name)
                RETURN e.name AS name, labels(e) AS labels,
                       e.description AS description, elementId(e) AS eid
                LIMIT 1
                """,
                name=entity_name,
            )
            entity = entity_result.single()
            if not entity:
                return None

            # Get connected entities
            related_entities = session.run(
                """
                MATCH (e) WHERE elementId(e) = $eid
                MATCH (e)-[r]-(other)
                WHERE NOT other:Message AND NOT other:Conversation
                RETURN other.name AS name, labels(other) AS labels,
                       type(r) AS relationship
                LIMIT $limit
                """,
                eid=entity["eid"],
                limit=limit,
            )

            # Get memories mentioning this entity
            memories = session.run(
                """
                MATCH (e) WHERE elementId(e) = $eid
                MATCH (msg)-[:MENTIONS]->(e)
                WHERE msg:Message
                RETURN msg.content AS content,
                       msg.role AS role
                ORDER BY msg.timestamp DESC
                LIMIT $limit
                """,
                eid=entity["eid"],
                limit=limit,
            )

            return {
                "entity": dict(entity),
                "related_entities": [dict(r) for r in related_entities],
                "memories": [dict(m) for m in memories],
            }

    except Exception as e:
        logger.warning(f"get_entity_neighborhood failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Format graph results for system prompt injection
# ---------------------------------------------------------------------------


def format_graph_context(
    entities: list[dict],
    related_memories: list[dict],
) -> str:
    """Format graph retrieval results for system prompt injection.

    Designed to complement, not duplicate, the pgvector recall section.
    Entities provide structural context (WHO/WHAT/WHERE is connected).
    Related memories surface relationally-linked content that vector
    search might miss.
    """
    sections = []

    if entities:
        entity_lines = []
        for e in entities[:5]:
            labels = [lbl for lbl in (e.get("labels") or []) if lbl not in ("Entity", "BaseNode")]
            type_str = labels[0] if labels else "Entity"
            mention_count = len(e.get("mentions") or [])
            name = e.get("name", "unknown")
            desc = e.get("description") or ""
            line = f"- **{name}** ({type_str})"
            if desc:
                line += f" — {desc[:120]}"
            line += f" [{mention_count} memory connections]"
            entity_lines.append(line)
        sections.append("## Graph context (entity connections)\n" + "\n".join(entity_lines))

    if related_memories:
        # Deduplicate and format
        seen_content = set()
        memory_lines = []
        for m in related_memories[:5]:
            content = (m.get("content") or "")[:200]
            if content in seen_content or not content:
                continue
            seen_content.add(content)
            via = ", ".join(m.get("via_entities") or [])
            role = m.get("role") or "memory"
            line = f"- ({role}) {content}"
            if via:
                line += f"\n  *Connected via: {via}*"
            memory_lines.append(line)
        if memory_lines:
            sections.append(
                "## Related memories (graph traversal)\n"
                "*These surfaced through entity connections, not semantic similarity.*\n" + "\n".join(memory_lines)
            )

    return "\n\n".join(sections)
