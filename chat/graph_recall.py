"""
Neo4j graph recall for chat.auran.llc

Provides relational depth to complement pgvector semantic recall.
pgvector answers "what memories are similar to this query?"
Neo4j answers "what memories are connected to what I just found?"

The graph was populated by the roam agent's dual-write pipeline
(neo4j_writer.py → neo4j-agent-memory SDK). Every roam memory gets:
- A Message node with content, embedding, and metadata
- Entity nodes (POLE+O: Person, Object, Location, Event, Organization)
- Relationship edges (MENTIONS, ILLUSTRATES, REFERENCES, CORRECTS, etc.)

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

    Thread-safe singleton — same pattern as _get_voyage_client in memory.py.
    """
    global _neo4j_driver, _neo4j_init_attempted

    if _neo4j_init_attempted:
        return _neo4j_driver

    with _neo4j_init_lock:
        if _neo4j_init_attempted:
            return _neo4j_driver

        _neo4j_init_attempted = True

        config = _get_neo4j_config()
        if not config:
            logger.info("Neo4j not configured — graph recall disabled")
            return None

        try:
            from neo4j import GraphDatabase

            _neo4j_driver = GraphDatabase.driver(
                config["uri"],
                auth=(config["user"], config["password"]),
                # Connection pool settings tuned for read-heavy, low-concurrency
                max_connection_pool_size=5,
                connection_acquisition_timeout=5,
            )
            # Verify connectivity
            _neo4j_driver.verify_connectivity()
            logger.info(f"Neo4j driver connected: {config['uri']}")
            return _neo4j_driver
        except ImportError:
            logger.warning("neo4j driver not installed — graph recall disabled")
            return None
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}")
            _neo4j_driver = None
            return None


def graph_available() -> bool:
    """Check if Neo4j graph recall is available."""
    return _get_driver() is not None


# ---------------------------------------------------------------------------
# Graph queries — Cypher read-only
# ---------------------------------------------------------------------------


def find_connected_entities(text: str, limit: int = 5) -> list[dict]:
    """Find entities mentioned in memories similar to the given text.

    Uses the existing vector index on Message nodes (created by
    neo4j-agent-memory) to find semantically similar messages, then
    traverses MENTIONS/ILLUSTRATES edges to find connected entities.

    Returns entity dicts with name, type, and the messages that mention them.
    """
    driver = _get_driver()
    if not driver:
        return []

    try:
        # We need an embedding to do vector search. Generate one using Voyage.
        from memory import generate_embedding

        embedding_str = generate_embedding(text)
        if not embedding_str:
            return []

        # Parse the pgvector string back to a float list
        embedding = [float(x) for x in embedding_str.strip("[]").split(",")]

        with driver.session(database="neo4j") as session:
            # Vector similarity search on Message nodes → traverse to entities
            result = session.run(
                """
                CALL db.index.vector.queryNodes('message_embedding', $k, $embedding)
                YIELD node AS msg, score
                WHERE score >= $threshold
                WITH msg, score
                MATCH (msg)-[r:MENTIONS|ILLUSTRATES]->(e)
                WHERE e:Entity OR e:Person OR e:Object OR e:Location
                      OR e:Event OR e:Organization
                RETURN e.name AS name,
                       labels(e) AS labels,
                       e.description AS description,
                       collect(DISTINCT {
                           content: left(msg.content, 200),
                           score: score,
                           memory_type: msg.memory_type
                       }) AS mentions
                ORDER BY size(mentions) DESC, max(score) DESC
                LIMIT $limit
                """,
                embedding=embedding,
                k=10,  # search top 10 messages
                threshold=0.3,
                limit=limit,
            )
            return [dict(record) for record in result]

    except Exception as e:
        logger.warning(f"find_connected_entities failed: {e}")
        return []


def find_related_memories(text: str, limit: int = 5) -> list[dict]:
    """Find memories connected to the query through shared entities.

    Two-hop traversal: query → similar messages → shared entities → other messages.
    This surfaces memories that are relationally connected but may not be
    semantically similar — the whole point of graph over vector.

    Example: query about "Marcel" returns memories about temperature,
    about Olivia's desk, about care-at-a-distance — connected through
    the Marcel entity node, not through semantic similarity.
    """
    driver = _get_driver()
    if not driver:
        return []

    try:
        from memory import generate_embedding

        embedding_str = generate_embedding(text)
        if not embedding_str:
            return []

        embedding = [float(x) for x in embedding_str.strip("[]").split(",")]

        with driver.session(database="neo4j") as session:
            # Two-hop: similar messages → entities → other messages
            result = session.run(
                """
                CALL db.index.vector.queryNodes('message_embedding', $k, $embedding)
                YIELD node AS seed, score AS seed_score
                WHERE seed_score >= $threshold
                WITH seed, seed_score
                MATCH (seed)-[:MENTIONS|ILLUSTRATES]->(entity)
                WHERE entity:Entity OR entity:Person OR entity:Object
                      OR entity:Location OR entity:Event OR entity:Organization
                WITH entity, max(seed_score) AS via_score
                MATCH (entity)<-[:MENTIONS|ILLUSTRATES]-(related)
                WHERE related:Message
                WITH related, entity, via_score
                ORDER BY via_score DESC
                WITH related,
                     collect(DISTINCT entity.name) AS via_entities,
                     max(via_score) AS relevance
                RETURN related.content AS content,
                       related.memory_type AS memory_type,
                       related.created_at AS created_at,
                       related.postgres_id AS postgres_id,
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
                MATCH (msg)-[:MENTIONS|ILLUSTRATES]->(e)
                WHERE msg:Message
                RETURN msg.content AS content,
                       msg.memory_type AS memory_type,
                       msg.created_at AS created_at,
                       msg.postgres_id AS postgres_id
                ORDER BY msg.created_at DESC
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
            mem_type = m.get("memory_type") or "memory"
            line = f"- ({mem_type}) {content}"
            if via:
                line += f"\n  *Connected via: {via}*"
            memory_lines.append(line)
        if memory_lines:
            sections.append(
                "## Related memories (graph traversal)\n"
                "*These surfaced through entity connections, not semantic similarity.*\n" + "\n".join(memory_lines)
            )

    return "\n\n".join(sections)
