from agent.schemas import Entity, Relationship
from agent.config import KG_MAX_ENTITIES
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import threading

class KnowledgeGraph:
    def __init__(self, max_entities: int = KG_MAX_ENTITIES):
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self.entity_timestamps: Dict[str, str] = {}
        self.entity_importance: Dict[str, int] = defaultdict(int)
        self.max_entities = max_entities
        self._lock = threading.Lock()

    def add_entity(self, entity: Entity):
        with self._lock:
            if entity.name in self.entities:
                self.entities[entity.name].properties.update(entity.properties)
            else:
                self.entities[entity.name] = entity
            self.entity_timestamps[entity.name] = datetime.utcnow().isoformat()
            if len(self.entities) > self.max_entities:
                self._prune_old_entities()

    def add_relationship(self, relationship: Relationship):
        with self._lock:
            for existing in self.relationships:
                if (existing.source == relationship.source and
                    existing.relation == relationship.relation and
                    existing.target == relationship.target):
                    existing.metadata.update(relationship.metadata)
                    return
            self.relationships.append(relationship)
            self.entity_importance[relationship.source] += 1
            self.entity_importance[relationship.target] += 1

    def add_extraction(self, entities: List[Entity], relationships: List[Relationship]):
        for entity in entities:
            self.add_entity(entity)
        for relationship in relationships:
            self.add_relationship(relationship)

    def get_entity(self, name: str) -> Optional[Entity]:
        with self._lock:
            return self.entities.get(name)

    def get_relationships_for(self, entity_name: str) -> List[Relationship]:
        with self._lock:
            return [
                r for r in self.relationships
                if r.source == entity_name or r.target == entity_name
            ]

    def _prune_old_entities(self):
        if len(self.entities) <= self.max_entities:
            return
        print(f"[KG Pruning] KG size ({len(self.entities)}) exceeds max ({self.max_entities}), pruning...")
        entity_scores = []
        for name in self.entities:
            importance = self.entity_importance.get(name, 0)
            timestamp = datetime.fromisoformat(self.entity_timestamps[name])
            recency_hours = (datetime.utcnow() - timestamp).total_seconds() / 3600
            score = importance * 1000 - recency_hours
            entity_scores.append((name, score))
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        entities_to_keep = set(name for name, _ in entity_scores[:self.max_entities])
        entities_to_remove = [name for name in self.entities if name not in entities_to_keep]
        for name in entities_to_remove:
            del self.entities[name]
            self.entity_timestamps.pop(name, None)
            self.entity_importance.pop(name, None)
        self.relationships = [
            r for r in self.relationships
            if r.source in entities_to_keep and r.target in entities_to_keep
        ]
        print(f"[KG Pruning] Removed {len(entities_to_remove)} entities, kept {len(self.entities)}")

    def get_recent_entities(self, limit: int = 50) -> List[Entity]:
        with self._lock:
            sorted_entities = sorted(
                self.entity_timestamps.items(), key=lambda x: x[1], reverse=True
            )[:limit]
            return [self.entities[name] for name, _ in sorted_entities if name in self.entities]

    def get_important_entities(self, limit: int = 50) -> List[Entity]:
        with self._lock:
            sorted_entities = sorted(
                self.entity_importance.items(), key=lambda x: x[1], reverse=True
            )[:limit]
            return [self.entities[name] for name, _ in sorted_entities if name in self.entities]

    def get_relationships_context(self, relationships: List[Relationship]) -> str:
        lines = []
        for r in relationships:
            line = f"{r.source} --{r.relation}--> {r.target}"
            if r.metadata:
                meta_parts = []
                if r.metadata.get("date"):
                    meta_parts.append(f"date: {r.metadata['date']}")
                if r.metadata.get("source"):
                    meta_parts.append(f"source: {r.metadata['source']}")
                if r.metadata.get("context"):
                    meta_parts.append(f"context: {r.metadata['context']}")
                if meta_parts:
                    line += f" [{', '.join(meta_parts)}]"
            lines.append(line)
        return "\n".join(lines)

    def summary(self) -> str:
        with self._lock:
            return (
                f"Knowledge Graph: {len(self.entities)} entities, "
                f"{len(self.relationships)} relationships"
            )

    def __repr__(self):
        return self.summary()


class KGRegistry:
    """
    Per-user KG registry. Each user gets their own isolated KnowledgeGraph instance.
    Instantiated once as a module-level singleton, imported by app.py and ontology_graph.py.
    """

    def __init__(self):
        # { user_id: KnowledgeGraph }
        self._graphs: Dict[str, KnowledgeGraph] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> KnowledgeGraph:
        """Get or create a KG for the given user."""
        with self._lock:
            if user_id not in self._graphs:
                self._graphs[user_id] = KnowledgeGraph(max_entities=KG_MAX_ENTITIES)
                print(f"[KGRegistry] Created new KG for user {user_id}")
            return self._graphs[user_id]

    def remove(self, user_id: str):
        """Remove a user's KG from memory (e.g. on logout)."""
        with self._lock:
            self._graphs.pop(user_id, None)

    def has(self, user_id: str) -> bool:
        with self._lock:
            return user_id in self._graphs


# Module-level singleton — all per-user KGs live here
kg_registry = KGRegistry()