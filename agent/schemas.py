
from typing import List,Optional,Dict,Any,Literal
from pydantic import BaseModel,Field

class Research_Router(BaseModel):
    need_research: bool
    domain: Optional[List[Literal["climate", "tech", "geopolitics"]]] = None
    answer: Optional[str] = None

class Entity(BaseModel):
    name: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)


class Relationship(BaseModel):
    source: str
    relation: str
    target: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OntologyExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

class HeadAgentResponse(BaseModel):
    route: Literal["direct", "pipeline"]
    answer: Optional[str] = None        # filled when route == "direct"
    acknowledgment: Optional[str] = None # filled when route == "pipeline"

class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if new, False if duplicate of existing memory")

class MemoryDecision(BaseModel):
    should_write: bool = Field(description="Whether any memories are worth storing")
    memories: List[MemoryItem] = Field(default_factory=list)