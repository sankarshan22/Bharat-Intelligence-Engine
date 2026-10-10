from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, List, Optional, Literal


class ChatStateRequired(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ChatState(ChatStateRequired, total=False):
    research_route: str
    done: str
    domain_route: Optional[List[Literal["climate", "tech", "geopolitics"]]]
    research_results: List[str]
    kg_sufficient: bool
    kg_context: Optional[str]
    head_route: str                      # "direct" or "pipeline"
    acknowledgment: Optional[str]
    user_id: str
    summary: Optional[str]              # running conversation summary for STM
    context_token_count: int
    # sid removed — transport concern, handled by FastAPI layer


class ClimateState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class TechState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class GeopoliticsState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class TavilyState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    search_queries: List[str]
    search_results: List[str]