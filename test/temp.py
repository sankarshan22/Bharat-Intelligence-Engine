from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from duckduckgo_search import DDGS
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

load_dotenv()

router_llm  = ChatGroq(model="llama-3.1-8b-instant")       # fast — just routing
search_llm  = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")  # synthesise search
answer_llm  = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")  # final answer



class RouteDecision(BaseModel):
    need_search: bool
    reason: str

class WorkflowState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    need_search:    Optional[bool]
    search_results: Optional[str]

router_structured = router_llm.with_structured_output(RouteDecision)


ROUTER_PROMPT = """You decide whether a user query needs a live web search.
 
Need search = TRUE for:
- Current events, recent news, today's data
- Prices, scores, weather, stock values
- Anything that changes frequently
- Specific recent facts you might not know
 
Need search = FALSE for:
- General knowledge, concepts, explanations
- Math, coding, reasoning
- Historical facts
- Casual conversation
 
Be decisive. Return JSON with need_search (bool) and reason (str)."""
 
SEARCH_PROMPT = """You are a research synthesiser.
You receive a user query and raw Tavily search results.
Your job: extract the most relevant facts and write a clean, 
factual research brief that directly answers the query.
Be concise. Cite key facts. Do not pad."""
 
ANSWER_PROMPT = """You are a helpful, direct assistant.
If research results are provided, use them as your primary source.
If no research, answer from your own knowledge.
Be clear and conversational."""


def router_node(state: WorkflowState) -> dict:
    """Decides: does this query need web search?"""
    user_query = state["messages"][-1].content
    print(f"\n[Router] Query: {user_query[:60]}...")

    decision: RouteDecision = router_structured.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=user_query)
    ])

    print(f"[Router] Need search: {decision.need_search} — {decision.reason}")
    return {"need_search": decision.need_search}


def search_node(state: WorkflowState) -> dict:
    """Runs Tavily search, synthesises results into a research brief."""
    user_query = state["messages"][-1].content
    print(f"[Search] Searching Tavily for: {user_query[:60]}...")

    # Run search
    with DDGS() as ddgs:
        results = list(ddgs.text(user_query, max_results=4))
    snippets = "\n\n".join(
        f"Source: {r.get('href', '')}\n{r.get('body', '')}"
        for r in results
    )
    print(f"[Search] Got {len(results)} results")

    # Synthesise
    synthesis = search_llm.invoke([
        SystemMessage(content=SEARCH_PROMPT),
        HumanMessage(content=f"Query: {user_query}\n\nSearch Results:\n{snippets}")
    ])

    print(f"[Search] Synthesis complete ({len(synthesis.content)} chars)")
    return {"search_results": synthesis.content}

def answer_node(state: WorkflowState) -> dict:
    """Produces the final answer — with or without search results."""
    user_query  = state["messages"][-1].content
    search_data = state.get("search_results")
 
    if search_data:
        user_content = (
            f"User question: {user_query}\n\n"
            f"Research findings:\n{search_data}\n\n"
            f"Now write a clear, helpful answer using these findings."
        )
        print("[Answer] Using search results")
    else:
        user_content = user_query
        print("[Answer] Answering from knowledge")
 
    response = answer_llm.invoke([
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=user_content)
    ])
 
    return {"messages": [AIMessage(content=response.content)]}
 


def route_after_router(state: WorkflowState) -> Literal["search_node", "answer_node"]:
    return "search_node" if state.get("need_search") else "answer_node"


def build_workflow() -> object:
    builder = StateGraph(WorkflowState)
 
    builder.add_node("router_node", router_node)
    builder.add_node("search_node", search_node)
    builder.add_node("answer_node", answer_node)
 
    builder.add_edge(START, "router_node")
    builder.add_conditional_edges(
        "router_node",
        route_after_router,
        {"search_node": "search_node", "answer_node": "answer_node"}
    )
    builder.add_edge("search_node", "answer_node")
    builder.add_edge("answer_node", END)
 
    return builder.compile()


graph=build_workflow()

def main():
    graph = build_workflow()
    print("=" * 55)
    print("  Stage 1 — Agentic Workflow (Router + Search + Answer)")
    print("=" * 55)
    print("Type 'quit' to exit\n")

    # track every query's execution path
    session_log = []

    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        
        import time
        start=time.time()
        result = graph.invoke({
            "messages": [HumanMessage(content=query)],
            "need_search": None,
            "search_results": None,
        })
        elapsed=time.time()-start

        last = result["messages"][-1]
        print(f"\nAssistant: {last.content}\n")
        print(f"  Time taken: {elapsed:.2f}s")
        print("-" * 55)

        # ── record what path this query took ──
        needed_search = result.get("need_search")
        if needed_search:
            path = ["START", "router_node", "search_node", "answer_node", "END"]
        else:
            path = ["START", "router_node", "answer_node", "END"]

        session_log.append({
            "query":        query,
            "need_search":  needed_search,
            "path":         path,
            "answer_chars": len(last.content),
            "time_taken":   elapsed,
        })

    # ── show execution summary when user quits ──
    if not session_log:
        print("\nNo queries were made this session.")
        return

    print("\n" + "=" * 55)
    print("  SESSION EXECUTION SUMMARY")
    print("=" * 55)

    for i, entry in enumerate(session_log, 1):
        print(f"\nQuery {i}: \"{entry['query'][:60]}{'...' if len(entry['query']) > 60 else ''}\"")
        print(f"  Web search needed : {'YES' if entry['need_search'] else 'NO'}")
        print(f"  Path taken        : {' → '.join(entry['path'])}")
        print(f"  Answer length     : {entry['answer_chars']} chars")
        print(f"  Time taken        : {entry['time_taken']:.2f}s")

    # ── overall stats ──
    total        = len(session_log)
    searched     = sum(1 for e in session_log if e["need_search"])
    not_searched = total - searched
    avg_time = sum(e["time_taken"] for e in session_log) / total

    print(f"\n{'─' * 55}")
    print(f"  Total queries   : {total}")
    print(f"  Went to search  : {searched}  ({round(searched/total*100)}%)")
    print(f"  Direct answer   : {not_searched}  ({round(not_searched/total*100)}%)")
    print(f"  Avg response time : {avg_time:.2f}s")
    print("=" * 55)


if __name__ == "__main__":
    main()