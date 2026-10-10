from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import asyncio
import ast

from agent.state import TavilyState
from agent.config import tavily_client, query_gen_llm, TAVILY_QUERY_GEN_PROMPT, TAVILY_REASONING_PROMPT


async def build_tavily_graph():

    async def query_generator(state: TavilyState):
        user_query = state["messages"][-1].content

        # ── CHANGED: agentic query generation — LLM reasons about what queries best answer the user's need ──
        response = await query_gen_llm.ainvoke([
            SystemMessage(content=TAVILY_QUERY_GEN_PROMPT),
            HumanMessage(content=user_query)
        ])

        raw = response.content.strip()
        # strip markdown fences if present
        raw = raw.replace("```python", "").replace("```", "").strip()
        try:
            queries = ast.literal_eval(raw)
        except Exception:
            queries = [user_query]

        print(f"[Tavily] Generated {len(queries)} search queries: {queries}")
        return {"search_queries": queries}

    async def tavily_search(state: TavilyState):
        queries = state["search_queries"]

        async def search_one(query: str) -> str:
            # ── CHANGED: increased max_results to 4 per query for richer coverage ──
            if not tavily_client:
                return f"Query: {query}\nResults:\nTavily client not initialized."
            result = await tavily_client.search(query=query, max_results=4)
            snippets = [r.get("content", "") for r in result.get("results", []) if r.get("content")]
            return f"Query: {query}\nResults:\n" + "\n".join(snippets)

        results = await asyncio.gather(*[search_one(q) for q in queries])
        return {"search_results": list(results)}

    async def reasoning_node(state: TavilyState):
        user_query = state["messages"][-1].content
        search_results = "\n\n---\n\n".join(state["search_results"])

        # ── CHANGED: reasoning node uses strategic analyst prompt — synthesizes, validates, contextualizes ──
        response = await query_gen_llm.ainvoke([
            SystemMessage(content=TAVILY_REASONING_PROMPT),
            HumanMessage(content=f"User Query: {user_query}\n\nSearch Results:\n{search_results}")
        ])

        return {"messages": [AIMessage(content=response.content)]}

    builder = StateGraph(TavilyState)
    builder.add_node("query_generator", query_generator)
    builder.add_node("tavily_search", tavily_search)
    builder.add_node("reasoning_node", reasoning_node)

    builder.add_edge(START, "query_generator")
    builder.add_edge("query_generator", "tavily_search")
    builder.add_edge("tavily_search", "reasoning_node")
    builder.add_edge("reasoning_node", END)

    return builder.compile()


async def main():
    graph = await build_tavily_graph()
    output = await graph.ainvoke({"messages": [HumanMessage(content=input("Human: "))]})
    print(output["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())