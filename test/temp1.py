from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph,MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from typing import TypedDict, Literal
from duckduckgo_search import DDGS
import time


load_dotenv()

supervisor_llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
search_llm     = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
answer_llm     = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

members = ["search_agent", "answer_agent"]
options = members + ["FINISH"]


class State(MessagesState):
    next: str


@tool
def web_search(query: str) -> str:
    """
    Search the web for current, real-time information.
    Use for: recent news, current events, live data, prices,
    weather, scores, anything that changes frequently.
    Args:
        query: the search query string
    """
    print(f"  [web_search] Searching: {query[:60]}...")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=4))
    snippets = "\n\n".join(
        f"Source: {r.get('href', '')}\n{r.get('body', '')}"
        for r in results
    )
    print(f"  [web_search] Got {len(results)} results")
    return snippets if snippets else "No results found."


SUPERVISOR_PROMPT = f"""You are a supervisor managing these workers: {members}.
 
Given the user request, decide which worker should act next.
Each worker will perform their task and report back to you.
 
ROUTING RULES:
- search_agent: call this FIRST when the query needs real-time web data
  (current events, news, prices, weather, live scores, "latest", "today")
- answer_agent: call this to produce the FINAL answer for the user
  (always call this last — it synthesises everything and responds to the user)
- FINISH: respond with this ONLY after answer_agent has given its response
 
WORKFLOW:
- Query needs search: search_agent → answer_agent → FINISH
- Query does NOT need search: answer_agent → FINISH
 
Respond with the name of the worker to act next, or FINISH.

CRITICAL RULE:
- You may call answer_agent ONLY ONCE per user query
- After answer_agent responds (you will see a message with name="answer_agent"), 
  you MUST respond with FINISH immediately
- Never call answer_agent twice"""

SEARCH_AGENT_PROMPT = """You are a web research specialist.
Search the web and return a clean, factual research brief.
Use the web_search tool. Do NOT answer the user directly — return findings only."""
 
ANSWER_AGENT_PROMPT = """You are a helpful, direct assistant who gives final answers.
If research findings are in the conversation, use them as your primary source.
If no research exists, answer from your own knowledge.
Be clear, conversational, and complete."""


class Router(TypedDict):
    """Supervisor decision — which worker to call next."""
    next: Literal["search_agent", "answer_agent", "FINISH"]


def supervisor_node(state: State) -> Command[Literal["search_agent", "answer_agent", "__end__"]]:
    print("[Supervisor] deciding...")
    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
 
    # structured output — forces supervisor to pick from the options list
    response = supervisor_llm.with_structured_output(Router).invoke(messages)
    next_step = response["next"]
    print(f"[Supervisor] routing to: {next_step}")
 
    # FINISH → end the graph
    if next_step == "FINISH":
        return Command(goto="__end__")
 
    return Command(goto=next_step)

def search_agent_node(state: State) -> Command[Literal["supervisor"]]:
    print("[search_agent] thinking...")
 
    search_llm_with_tools = search_llm.bind_tools([web_search])
    tool_node = ToolNode([web_search])
 
    # build internal mini-graph for the agent loop
    class SearchState(MessagesState):
        pass
 
    def search_model(s):
        msgs = [SystemMessage(content=SEARCH_AGENT_PROMPT)] + s["messages"]
        resp = search_llm_with_tools.invoke(msgs)
        return {"messages": [resp]}
 
    inner = StateGraph(SearchState)
    inner.add_node("model", search_model)
    inner.add_node("tools", tool_node)
    inner.add_edge(START, "model")
    inner.add_conditional_edges("model", tools_condition, {"tools": "tools", "__end__": END})
    inner.add_edge("tools", "model")
    agent = inner.compile()
 
    result = agent.invoke({"messages": state["messages"]})
 
    # extract the final text response from the agent
    final_msg = result["messages"][-1]
 
    # return result to supervisor as a HumanMessage tagged with agent name
    return Command(
        update={
            "messages": [
                AIMessage(content=final_msg.content, name="search_agent")
            ]
        },
        goto="supervisor",   # always report back to supervisor
    )

def answer_agent_node(state: State) -> Command[Literal["supervisor"]]:
    print("[answer_agent] thinking...")
 
    messages = [SystemMessage(content=ANSWER_AGENT_PROMPT)] + state["messages"]
    response = answer_llm.invoke(messages)
 
    # return final answer to supervisor as HumanMessage tagged with agent name
    return Command(
        update={
            "messages": [
                AIMessage(content=response.content, name="answer_agent")
            ]
        },
        goto="supervisor",   # always report back to supervisor
    )


def build_supervisor_system():
    builder = StateGraph(State)
 
    builder.add_node("supervisor",    supervisor_node)
    builder.add_node("search_agent",  search_agent_node)
    builder.add_node("answer_agent",  answer_agent_node)
 
    # always start at supervisor
    builder.add_edge(START, "supervisor")
 
    # workers route back to supervisor via Command — no explicit edges needed
    # supervisor routes to __end__ via Command when it returns FINISH
 
    return builder.compile()


def main():
    print("Building supervisor system...")
    graph = build_supervisor_system()
 
    print("=" * 55)
    print("  Stage 2 — Supervisor + 2 Custom Graph Agents")
    print("  Supervisor → search_agent (if needed)")
    print("             → answer_agent (always last)")
    print("=" * 55)
    print("Type 'quit' to exit\n")
 
    session_log = []
 
    while True:
        query = input("You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
 
        print()
        start = time.time()
        result = graph.invoke({"messages": [HumanMessage(content=query)]})
        elapsed = time.time() - start
 
        messages = result.get("messages", [])
 
        # final answer = last message from answer_agent
        final = next(
            (m for m in reversed(messages)
             if isinstance(m, (HumanMessage, AIMessage)) and getattr(m, "name", "") == "answer_agent"),
            None
        )
 
        if final:
            print(f"Assistant: {final.content}\n")
        print(f"  Time taken: {elapsed:.2f}s")
        print("-" * 55)
 
        # detect path from message names
        agent_calls = [
            getattr(m, "name", "")
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and getattr(m, "name", "") in members
        ]
        used_search = "search_agent" in agent_calls
        path = ["START", "supervisor"]
        if used_search:
            path.append("search_agent → supervisor")
        path += ["answer_agent → supervisor", "FINISH"]
 
        session_log.append({
            "query":        query,
            "used_search":  used_search,
            "path":         path,
            "answer_chars": len(final.content) if final else 0,
            "time_taken":   elapsed,
        })
 
    # session summary
    if not session_log:
        print("\nNo queries were made this session.")
        return
 
    print("\n" + "=" * 55)
    print("  SESSION EXECUTION SUMMARY")
    print("=" * 55)
 
    for i, entry in enumerate(session_log, 1):
        q = entry["query"]
        print(f"\nQuery {i}: \"{q[:60]}{'...' if len(q) > 60 else ''}\"")
        print(f"  Web search used : {'YES' if entry['used_search'] else 'NO'}")
        print(f"  Path taken      : {' → '.join(entry['path'])}")
        print(f"  Answer length   : {entry['answer_chars']} chars")
        print(f"  Time taken      : {entry['time_taken']:.2f}s")
 
    total        = len(session_log)
    searched     = sum(1 for e in session_log if e["used_search"])
    not_searched = total - searched
    avg_time     = sum(e["time_taken"] for e in session_log) / total
 
    print(f"\n{'─' * 55}")
    print(f"  Total queries     : {total}")
    print(f"  Went to search    : {searched}  ({round(searched/total*100)}%)")
    print(f"  Direct answer     : {not_searched}  ({round(not_searched/total*100)}%)")
    print(f"  Avg response time : {avg_time:.2f}s")
    print("=" * 55)
 
 
if __name__ == "__main__":
    main()
 