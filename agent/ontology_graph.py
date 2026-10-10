from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
import asyncio
import time
import json
import ast
from agent.knowledge_graph import kg_registry

from agent.schemas import Research_Router, OntologyExtraction, HeadAgentResponse
from agent.config import (
    DOMAINS,
    inital_llm,
    ontology_llm,
    aggregator_llm,
    entity_extractor_llm,
    kg_sufficiency_llm,
    head_agent_llm,
    INITIAL_SYSTEM_PROMPT,
    ONTOLOGY_SYSTEM_PROMPT,
    AGGREGATOR_SYSTEM_PROMPT,
    ENTITY_EXTRACTOR_PROMPT,
    KG_SUFFICIENCY_PROMPT,
    HEAD_AGENT_PROMPT,
    CONTEXT_TPM_LIMIT,
    CONTEXT_WARN_THRESHOLD,
)
from agent.domain_graph import build_climate_graph, build_tech_graph, build_geopolitics_graph
from agent.tavily_search import build_tavily_graph
from agent.state import ChatState
from agent.stm import run_stm, build_llm_context
from agent.ltm import ltm_node, build_ltm_context


research_router_llm = inital_llm.with_structured_output(
    Research_Router, method="json_mode"
)

ontology_llm_structured = ontology_llm.with_structured_output(
    OntologyExtraction, method="json_mode"
)

head_agent_llm_structured = head_agent_llm.with_structured_output(
    HeadAgentResponse, method="json_mode"
)


async def build_graph(checkpointer=None):

    async def head_agent_node(state: ChatState):
        messages = build_llm_context(state["messages"], state.get("summary"))
        user_id = state.get("user_id", "")

        current_query = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            ""
        )
        ltm_context = build_ltm_context(user_id, current_query) if user_id else ""

        system_prompt = HEAD_AGENT_PROMPT
        if ltm_context:
            system_prompt = HEAD_AGENT_PROMPT + "\n\n" + ltm_context

        try:
            response: HeadAgentResponse = await head_agent_llm_structured.ainvoke([
                SystemMessage(content=system_prompt),
                *messages
            ])

            if response.route == "direct":
                print("[Head Agent] Direct answer — no pipeline needed.")
                return {
                    "head_route": "direct",
                    "messages": [AIMessage(content=response.answer or "")],
                }
            return {
                "head_route": "pipeline",
                "acknowledgment": response.acknowledgment or "On it — let me look into that for you.",
            }
        except Exception as e:
            print(f"[Head Agent] Error: {e} — defaulting to pipeline")
            return {
                "head_route": "pipeline",
                "acknowledgment": "Let me look into that for you.",
            }


    async def kg_check_node(state: ChatState):
        user_query = state["messages"][-1].content
        kg = kg_registry.get(state.get("user_id", ""))

        KG_SUMMARY_TRIGGERS = [
            "what do you know", "summarize your knowledge", "knowledge graph summary",
            "what have you learned", "what's in your kg", "give me the summary",
            "what do you know so far", "summary of your knowledge"
        ]
        is_kg_summary = any(t in user_query.lower() for t in KG_SUMMARY_TRIGGERS)

        if is_kg_summary:
            if not kg.entities:
                print("[KG Check] KG summary on empty KG — routing directly to aggregator.")
                return {
                    "kg_sufficient": True,
                    "kg_context": "The knowledge graph is currently empty. No data has been collected yet.",
                    "research_route": "sufficient"
                }
            else:
                all_entities = list(kg.entities.values())
                all_relationships = kg.relationships
                kg_context = "ENTITIES:\n"
                kg_context += "\n".join(f"{e.name} ({e.type})" for e in all_entities)
                kg_context += "\n\nRELATIONSHIPS:\n"
                kg_context += kg.get_relationships_context(all_relationships)
                print("[KG Check] KG summary — routing directly to aggregator with full context.")
                return {
                    "kg_sufficient": True,
                    "kg_context": kg_context,
                    "research_route": "sufficient"
                }

        if not kg.entities:
            print("[KG Check] KG is empty — routing to research.")
            return {"kg_sufficient": False, "research_route": "research"}

        all_entity_names = list(kg.entities.keys())

        if len(all_entity_names) > 100:
            print(f"[KG Check] Large KG detected ({len(all_entity_names)} entities), using sampling")
            entity_sample = all_entity_names[:100]
        else:
            entity_sample = all_entity_names

        entity_response = await entity_extractor_llm.ainvoke([
            SystemMessage(content=ENTITY_EXTRACTOR_PROMPT),
            HumanMessage(content=f"User Query: {user_query}\n\nAvailable entities: {entity_sample}")
        ])
        try:
            raw_names = entity_response.content.strip().replace("```python", "").replace("```", "").strip()
            relevant_names = ast.literal_eval(raw_names)
        except Exception as e:
            print(f"[KG Check] Entity extraction failed: {e}, defaulting to empty")
            relevant_names = []

        if not relevant_names:
            print("[KG Check] No relevant entities — routing to research.")
            return {"kg_sufficient": False, "research_route": "research"}

        subgraph_entities = [kg.get_entity(n) for n in relevant_names if kg.get_entity(n)]
        subgraph_relationships = []
        seen = set()
        for name in relevant_names:
            for r in kg.get_relationships_for(name):
                key = (r.source, r.relation, r.target)
                if key not in seen:
                    seen.add(key)
                    subgraph_relationships.append(r)

        max_entities = 50
        max_relationships = 100

        if len(subgraph_entities) > max_entities:
            print(f"[KG Check] Truncating entities from {len(subgraph_entities)} to {max_entities}")
            subgraph_entities = subgraph_entities[:max_entities]

        if len(subgraph_relationships) > max_relationships:
            print(f"[KG Check] Truncating relationships from {len(subgraph_relationships)} to {max_relationships}")
            subgraph_relationships = subgraph_relationships[:max_relationships]

        kg_context = "ENTITIES:\n"
        kg_context += "\n".join(f"{e.name} ({e.type})" for e in subgraph_entities)
        kg_context += "\n\nRELATIONSHIPS:\n"
        kg_context += kg.get_relationships_context(subgraph_relationships)

        sufficiency_response = await kg_sufficiency_llm.ainvoke([
            SystemMessage(content=KG_SUFFICIENCY_PROMPT),
            HumanMessage(content=f"User Query: {user_query}\n\nKnowledge Graph:\n{kg_context}")
        ])

        try:
            raw = sufficiency_response.content.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
        except Exception as e:
            print(f"[KG Check] Sufficiency check failed: {e}, routing to research.")
            return {"kg_sufficient": False, "research_route": "research"}

        sufficient = result.get("sufficient", False)
        missing_domains = result.get("missing_domains", [])
        reason = result.get("reason", "")
        print(f"[KG Check] Sufficient: {sufficient} | Missing: {missing_domains} | Reason: {reason}")

        if sufficient:
            return {"kg_sufficient": True, "kg_context": kg_context, "research_route": "sufficient"}
        elif missing_domains:
            return {"kg_sufficient": False, "kg_context": kg_context, "domain_route": missing_domains, "research_route": "partial"}
        else:
            return {"kg_sufficient": False, "research_route": "research"}


    async def intial_chat(state: ChatState):
        current_query = state["messages"][-1].content
        research_route = state.get("research_route", "research")

        if research_route == "partial" and state.get("domain_route"):
            already_missing = state["domain_route"]
            partial_hint = (
                f"\n\nNOTE: The knowledge graph already has partial data for this query. "
                f"The following domains have been identified as missing or incomplete: {already_missing}. "
                f"Only route to these domains — do not re-fetch domains that already have data."
            )
        else:
            partial_hint = ""

        try:
            response = await research_router_llm.ainvoke([
                INITIAL_SYSTEM_PROMPT,
                HumanMessage(content=current_query + partial_hint)
            ])

            if research_route == "partial" and state.get("domain_route"):
                print(f"[Initial Chat] Partial hit — using KG-identified missing domains: {state['domain_route']}")
                return {
                    "research_route": "partial",
                    "domain_route": state["domain_route"],
                }
            return {
                "research_route": "research",
                "domain_route": response.domain or [d["name"] for d in DOMAINS],
            }
        except Exception as e:
            print(f"[Initial Chat] Error during routing: {e}")
            return {
                "research_route": research_route,
                "domain_route": state.get("domain_route") or [d["name"] for d in DOMAINS],
            }


    # ── NOTE: Domain subgraphs are NOT built at startup. ──
    # They are built lazily inside research_node on first invocation.
    # This avoids BlockingError from MCP stdio subprocess spawning
    # during ASGI server startup (caught by blockbuster in langgraph dev).
    _graph_cache: dict = {}

    async def research_node(state: ChatState):
        domains = state["domain_route"] or [d["name"] for d in DOMAINS]
        query = state["messages"][-1].content

        # ── Lazy build: build each subgraph once, cache it for reuse ──
        if "tavily" not in _graph_cache:
            try:
                print("[Research] Building tavily graph (first run)...")
                _graph_cache["tavily"] = await build_tavily_graph()
            except Exception as e:
                print(f"[Research] Failed to build tavily graph: {e}")

        if "climate" in domains and "climate" not in _graph_cache:
            try:
                print("[Research] Building climate graph (first run)...")
                _graph_cache["climate"] = await build_climate_graph()
            except Exception as e:
                print(f"[Research] Failed to build climate graph: {e}")

        if "tech" in domains and "tech" not in _graph_cache:
            try:
                print("[Research] Building tech graph (first run)...")
                _graph_cache["tech"] = await build_tech_graph()
            except Exception as e:
                print(f"[Research] Failed to build tech graph: {e}")

        if "geopolitics" in domains and "geopolitics" not in _graph_cache:
            try:
                print("[Research] Building geopolitics graph (first run)...")
                _graph_cache["geopolitics"] = await build_geopolitics_graph()
            except Exception as e:
                print(f"[Research] Failed to build geopolitics graph: {e}")

        async def run_tavily():
            if "tavily" not in _graph_cache:
                return AIMessage(content="Tavily agent encountered an error: Tavily graph unavailable")
            try:
                result = await _graph_cache["tavily"].ainvoke(
                    {"messages": [HumanMessage(content=query)]}
                )
                return result["messages"][-1]
            except Exception as e:
                print(f"[Research] Tavily error: {e}")
                return AIMessage(content=f"Tavily agent encountered an error: {str(e)}")

        async def run_climate():
            if "climate" not in _graph_cache:
                return AIMessage(content="Climate agent encountered an error: Climate MCP server unavailable")
            try:
                result = await _graph_cache["climate"].ainvoke(
                    {"messages": [HumanMessage(content=query)]},
                    config={"recursion_limit": 25}
                )
                return result["messages"][-1]
            except Exception as e:
                print(f"[Research] Climate error: {e}")
                return AIMessage(content=f"Climate agent encountered an error: {str(e)}")

        async def run_tech():
            if "tech" not in _graph_cache:
                return AIMessage(content="Tech agent encountered an error: Tech MCP server unavailable")
            try:
                result = await _graph_cache["tech"].ainvoke(
                    {"messages": [HumanMessage(content=query)]},
                    config={"recursion_limit": 20}
                )
                return result["messages"][-1]
            except Exception as e:
                print(f"[Research] Tech error: {e}")
                return AIMessage(content=f"Tech agent encountered an error: {str(e)}")

        async def run_geopolitics():
            if "geopolitics" not in _graph_cache:
                return AIMessage(content="Geopolitics agent encountered an error: Geopolitics MCP server unavailable")
            try:
                result = await _graph_cache["geopolitics"].ainvoke(
                    {"messages": [HumanMessage(content=query)]},
                    config={"recursion_limit": 20}
                )
                return result["messages"][-1]
            except Exception as e:
                print(f"[Research] Geopolitics error: {e}")
                return AIMessage(content=f"Geopolitics agent encountered an error: {str(e)}")

        tasks = []
        if "climate" in domains:
            tasks.append(run_climate())
        if "tech" in domains:
            tasks.append(run_tech())
        if "geopolitics" in domains:
            tasks.append(run_geopolitics())

        # Tavily always runs alongside domain agents
        tasks.append(run_tavily())

        results = await asyncio.gather(*tasks)
        return {
            "research_results": [r.content for r in results],
        }


    async def ontology_node(state: ChatState):
        research_results = state.get("research_results", [])
        kg = kg_registry.get(state.get("user_id", ""))

        if not research_results:
            print("[Ontology] No research results to process.")
            return {}

        combined = "\n\n---\n\n".join(research_results)

        max_research_chars = 15000
        if len(combined) > max_research_chars:
            print(f"[Ontology] Truncating research from {len(combined)} to {max_research_chars} chars")
            combined = combined[:max_research_chars] + "\n\n[... content truncated due to length ...]"

        print(f"[Ontology] Extracting from {len(research_results)} research result(s)...")

        try:
            extraction: OntologyExtraction = await ontology_llm_structured.ainvoke(
                [
                    SystemMessage(content=ONTOLOGY_SYSTEM_PROMPT),
                    HumanMessage(content=f"Research Results:\n{combined}"),
                ]
            )

            for entity in extraction.entities:
                entity.properties["domain"] = "_".join(state.get("domain_route") or [])

            kg.add_extraction(extraction.entities, extraction.relationships)

            print(f"[Ontology] {kg.summary()}")
            print("\n[Ontology] Entities:")
            for name, entity in list(kg.entities.items())[:10]:
                print(f"  - [{entity.type}] {name} | {entity.properties}")
            if len(kg.entities) > 10:
                print(f"  ... and {len(kg.entities) - 10} more entities")

            print("\n[Ontology] Relationships:")
            for r in kg.relationships[:10]:
                print(f"  - {r.source} --{r.relation}--> {r.target}")
            if len(kg.relationships) > 10:
                print(f"  ... and {len(kg.relationships) - 10} more relationships")

        except Exception as e:
            print(f"[Ontology] Extraction failed: {e}")

        return {"research_results": []}


    async def aggregator_node(state: ChatState):
        user_query = next(
            m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)
        )
        kg = kg_registry.get(state.get("user_id", ""))

        if state.get("kg_context") and state.get("kg_sufficient"):
            kg_context = state["kg_context"]
            print("[Aggregator] Using cached KG context.")
        else:
            all_entity_names = list(kg.entities.keys())

            if len(all_entity_names) > 100:
                entity_sample = all_entity_names[:100]
            else:
                entity_sample = all_entity_names

            entity_response = await entity_extractor_llm.ainvoke([
                SystemMessage(content=ENTITY_EXTRACTOR_PROMPT),
                HumanMessage(content=f"User Query: {user_query}\n\nAvailable entities: {entity_sample}")
            ])
            try:
                raw_names = entity_response.content.strip().replace("```python", "").replace("```", "").strip()
                relevant_names = ast.literal_eval(raw_names)
            except Exception:
                relevant_names = all_entity_names[:20]

            print(f"[Aggregator] Relevant entities: {relevant_names[:10]}...")

            subgraph_entities = [kg.get_entity(n) for n in relevant_names if kg.get_entity(n)]
            subgraph_relationships = []
            seen = set()
            for name in relevant_names:
                for r in kg.get_relationships_for(name):
                    key = (r.source, r.relation, r.target)
                    if key not in seen:
                        seen.add(key)
                        subgraph_relationships.append(r)

            max_entities = 50
            max_relationships = 100

            if len(subgraph_entities) > max_entities:
                subgraph_entities = subgraph_entities[:max_entities]
            if len(subgraph_relationships) > max_relationships:
                subgraph_relationships = subgraph_relationships[:max_relationships]

            print(f"[Aggregator] Subgraph: {len(subgraph_entities)} entities, {len(subgraph_relationships)} relationships")

            kg_context = "ENTITIES:\n"
            kg_context += "\n".join(f"{e.name} ({e.type})" for e in subgraph_entities)
            kg_context += "\n\nRELATIONSHIPS:\n"
            kg_context += kg.get_relationships_context(subgraph_relationships)

        try:
            full_content = ""
            async for chunk in aggregator_llm.astream([
                SystemMessage(content=AGGREGATOR_SYSTEM_PROMPT),
                HumanMessage(content=f"User Query: {user_query}\n\n{kg_context}")
            ]):
                if chunk.content:
                    full_content += chunk.content
            return {"messages": [AIMessage(content=full_content)]}
        except Exception as e:
            print(f"[Aggregator] Error: {e}")
            return {"messages": [AIMessage(content=f"I encountered an error while aggregating the knowledge: {str(e)}")]}


    async def summarizer_node(state: ChatState):
        """
        STM summarization node — transport-agnostic.
        No sid or socketio. The FastAPI layer detects summarization
        by watching for the 'summarizer_node' on_chain_start event in the stream.
        """
        messages = state["messages"]
        existing_summary = state.get("summary")

        new_summary, _, token_count = await run_stm(
            messages=messages,
            existing_summary=existing_summary,
        )

        if new_summary:
            print("[STM] Summarization complete. Summary stored.")
            return {
                "summary": new_summary,
                "context_token_count": token_count,
            }

        return {
            "context_token_count": token_count,
        }


    # ── Graph Build ──
    builder = StateGraph(ChatState)

    def head_agent_condition(state: ChatState):
        return state.get("head_route", "pipeline")

    def kg_check_condition(state: ChatState):
        route = state.get("research_route", "research")
        if route == "sufficient":
            return "sufficient"
        if route == "partial":
            return "partial"
        return "research"

    def should_summarize(state: ChatState):
        messages = state.get("messages", [])
        total = count_tokens_approximately(messages)
        threshold = CONTEXT_TPM_LIMIT * CONTEXT_WARN_THRESHOLD
        return "summarize" if total >= threshold else "skip"

    builder.add_node("head_agent_node", head_agent_node)
    builder.add_node("ltm_node", ltm_node)
    builder.add_node("kg_check_node", kg_check_node)
    builder.add_node("first_call", intial_chat)
    builder.add_node("research_node", research_node)
    builder.add_node("ontology_node", ontology_node)
    builder.add_node("aggregator_node", aggregator_node)
    builder.add_node("summarizer_node", summarizer_node)

    builder.add_edge(START, "head_agent_node")
    builder.add_conditional_edges(
        "head_agent_node", head_agent_condition,
        {"direct": "ltm_node", "pipeline": "kg_check_node"}
    )
    builder.add_conditional_edges(
        "kg_check_node", kg_check_condition,
        {"sufficient": "aggregator_node", "partial": "first_call", "research": "first_call"}
    )
    builder.add_edge("first_call", "research_node")
    builder.add_edge("research_node", "ontology_node")
    builder.add_edge("ontology_node", "aggregator_node")
    builder.add_conditional_edges(
        "aggregator_node",
        should_summarize,
        {"summarize": "summarizer_node", "skip": "ltm_node"}
    )
    builder.add_edge("summarizer_node", "ltm_node")
    builder.add_edge("ltm_node", END)

    # ── Checkpointer ──
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# ── Local dev entrypoint ──
async def main():
    from langgraph.checkpoint.memory import InMemorySaver

    config = {"configurable": {"thread_id": "1"}}

    # Inject InMemorySaver only for local testing
    async def build_graph_local():
        g = await build_graph()
        # Re-compile with local checkpointer for dev
        return g

    graph = await build_graph_local()

    while True:
        question = input("Human: ")
        start = time.time()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(question)]}, config=config
        )
        elapsed = time.time() - start
        last_message = result["messages"][-1]
        if result.get("research_route") == "research":
            domains = result.get("domain_route", [])
            prefix = " & ".join(d.capitalize() for d in domains) + " Agent: "
        else:
            prefix = "AI: "

        print(prefix, last_message.content)
        print(f"\nResponse time: {elapsed:.2f}s")
        repeat = input("Done?(Yes/No) ")
        if repeat.lower() == "yes":
            break

    print("\n\nMessages Stored")
    snap = graph.get_state(config)
    vals = snap.values
    for m in vals.get("messages", []):
        print("-", type(m).__name__, ":", m.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())