from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition


from langchain_mcp_adapters.client import MultiServerMCPClient

from agent.config import (
    climate_llm, tech_llm, geopolitics_llm,
    CLIMATE_SYSTEM_PROMPT, TECH_SYSTEM_PROMPT, GEOPOLITICS_SYSTEM_PROMPT,
)
from agent.state import ClimateState, TechState, GeopoliticsState
from agent.config import CLIMATE_MCP_URL, TECH_MCP_URL, GEOPOLITICS_MCP_URL


async def build_climate_graph():

    climate_client = MultiServerMCPClient(
        {
            "Climate_Server": {
                "transport": "sse",
                "url": CLIMATE_MCP_URL,
            }
        }
    )
    climate_tools = await climate_client.get_tools()
    climate_llm_with_tools = climate_llm.bind_tools(tools=climate_tools)

    async def climate_model(state: ClimateState):
        messages = [SystemMessage(content=CLIMATE_SYSTEM_PROMPT)] + state["messages"]
        output = await climate_llm_with_tools.ainvoke(messages)
        return {"messages": [output]}

    climate_tool_node = ToolNode(climate_tools)

    builder = StateGraph(ClimateState)
    builder.add_node("climate_api", climate_model)
    builder.add_node("climate_mcp", climate_tool_node)
    builder.add_edge(START, "climate_api")
    builder.add_conditional_edges(
        "climate_api", tools_condition, {"tools": "climate_mcp", "__end__": END}
    )
    builder.add_edge("climate_mcp", "climate_api")

    return builder.compile()


async def build_tech_graph():

    tech_client = MultiServerMCPClient(
        {
            "TECH_SERVER": {
                "transport": "sse",
                "url": TECH_MCP_URL,
            }
        }
    )

    tech_tools = await tech_client.get_tools()
    tech_llm_with_tools = tech_llm.bind_tools(tools=tech_tools)

    async def tech_model(state: TechState):
        messages = [SystemMessage(content=TECH_SYSTEM_PROMPT)] + state["messages"]
        output = await tech_llm_with_tools.ainvoke(messages)
        return {"messages": [output]}

    tech_tool_node = ToolNode(tech_tools)

    builder = StateGraph(TechState)
    builder.add_node("tech_api", tech_model)
    builder.add_node("tech_mcp", tech_tool_node)
    builder.add_edge(START, "tech_api")
    builder.add_conditional_edges(
        "tech_api", tools_condition, {"tools": "tech_mcp", "__end__": END}
    )
    builder.add_edge("tech_mcp", "tech_api")

    return builder.compile()


async def build_geopolitics_graph():

    geopolitics_client = MultiServerMCPClient(
        {
            "News_Server": {
                "transport": "sse",
                "url": GEOPOLITICS_MCP_URL,
            }
        }
    )

    geopolitics_tools = await geopolitics_client.get_tools()
    geopolitics_llm_with_tools = geopolitics_llm.bind_tools(tools=geopolitics_tools)

    async def geopolitics_model(state: GeopoliticsState):
        messages = [SystemMessage(content=GEOPOLITICS_SYSTEM_PROMPT)] + state["messages"]
        output = await geopolitics_llm_with_tools.ainvoke(messages)
        return {"messages": [output]}

    geopolitics_tool_node = ToolNode(geopolitics_tools)

    builder = StateGraph(GeopoliticsState)
    builder.add_node("geopolitics_api", geopolitics_model)
    builder.add_node("geopolitics_mcp", geopolitics_tool_node)
    builder.add_edge(START, "geopolitics_api")
    builder.add_conditional_edges(
        "geopolitics_api", tools_condition, {"tools": "geopolitics_mcp", "__end__": END}
    )
    builder.add_edge("geopolitics_mcp", "geopolitics_api")

    return builder.compile()