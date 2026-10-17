import asyncio
from typing import Annotated, TypedDict
from yaspin import yaspin

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    BaseMessage,
)
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

WEATHER_CODES = {
    0: "快晴",
    1: "晴れ",
    2: "晴れ時々曇り",
    3: "曇り",
    45: "霧",
    48: "霧",
    51: "弱い霧雨",
    53: "霧雨",
    55: "強めの霧雨",
    61: "弱い雨",
    63: "雨",
    65: "強い雨",
    80: "弱いにわか雨",
    81: "にわか雨",
    82: "強いにわか雨",
    95: "雷雨",
}


@tool
def get_time() -> str:
    """現在の時刻を返す"""
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


@tool
def get_weather() -> str:
    """現在の東京の天気と気温を調べる。"""
    import requests

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 35.6762,
        "longitude": 139.6503,
        "current": "temperature_2m,weather_code",
        "timezone": "Asia/Tokyo",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        cur = response.json()["current"]
        code = cur["weather_code"]
        desc = WEATHER_CODES.get(code, f"不明(コード{code})")
        return f"現在の東京の天気： {desc}、気温： {cur['temperature_2m']}℃"
    except Exception as e:
        return f"現在の東京の天気は取得できませんでした：{e}"

llm = ChatOllama(model="qwen3.5:9b", temperature=0, reasoning=False)

async def time_worker(query: str) -> str:

    WORKER_SYSTEM_PROMPT = """あなたは時間担当エージェントです。
    ユーザーの質問に対して、 get_time ツールを使って回答してください。
    - 時刻に関する質問には必ず get_time ツールを使うこと。
    - 十分な情報が得られたら、日本語で簡潔に回答すること。
    """
    class SubState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    async def call_model(state: SubState) -> dict:
        messages = [SystemMessage(content=WORKER_SYSTEM_PROMPT)] + state["messages"]
        llm_with_tool = llm.bind_tools([get_time])
        with yaspin(text="[TIME_WORKER]") as spinner:
            response = await llm_with_tool.ainvoke(messages)
        return {"messages": [response]}

    def build_graph():
        graph_builder = StateGraph(SubState)
        graph_builder.add_node("agent", call_model)
        graph_builder.add_node("tools", ToolNode([get_time]))

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "agent")
        return graph_builder.compile()
    
    async def main(query):
        graph = build_graph()
        turn_input = {"messages": [HumanMessage(content=query)]}
        result = await graph.ainvoke(turn_input)

        return result["messages"][-1].content
    return await main(query)

async def weather_worker(query: str) -> str:

    WORKER_SYSTEM_PROMPT = """あなたは天気担当エージェントです。
    ユーザーの質問に対して、 get_weather ツールを使って回答してください。
    - 天気に関する質問には必ず get_weather ツールを使うこと。
    - 十分な情報が得られたら、日本語で簡潔に回答すること。
    """
    class SubState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    async def call_model(state: SubState) -> dict:
        messages = [SystemMessage(content=WORKER_SYSTEM_PROMPT)] + state["messages"]
        llm_with_tool = llm.bind_tools([get_weather])
        with yaspin(text="[WEATHER_WORKER]") as spinner:
            response = await llm_with_tool.ainvoke(messages)
        return {"messages": [response]}

    def build_graph():
        graph_builder = StateGraph(SubState)
        graph_builder.add_node("agent", call_model)
        graph_builder.add_node("tools", ToolNode([get_weather]))

        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "agent")
        return graph_builder.compile()
    
    async def main(query):
        graph = build_graph()
        turn_input = {"messages": [HumanMessage(content=query)]}
        result = await graph.ainvoke(turn_input)

        return result["messages"][-1].content
    return await main(query)


@tool
async def time_tool(query: str) -> str:
    """現在時刻を取得するエージェント

    Args:
        query: 時刻に関する質問
    """
    print(f"[SUPERVISOR] {query} -> [TIME_WORKER]")
    response = await time_worker(query)
    print(f"[TIME_WORKER] {response} -> [SUPERVISOR]")
    return response

@tool
async def weather_tool(query: str) -> str:
    """現在の東京の天気を取得するエージェント。

    Args:
        query: 東京の天気に関する質問
    """
    print(f"[SUPERVISOR] {query} -> [WEATHER_WORKER]")
    response = await weather_worker(query)
    print(f"[WEATHER_WOKER] {response} -> [SUPERVISOR]")
    return response

tools = [time_tool, weather_tool]
llm_with_tools = llm.bind_tools(tools)

class MainState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

MAIN_SYSTEM_PROMPT = """あなたは有能なアシスタントです。
ユーザーの質問に関して、 get_time / get_weather ツールを使って適切に回答してください。

- 現在時刻に関する質問は必ず get_time を使うこと。
- 現在の東京の天気に関する質問は必ず get_weather を使うこと。
- ツールへの質問は一つにまとめず、複数回に分けて一つ一つ調べること。
- 十分な情報が得られたら、日本語で分かりやすく回答すること。
"""


async def call_model(state: MainState) -> dict:
    messages = [SystemMessage(content=MAIN_SYSTEM_PROMPT)] + state["messages"]
    with yaspin(text="[SUPERVISOR]") as spinner:
        response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}

def build_graph():
    graph_builder = StateGraph(MainState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile()

async def main():
    query = "現在の時刻と東京の天気を教えて"
    graph = build_graph()
    turn_input = {"messages": [HumanMessage(content=query)]}
    print(f"[USER] {query} -> [SUPERVISOR]")
    result = await graph.ainvoke(turn_input)
    print("[SUPERVISOR] " + result["messages"][-1].content + f" -> [USER]")

if __name__ == "__main__":
    asyncio.run(main())
